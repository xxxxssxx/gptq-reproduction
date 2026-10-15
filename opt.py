import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *
from experiment_logger import append_experiment_result, GPUMemoryMonitor
import sys


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    # 跳过一些随机初始化
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto',cache_dir='/mnt/e9d82f08-5ec9-47fa-a6fd-4b29f09d286c/xsx/cache/')
    model.seqlen = model.config.max_position_embeddings  # 模型支持的最大序列长度
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False  # 临时不启用KV Cache
    layers = model.model.decoder.layers  # 所有transformer块组成的列表

    #  把模型头部的模块搬入GPU
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )  # 设nsamples=N,seqlen=S,hidden_size=H。inputs (N, S, H）
    cache = {'i': 0, 'attention_mask': None}  # i记录已保存的校准样本数。这里的cache只是字典，并不是KV Cache。
    # 保存一个层的全部输入。

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp  # 仅捕获输入，不执行forward。
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])  # 临时把模块替换为Catcher。
    # 把layers[0]包装为Catcher，则layers[0]及以前的embedding等层都会正常forward，在layers[0] forward后截断。
    for batch in dataloader:  # 这里每个batch里只有一条样本。
        try:
            model(batch[0].to(dev))  # 这里只用了dataloader的input，没用target。attention_mask会在内部构建
        except ValueError:
            pass
    layers[0] = layers[0].module  # 把Catcher替换回去

    # 这些模块挪回cpu，清一下显存。
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']  # 第一个块在前向传播的时候，transformer内部会构建因果掩码。把这个因果掩码保存下来，后面要显式输入。

    print('Ready.')  # 此时，cache里是因果掩码，inps里是layers[0]的输入。outs为空。

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)  # 找出当前transformer块里的线性层
        gptq = {}
        for name in subset:  # 每个线性层都分配独立的GPTQ对象和量化器
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                # 预先存下name的值
                gptq[name].add_batch(inp[0].data, out.data)  # inp[0].data:该层的第一个输入张量(为了去掉batch维度)
            return tmp

        handles = []
        for name in subset:
            # 每个线性层完成前向计算时，对应的hook都会被自动调用。
            handles.append(subset[name].register_forward_hook(add_batch(name)))  # 创建hook->注册hook->append hook
            # add_batch()的作用是根据每个线性层的输入估计Hessian。
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]  # outs在DEBUG=False时不参与计算
            # 每个线性层前向后，都会执行对应的hook（也就是add_batch(name)）。
            # 每一条校准数据依次通过该层，也就是依次通过各个线性层。所以每一个线性层都要执行args.nsamples次add_batch()。
            # 逐条校准数据通过线性层而不是把校准数据堆起来是为了控制显存占用。
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                blocksize=args.blocksize, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer  # i = 0,name = 'self_attn.q_proj'->quantizers['model.decoder.layers.0.self_attn.q_proj'] = (gptq['self_attn.q_proj'].quantizer)
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]  # 产生量化后的模块的输出

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps  # 将量化后的模块的输出作为量化下一个模块时的输入。ping-pong buffer,这种写法是为了保住已分配好的两块显存。
        # inps = outs
        # outs = torch.zeros_like(inps)
        # 也可以这么写，但这样要重新释放和分配显存

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def opt_eval(model, testenc, dev):
    # perplexity评估
    # 使用单卡评估。
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen  # 把超长字符串根据模型的context length切分成样本

    use_cache = model.config.use_cache
    model.config.use_cache = False  # 不使用KV cache，因为这里不是逐token自回归生成，而是一次性forward。
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )  # （nsamples, seq_len, hidden_size）
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp  # （seq_len, hidden_size）
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    
    # 给第一个transformer block包装成Catcher，为了获取第一个transformer block的输入。
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        # 将超长序列分成多条样本，并逐条通过模型。
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)  # 获取每一条样本对应的layers[0]输入的同时，这里还会重复计算多次attention_mask
        except ValueError:
            pass
    layers[0] = layers[0].module

    # 把模型头部移回cpu，清理显存
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)  # （nsamples, seq_len, hidden_size）
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        # 逐个transformer block推理
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            # RTN 基线。
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            # 逐条输入样本，获取输出
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps  # 同样ping-pong buffer

    # 全部执行完后，inps是最后一个transformer block的输出

    # 模型尾部放到GPU
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        # 逐样本计算logits

        hidden_states = inps[i].unsqueeze(0)
        # 通过模型尾部
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)  # （1, seq_len, vocabulary_size）

        shift_logits = lm_logits[:, :-1, :].contiguous()  # （1, seq_len-1, vocabulary_size），包含对[x_2,...x_{seq_len}]的预测
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]  # （1, seq_len-1），包含[x_2,...x_{seq_len}]的真值
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # （seq_len-1, vocabulary_size，（seq_len-1）
        # reduction="mean"的情况下，loss是一个标量，是seq_len-1个token的预测nll的平均值
        neg_log_likelihood = loss.float() * model.seqlen  # 这里乘的应该是model.seqlen-1
        nlls.append(neg_log_likelihood)  # nlls里存放所有样本的nll

    # 设真实token序列x_1 x_2 x_3 x_4。语言模型做的是：p(x_2|x_1)，p(x_3|x_1,x_2)，p(x_4|x_1,x_2,x_3)
    # 假设模型对这三个真实token给出的概率分别p_1 p_2 p_3，则三个nll分别为-log p_1, -log p_2, -log_p_3
    # p越大越好。所以nll是一个越小越好的正数。

    # perplexity的定义是 ppl = exp(avg(nll)) ，是模型给出真实token概率的几何平均。
    # 设ppl=N，则模型的预测能力等价于模型从N个选择里随机选择一个作为结果。
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    ppl = ppl.item()

    print(ppl)

    model.config.use_cache = use_cache

    return ppl

# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)  # 只读取模型配置，不载入权重
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)  # 把后续创建张量的数据类型设回FP32。
    model = model.eval()

    # 这几个层不量化
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=args.faster_kernel)  # 替换为3bit层，此时权重值是0

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))  # 载入权重值
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def opt_multigpu(model, gpus):
    # 将模型分层放到多个GPU 

    if len(gpus) == 1:
        model.to(gpus[0])
        model.gpus = gpus
        return

    cache = {'mask': None}  # 保存当前GPU上的 attention_mask

    class MoveModule(nn.Module):
        # 在真正执行这个 layer 之前，确保输入已经在这个 layer 所在的 GPU 上。
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            # 把输入搬到对应设备上
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            # 把mask搬到对应设备上。维护一个cache['mask']是为了避免同一个GPU上的多个transfomer block重复复制attention mask，都复用cache['mask']就可以。
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)  # decoder layer的调用一般是layer(hidden_state, attention_mask,...)
            return tmp

    import copy

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))

    # 模型头部的embedding放到第一个gpu
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])

    # transformer layers
    for i in range(len(layers)):
        device = gpus[i // pergpu]
        layers[i] = MoveModule(layers[i].to(device))
    # 假设共12个层，8个卡。则pergpu=2。layer0,1在gpu0; layer2,3在gpu1 ... layer11,12在gpu5。
    # 先把各个transformer block分配到各个GPU，再给各个transformer block包装层MoveModule，使其在forward前先实现搬运行为

    # 最后一层所在GPU
    last_gpu = gpus[(len(layers)-1)//pergpu]

    # output相关模块放最后一个layer所在GPU
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(last_gpu)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(last_gpu)
    model.lm_head = copy.deepcopy(model.lm_head).to(last_gpu)

    model.gpus = gpus

def remove_multigpu(model):
    # opt_multigpu把模型包成了MoveModule，则每层的期望设备被固定为self.dev。
    # 尽管在eval的时候模型都挪到gpu[0]，在foward前还会执行数据搬运行为。
    # 导致模型和数据设备不同。所以要取消MoveModule的包装。
    layers = model.model.decoder.layers

    for i in range(len(layers)):
        if hasattr(layers[i], 'module'):
            layers[i] = layers[i].module

    if hasattr(model, 'gpus'):
        del model.gpus

def benchmark(model, input_ids, check=False):
    # 用KV cache 按 token 逐步执行模型，测量自回归解码的单token延迟

    # 输入搬到gpu0
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}  # KV cache
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp

    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))  
    # 给每个transformer block注册这个hook：完成前向后，清除对应的KV cache。
    # 逐层及时释放旧KV cache，目的是避免同时存在旧KV cache和新KV cache，降低显存峰值。

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)  # (1, args.benchmark)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),  # (1,1) 模拟自回归过程，只输入新token
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)  # 这里的新 KV cache是开源模型组织好的，已完成新旧KV cache的拼接。
            del out
        sync()
        import numpy as np
        median_latancy_per_token = np.median(times)
        print(f"median_latancy_per_token: {median_latancy_per_token}")
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

    return median_latancy_per_token


if __name__ == '__main__':

    from datetime import datetime

    run_start_time = datetime.now().astimezone().strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Lazy batch size for quantization; default 128'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    monitor = GPUMemoryMonitor()
    monitor.start()

    print("debug1")

    # 选择加载已量化的3-bit模型还是加载原始的OPT
    if args.load:
        model = load_quant3(args.model, args.load)
    else:
        model = get_opt(args.model)
        model.eval()

    print("debug2")

    # 这里是为了加载dataloader（量化校准数据集），testloader(测试量化后模型预测质量，计算困惑度指标)后面重新加载了
    # dataloader:一个list，里面是 (inp, tar)，两者形状都是 [1, seqlen]。target实际没有被使用到。
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    print("debug3")


    quant_time = None

    if args.wbits < 16 and not args.nearest:
        # 如果权重的目标位宽<16，且不是RTN，则使用GPTQ量化，并计时。
        torch.cuda.synchronize(DEV)
        tick = time.perf_counter()
        quantizers = opt_sequential(model, dataloader, DEV)
        torch.cuda.synchronize(DEV)
        quant_time = time.perf_counter() - tick

        print(f'Quantization time: {quant_time:.3f} s')

    print("debug4")

    median_latancy_per_token = None
    if args.benchmark:  # 测量逐token推理的耗时，只用一小段文本
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)  # 拆分模型，每张卡放一部分层
        else:
            model = model.to(DEV)

        input_ids = next(iter(dataloader))[0][:, :args.benchmark]  # 取第1个校准样本的前args.benchmark个token
        median_latancy_per_token = benchmark(model, input_ids, check=args.check)
        remove_multigpu(model)

    if args.load:
        monitor.stop()
        sys.exit(0)
    print("debug5")

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']

    ppl_results = {}
    # 字典，例如{'wikitext2': 31.12, 'ptb': 45.17, 'c4': 29.22}
    for dataset in datasets:
        # testloader 里是一batch size为1的很长的字符串。
        # 评估预测质量，要评估大量文本，需要控制显存占用。
        dataloader, testloader = get_loaders(
            dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen
        )
        print(dataset)
        ppl = opt_eval(model, testloader, DEV)
        ppl_results[dataset] = ppl

    print("debug6")

    if args.save:
        # 之前的量化以及量化评估阶段仍使用浮点张量保存量化后的数值，这里的opt_pack3才打包成3bit格式。这个代码只提供了3bit的load和pack。
        opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
        print("debug7")
        monitor.stop()
        sys.exit(0)
    
    # 记录显存
    monitor.stop()
    memory_result = monitor.result()

    result = {
    'start_time': run_start_time,
    'model': args.model,
    'RTN': args.nearest,
    'calib_dataset': args.dataset,

    'wbits': args.wbits,
    'lazy_batch_size': args.blocksize,
    # 'groupsize': args.groupsize,
    # 'nsamples': args.nsamples,
    # 'percdamp': args.percdamp,

    # 'sym': args.sym,
    # 'act_order': args.act_order,
    # 'static_groups': args.static_groups,
    # 'new_eval': args.new_eval,

    'quant_time_s': f"{quant_time:.3f}" if quant_time is not None else "",
    'median_latancy_per_token': f"{median_latancy_per_token:.3f}" if median_latancy_per_token is not None else "",

    'wikitext2_ppl': f"{ppl_results.get('wikitext2'):.3f}",
    'ptb_ppl': f"{ppl_results.get('ptb'):.3f}",
    'c4_ppl': f"{ppl_results.get('c4'):.3f}",

    'peak_per_gpu_GB': str([f"{x:.3f}" for x in memory_result['peak_per_gpu_GB']]),
    'peak_single_gpu_GB': f"{memory_result['peak_single_gpu_GB']:.3f}",
    'peak_total_gpu_GB': f"{memory_result['peak_total_gpu_GB']:.3f}",

    # 'config_json': json.dumps(vars(args))
    }

    append_experiment_result(
        './results/opt_results.csv',
        result
    )