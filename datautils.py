import numpy as np
import torch

HF_CACHE = '/mnt/e9d82f08-5ec9-47fa-a6fd-4b29f09d286c/xsx/cache/'

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# def get_wikitext2(nsamples, seed, seqlen, model):
#     from datasets import load_dataset
#     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train',cache_dir=HF_CACHE)
#     testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test',cache_dir=HF_CACHE)

#     from transformers import AutoTokenizer 
#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=HF_CACHE, local_files_only=True)
#     trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
#     testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

#     import random
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)  
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))
#     return trainloader, testenc

def read_wikitext_like_hf(path):
    texts = []

    with open(path, encoding='utf-8') as f:
        for row in f:
            if row.strip():
                texts.append(row)
            else:
                texts.append('')

    return texts

def get_wikitext2(nsamples, seed, seqlen, model):
    data_dir = HF_CACHE + "wikitext-2-raw"

    train_texts = read_wikitext_like_hf(
        f'{data_dir}/wiki.train.raw'
    )
    test_texts = read_wikitext_like_hf(
        f'{data_dir}/wiki.test.raw'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=False,
        cache_dir=HF_CACHE
    )

    trainenc = tokenizer(
        "\n\n".join(train_texts),
        return_tensors='pt'
    )

    testenc = tokenizer(
        "\n\n".join(test_texts),
        return_tensors='pt'
    )

    import random
    random.seed(seed)

    trainloader = []

    for _ in range(nsamples):
        i = random.randint(
            0,
            trainenc.input_ids.shape[1] - seqlen - 1
        )
        j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train',cache_dir=HF_CACHE)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation',cache_dir=HF_CACHE)

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,cache_dir=HF_CACHE)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
    'json',
    data_files={
        'train':
        '/mnt/e9d82f08-5ec9-47fa-a6fd-4b29f09d286c/xsx/cache/c4/en/c4-train.00000-of-01024.json.gz'
    },
    split='train')

    valdata = load_dataset(
        'json',
        data_files={
            'validation':
            '/mnt/e9d82f08-5ec9-47fa-a6fd-4b29f09d286c/xsx/cache/c4/en/c4-validation.00000-of-00008.json.gz'
        },
        split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,cache_dir=HF_CACHE)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train',cache_dir=HF_CACHE)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test',cache_dir=HF_CACHE)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,cache_dir=HF_CACHE)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',cache_dir=HF_CACHE
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',cache_dir=HF_CACHE
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,cache_dir=HF_CACHE)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
