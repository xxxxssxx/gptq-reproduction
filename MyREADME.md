### 复现结果对应论文中内容

复现GPTQ论文中OPT模型的部分结果。OPT-66B量化时需要的RAM超过了120G（当前服务器的可用值），因此只选择了OPT-125M、OPT-13B、OPT-30B模型。

- 3bit量化时长（Table 2）
- 3bit量化后性能（Table 3）
- 3bit-kernel的推理加速。

除以上论文内的内容，额外观察了不同 lazy batch size 取值对量化速度和量化性能的影响。

### 文件结构
```text
gptq-reproduction/
├── README.md
├── MyREADME.md
├── opt.py
├── launch.sh
├── environment/
│   ├── pip_freeze.txt
│   ├── conda_env.yaml
│   └── nvidia_smi.txt
├── logs/
│   ├── 3bit_quantization.txt
│   ├── 3bit_cuda_kernel.txt
│   └── lazy_batch_size.txt
├── results/
│   └── opt_results.csv
└── other files and folders
```

上述是对复现GPTQ结果的主要文件结构。其中：
- MyREADME.md：本项目的复现说明文件。
- README.md：原仓库的说明文件。
- opt.py：模型量化与评测的主函数。
- launch.sh：包含该复现的全部实验的命令行。
- enviroment文件夹：包含实验的硬件和软件环境。
- logs文件夹：包含实验的原始日志。
- results文件夹：包含表格形式的实验结果。
- other files and folders：原始 GPTQ 仓库中的其他源码及辅助文件。



### 相对于原始代码的主要修改

- Lazy Batch Update Block Size 可配置化：将原代码中固定的块大小改为可调参数，以分析不同块大小对 GPTQ 量化运行时间及量化精度的影响。

- 多 GPU 设备分配优化：使模型与变量分配适应任意显存的GPU。

### 数据与随机种子

- 所有实验均采用默认种子seed=0。

- 数据来源：

    - wikitext数据集与原论文相同

    - PTB数据集：https://blog.csdn.net/qq_41185868/article/details/79129237

    - C4数据集：https://huggingface.co/datasets/allenai/c4

