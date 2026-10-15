#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./logs
mkdir -p ./3bit_checkpoint

export SMTP_HOST="smtp.qq.com"
export SMTP_USER="3031817469@qq.com"
export SMTP_PASSWORD="hksgzzigjmspddac"
export SMTP_TO="3031817469@qq.com"

notify_on_exit() {
    exit_code=$?
    trap - EXIT

    python notify_email.py "$exit_code" || \
        echo "Warning: Email notification failed."

    exit "$exit_code"
}

trap notify_on_exit EXIT

# # ==========================================
# # Part 1: 3-bit quantization
# # ==========================================
# (
#     PS4='+[$(date "+%Y-%m-%d %H:%M:%S")] '
#     set -x

#     python -u opt.py facebook/opt-125m c4
#     python -u opt.py facebook/opt-125m c4 --wbits 3 --nearest
#     python -u opt.py facebook/opt-125m c4 --wbits 3

#     python -u opt.py facebook/opt-13b c4
#     python -u opt.py facebook/opt-13b c4 --wbits 3 --nearest
#     python -u opt.py facebook/opt-13b c4 --wbits 3

#     python -u opt.py facebook/opt-30b c4
#     python -u opt.py facebook/opt-30b c4 --wbits 3 --nearest
#     python -u opt.py facebook/opt-30b c4 --wbits 3

# ) 2>&1 | tee -a ./logs/3bit_quantization.txt


# # ==========================================
# # Part 2: 3-bit CUDA kernel
# # ==========================================
# (
#     PS4='+[$(date "+%Y-%m-%d %H:%M:%S")] '
#     set -x

#     python -u opt.py facebook/opt-13b c4 --wbits 3 --save ./3bit_checkpoint/opt13b-3bit.pt
#     python -u opt.py facebook/opt-13b c4 --load ./3bit_checkpoint/opt13b-3bit.pt --benchmark 128
#     python -u opt.py facebook/opt-13b c4 --benchmark 128

#     python -u opt.py facebook/opt-30b c4 --wbits 3 --save ./3bit_checkpoint/opt30b-3bit.pt
#     python -u opt.py facebook/opt-30b c4 --load ./3bit_checkpoint/opt30b-3bit.pt --benchmark 128
#     python -u opt.py facebook/opt-30b c4 --benchmark 128

# ) 2>&1 | tee -a ./logs/3bit_cuda_kernel.txt


# ==========================================
# Part 3: Different lazy batch sizes
# ==========================================
(
    PS4='+[$(date "+%Y-%m-%d %H:%M:%S")] '
    set -x

    python -u opt.py facebook/opt-125m c4 --wbits 3 --blocksize 256
    # python -u opt.py facebook/opt-125m c4 --wbits 3 --blocksize 512

    # python -u opt.py facebook/opt-13b c4 --wbits 3 --blocksize 256
    # python -u opt.py facebook/opt-13b c4 --wbits 3 --blocksize 512
    # python -u opt.py facebook/opt-13b c4 --wbits 3 --blocksize 1024

    # python -u opt.py facebook/opt-30b c4 --wbits 3 --blocksize 256
    # python -u opt.py facebook/opt-30b c4 --wbits 3 --blocksize 512
    # python -u opt.py facebook/opt-30b c4 --wbits 3 --blocksize 1024

    python -u opt.py facebook/opt-13b c4 --wbits 3 --blocksize 32
    # python -u opt.py facebook/opt-13b c4 --wbits 3 --blocksize 64
    python -u opt.py facebook/opt-125m c4 --wbits 3 --blocksize 512

    # python -u opt.py facebook/opt-30b c4 --wbits 3 --blocksize 32
    # python -u opt.py facebook/opt-30b c4 --wbits 3 --blocksize 64

) 2>&1 | tee -a ./logs/lazy_batch_size.txt