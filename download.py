import time
from huggingface_hub import snapshot_download

while True:
    try:
        path = snapshot_download(
            repo_id="facebook/opt-66b",
            cache_dir="/mnt/e9d82f08-5ec9-47fa-a6fd-4b29f09d286c/xsx/cache/",
            resume_download=True,

            # 只下载 PyTorch GPTQ 需要的文件
            allow_patterns=[
                "pytorch_model-*.bin",
                "*.json",
                "*.txt",
            ],
        )

        print("Download completed successfully!")
        print("Snapshot path:", path)
        break

    except KeyboardInterrupt:
        # Ctrl+C 时允许你手动终止，而不是无限重试
        raise

    except Exception as e:
        print("\nDownload failed:")
        print(e)
        print("\nRetrying in 10 seconds...\n")
        time.sleep(10)