import csv
import json
import os

import torch
import time
import threading


def append_experiment_result(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


class GPUMemoryMonitor:

    def __init__(self, interval=0.05):
        self.interval = interval
        self.running = False

        self.peak_per_gpu = [
            0 for _ in range(torch.cuda.device_count())
        ]

        self.peak_total = 0


    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor
        )
        self.thread.start()


    def _monitor(self):

        while self.running:

            current_total = 0

            for i in range(torch.cuda.device_count()):

                mem = torch.cuda.memory_allocated(i)

                # 单卡峰值
                self.peak_per_gpu[i] = max(
                    self.peak_per_gpu[i],
                    mem
                )

                # 当前所有GPU总显存
                current_total += mem


            # 总显存峰值
            self.peak_total = max(
                self.peak_total,
                current_total
            )

            time.sleep(self.interval)


    def stop(self):

        self.running = False
        self.thread.join()


    def result(self):

        return {
            "peak_per_gpu_GB":
                [
                    x / 1024**3
                    for x in self.peak_per_gpu
                ],

            "peak_single_gpu_GB":
                max(self.peak_per_gpu) / 1024**3,

            "peak_total_gpu_GB":
                self.peak_total / 1024**3
        }