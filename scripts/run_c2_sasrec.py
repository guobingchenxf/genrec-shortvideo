"""C2 战役：训练 SASRec-lite 基线并进入两种评估协议。

- 训练：genrec.models.sasrec.run（150k 样本 × 3 epoch，采样 softmax 256 负例，CPU 4 线程）
- 稠密协议：run_eval.run(["sasrec"])（增量合并进 main_table.json）
- 标准协议：sampled.run(["sasrec"], max_users=2000)（增量合并进 sampled_protocol_n2000.json）

进程存活期间请求系统保持唤醒（Windows SetThreadExecutionState，不改系统设置）；
单个步骤失败会记录错误并继续后续步骤。
"""

import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config
from genrec.eval import run_eval, sampled
from genrec.models import sasrec as sasrec_mod


def _keep_awake():
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required)
        print(f"[c2] keep-awake: {'OK' if ok else 'FAILED'}")
    except Exception as exc:  # noqa: BLE001 - 非 Windows 平台忽略即可
        print(f"[c2] keep-awake skipped: {exc}")


def main():
    _keep_awake()
    cfg = load_config()
    torch.set_num_threads(int(cfg["models"]["sasrec"]["num_threads"]))
    t_start = time.perf_counter()

    print(f"\n[c2] train sasrec @{time.strftime('%H:%M:%S')}")
    try:
        log = sasrec_mod.run(cfg)
        print(f"[c2] train done: {log['epochs_log']}")
    except Exception as exc:  # noqa: BLE001 - 记录并继续
        print(f"[c2] train FAILED: {exc}")
        print(traceback.format_exc()[-800:])

    print(f"\n[c2] dense eval @{time.strftime('%H:%M:%S')}")
    try:
        run_eval.run(cfg, ["sasrec"])
    except Exception as exc:  # noqa: BLE001 - 记录并继续
        print(f"[c2] dense eval FAILED: {exc}")

    print(f"\n[c2] sampled eval @{time.strftime('%H:%M:%S')}")
    try:
        sampled.run(cfg, ["sasrec"], max_users=2000)
    except Exception as exc:  # noqa: BLE001 - 记录并继续
        print(f"[c2] sampled eval FAILED: {exc}")

    total = round(time.perf_counter() - t_start, 1)
    print(f"\n[c2] ALL DONE in {total:.0f}s")


if __name__ == "__main__":
    main()
