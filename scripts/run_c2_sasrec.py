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
