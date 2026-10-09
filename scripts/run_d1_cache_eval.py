"""D1：前缀缓存解码的等价性验证 + 延迟重测。
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config
from genrec.eval import run_eval

METHODS = ["gen-sid", "gen-sid-v2", "gen-sid-b1", "gen-sid-b3"]
BEAMS = [1, 2, 5, 10, 20]


def _keep_awake():
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required)
        print(f"[d1] keep-awake: {'OK' if ok else 'FAILED'}")
    except Exception as exc:  # noqa: BLE001 - 非 Windows 平台忽略即可
        print(f"[d1] keep-awake skipped: {exc}")


def _lists(path):
    z = np.load(path)
    return z["user_ids"], [[int(v) for v in row if v >= 0]
                           for row in z["padded"]]


def main():
    _keep_awake()
    cfg = load_config()
    exp = cfg.root / "results" / "experiments"
    backup = Path.home() / "AppData" / "Local" / "Temp" / "d1_lists_backup"
    backup.mkdir(parents=True, exist_ok=True)
    names = ([f"lists_{m}.npz" for m in METHODS]
             + [f"lists_gen-sid-b3_beam{b}.npz" for b in BEAMS])
    for name in names:
        shutil.copy(exp / name, backup / name)
    print(f"[d1] backed up {len(names)} naive list archives -> {backup}")

    t0 = time.perf_counter()
    for m in METHODS:
        print(f"\n[d1] (cached) eval {m} @ beam50 @{time.strftime('%H:%M:%S')}")
        run_eval.run(cfg, [m])
    for b in BEAMS:
        print(f"\n[d1] (cached) eval gen-sid-b3 @ beam{b} "
              f"@{time.strftime('%H:%M:%S')}")
        run_eval.run(cfg, ["gen-sid-b3"], beam=b)

    eq = {}
    for name in names:
        uid_o, old = _lists(backup / name)
        uid_n, new = _lists(exp / name)
        same_users = bool((uid_o == uid_n).all())
        identical = sum(1 for a, b in zip(old, new) if a == b) if same_users \
            else 0
        eq[name] = {"users": len(old) if same_users else -1,
                    "identical": int(identical),
                    "identical_ratio": round(identical / max(1, len(old)), 6)}
    (exp / "d1_equivalence.json").write_text(
        json.dumps(eq, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[d1] 等价性：", json.dumps(eq, ensure_ascii=False))

    mt = json.loads((exp / "main_table.json").read_text(encoding="utf-8"))
    pj = json.loads((exp / "beam_pareto.json").read_text(encoding="utf-8"))
    for b in BEAMS:
        ej = json.loads(
            (exp / f"eval_gen-sid-b3_beam{b}.json").read_text(encoding="utf-8"))
        p = pj["points"][str(b)]
        p["ms_per_user_naive"] = p.get("ms_per_user")
        p["eval_wall_seconds_naive"] = p.get("eval_wall_seconds")
        p["ms_per_user"] = ej["ms_per_user"]
        p["eval_wall_seconds"] = ej["eval_wall_seconds"]
    p = pj["points"]["50"]
    p["ms_per_user_naive"] = p.get("ms_per_user")
    p["ms_per_user"] = mt["gen-sid-b3"]["ms_per_user"]
    p["eval_wall_seconds_naive"] = p.get("eval_wall_seconds")
    p["eval_wall_seconds"] = mt["gen-sid-b3"]["eval_wall_seconds"]
    pj["note"] = (pj["note"]
                  + "；D1：延迟为前缀缓存解码（cached）版本，naive 对照见 *_naive 字段")
    (exp / "beam_pareto.json").write_text(
        json.dumps(pj, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = round(time.perf_counter() - t0, 1)
    lat = {m: mt[m]["ms_per_user"] for m in METHODS}
    print(f"[d1] 主表延迟（cached, ms/用户）: {lat}")
    print(f"[d1] ALL DONE in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
