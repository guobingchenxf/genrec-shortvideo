import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config

DEFAULT_METHODS = ("pop,itemcf,gen-sid,gen-sid-v2,gen-sid-b1,gen-sid-b3,"
                   "gen-raw,gru-raw,sasrec")


def first_seen_map(df):
    g = df.dropna(subset=["timestamp"]).groupby("video_id")["timestamp"].min()
    return {int(k): float(v) for k, v in g.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    args = parser.parse_args()
    cfg = load_config()
    processed = cfg.path("paths", "processed_dir")
    raw_dir = cfg.path("paths", "raw_dir") / cfg["data"]["dir_name"]
    experiments = cfg.root / "results" / "experiments"
    manifest = json.loads(
        (processed / "manifest.json").read_text(encoding="utf-8"))
    t0 = float(manifest["time_split"]["t0"])

    big = pd.read_csv(raw_dir / "big_matrix.csv",
                      usecols=["video_id", "timestamp"],
                      dtype={"video_id": "int32", "timestamp": "float64"})
    small = pd.read_csv(raw_dir / "small_matrix.csv",
                        usecols=["video_id", "timestamp"],
                        dtype={"video_id": "int32", "timestamp": "float64"})
    fs = first_seen_map(big)
    for k, v in first_seen_map(small).items():
        if k not in fs or v < fs[k]:
            fs[k] = v

    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    n_new = sum(1 for v in video_ids if fs.get(int(v), 0.0) > t0)
    targets_raw = json.loads(
        (processed / "eval_targets.json").read_text(encoding="utf-8"))["targets"]
    targets = {int(k): [int(v) for v in vs] for k, vs in targets_raw.items()}

    out = {"note": "new = 首次曝光时间晚于训练窗截止 t0；recall 为 micro（Σ命中/Σ目标）",
           "t0": t0, "catalog_size": len(video_ids),
           "catalog_new_items": int(n_new), "results": {}}
    print(f"词表中上新物品（首次曝光 > t0）：{n_new}/{len(video_ids)}")
    print(f"{'method':12s} {'new_recall':>10s} {'old_recall':>10s} "
          f"{'new_targets':>11s} {'new_slots':>10s}")
    for m in args.methods.split(","):
        f = experiments / f"lists_{m}.npz"
        if not f.exists():
            print(f"{m:12s} (lists 存档不存在，跳过)")
            continue
        z = np.load(f)
        padded, user_ids = z["padded"], z["user_ids"]
        hit_new = tot_new = hit_old = tot_old = 0
        slots_new = slots = 0
        for i, uid in enumerate(user_ids):
            ranked = [int(v) for v in padded[i] if v >= 0]
            tg = targets.get(int(uid), [])
            tg_set = set(tg)
            for v in tg:
                if fs.get(int(v), 0.0) > t0:
                    tot_new += 1
                else:
                    tot_old += 1
            for v in ranked[:50]:
                slots += 1
                is_new = fs.get(int(v), 0.0) > t0
                if is_new:
                    slots_new += 1
                if v in tg_set:
                    if is_new:
                        hit_new += 1
                    else:
                        hit_old += 1
        r_new = hit_new / tot_new if tot_new else None
        r_old = hit_old / tot_old if tot_old else None
        share = round(slots_new / max(1, slots), 5)
        out["results"][m] = {
            "new_recall@50": r_new, "old_recall@50": r_old,
            "new_targets": tot_new, "old_targets": tot_old,
            "new_slot_share_in_top50": share}
        rs = f"{r_new:.4f}" if r_new is not None else "n/a"
        ro = f"{r_old:.4f}" if r_old is not None else "n/a"
        print(f"{m:12s} {rs:>10s} {ro:>10s} {tot_new:>11d} {share:>10.5f}")
    dst = experiments / "coldstart_analysis.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
