"""长尾 / 冷启动分桶分析：按目标视频的训练期热度分桶，计算各方法的桶内命中。

用法：python scripts/analyze_buckets.py [--methods itemcf,gen-sid,gen-sid-v2,gen-raw]
输入：results/experiments/lists_*.npz（排序列表存档，由 evaluate 生成）
      data/processed/eval_targets.json、data/processed/baseline_cache.npz（训练期热度）
输出：results/experiments/bucket_analysis.json + 终端表格

口径说明：
- 视频热度 = 大矩阵训练窗口（timestamp <= t0）内的交互次数（来自基线缓存）。
- 桶划分：[0]（训练期零曝光，冷启动物品）、[1,5)、[5,50)、[50,500)、[500,+inf)。
- 桶内 recall@50 = Σ命中 / Σ目标（micro 口径，只统计在该桶有目标的用户）；
  另报"长尾曝光占比"= Top-50 中热度 <=5 的视频占比（micro）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config

BUCKETS = [(0, 0), (1, 4), (5, 49), (50, 499), (500, None)]
BUCKET_NAMES = ["pop=0", "pop 1-4", "pop 5-49", "pop 50-499", "pop 500+"]


def bucket_of(pop_count):
    for i, (lo, hi) in enumerate(BUCKETS):
        if pop_count >= lo and (hi is None or pop_count <= hi):
            return i
    return len(BUCKETS) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",
                        default="pop,itemcf,gen-sid,gen-sid-v2,gen-raw,gru-raw")
    args = parser.parse_args()
    cfg = load_config()

    processed = cfg.path("paths", "processed_dir")
    experiments = cfg.root / "results" / "experiments"
    cache = np.load(processed / "baseline_cache.npz")
    pop = cache["pop"]
    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    index_of = {int(v): i for i, v in enumerate(video_ids)}
    targets_raw = json.loads(
        (processed / "eval_targets.json").read_text(encoding="utf-8"))["targets"]
    targets = {int(k): [int(v) for v in vs] for k, vs in targets_raw.items()}

    out = {}
    print(f"{'method':12s} " + " ".join(f"{n:>10s}" for n in BUCKET_NAMES)
          + f" {'tail_slots':>10s}")
    for m in args.methods.split(","):
        f = experiments / f"lists_{m}.npz"
        if not f.exists():
            print(f"{m:12s} (lists 存档不存在，跳过)")
            continue
        z = np.load(f)
        padded, user_ids = z["padded"], z["user_ids"]
        hit = np.zeros(len(BUCKETS))
        tot = np.zeros(len(BUCKETS))
        tail_slots = 0
        total_slots = 0
        for i, uid in enumerate(user_ids):
            ranked = [int(v) for v in padded[i] if v >= 0]
            tg = targets.get(int(uid), [])
            tg_set = set(tg)
            for v in tg:
                j = index_of.get(v)
                if j is not None:
                    tot[bucket_of(int(pop[j]))] += 1
            for rank, v in enumerate(ranked):
                j = index_of.get(v)
                if j is None:
                    continue
                b = bucket_of(int(pop[j]))
                total_slots += 1
                if int(pop[j]) <= 5:
                    tail_slots += 1
                if v in tg_set:
                    hit[b] += 1
        recall = [float(hit[b] / tot[b]) if tot[b] > 0 else None
                  for b in range(len(BUCKETS))]
        out[m] = {
            "bucket_recall@50": recall,
            "bucket_targets": [int(t) for t in tot],
            "tail_video_share_in_top50": round(tail_slots / max(1, total_slots), 4),
        }
        row = " ".join(
            f"{(f'{r:.4f}' if r is not None else 'n/a'):>10s}" for r in recall)
        print(f"{m:12s} {row} {out[m]['tail_video_share_in_top50']:>10.4f}")

    payload = {
        "note": "bucket recall is micro-averaged (sum hits / sum targets) over users",
        "buckets": BUCKET_NAMES,
        "results": out,
    }
    dst = experiments / "bucket_analysis.json"
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
