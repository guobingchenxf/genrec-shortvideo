import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config
from genrec.eval.metrics import coverage_at_k, intra_list_diversity, novelty

DEFAULT_METHODS = ("pop,itemcf,gen-sid,gen-sid-v2,gen-sid-b1,gen-sid-b3,"
                   "gen-raw,gru-raw,sasrec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config()
    processed = cfg.path("paths", "processed_dir")
    experiments = cfg.root / "results" / "experiments"

    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    index_of = {int(v): i for i, v in enumerate(video_ids)}
    feats = np.load(processed / "content_feats.npz")["feats"]
    pop = np.load(processed / "baseline_cache.npz")["pop"]

    out = {"note": "ILD@K=Top-K 两两余弦距离均值（内容特征 TF-IDF+SVD64）；"
                   "Novelty@K=平均 -log2(p)，p 为训练窗热度占比（Laplace 平滑）",
           "k": args.k, "results": {}}
    print(f"{'method':12s} {'ILD':>8s} {'Novelty':>9s} {'coverage':>9s}"
          f"   (K={args.k})")
    for m in args.methods.split(","):
        f = experiments / f"lists_{m}.npz"
        if not f.exists():
            print(f"{m:12s} (lists 存档不存在，跳过)")
            continue
        z = np.load(f)
        padded = z["padded"]
        lists_idx = [[index_of[int(v)] for v in row if v >= 0]
                     for row in padded]
        lists_vid = [[int(v) for v in row if v >= 0] for row in padded]
        ild = intra_list_diversity(lists_idx, feats, k=args.k)
        nov = novelty(lists_idx, pop, k=args.k)
        cov = coverage_at_k(lists_vid, args.k, len(video_ids))
        out["results"][m] = {"ild@k": round(ild, 4), "novelty@k": round(nov, 4),
                             "coverage@k": round(cov, 4)}
        print(f"{m:12s} {ild:>8.4f} {nov:>9.4f} {cov:>9.4f}")
    dst = experiments / "diversity_metrics.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
