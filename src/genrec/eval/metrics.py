"""排序评估指标（纯函数，便于单元测试）。

口径说明：评估为"下一时段观看集合预测"——targets 是用户在评估窗口内
真实观看过的视频集合（全观测矩阵保证其完整性）；相关度为二值。
"""

import math

import numpy as np


def recall_at_k(ranked, targets, k):
    if not targets:
        return 0.0
    hit = len(set(ranked[:k]) & targets)
    return hit / len(targets)


def ndcg_at_k(ranked, targets, k):
    if not targets:
        return 0.0
    dcg = 0.0
    for i, v in enumerate(ranked[:k]):
        if v in targets:
            dcg += 1.0 / math.log2(i + 2)
    ideal = min(k, len(targets))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def coverage_at_k(lists, k, catalog_size):
    seen = set()
    for ranked in lists:
        seen.update(ranked[:k])
    return len(seen) / catalog_size


def evaluate_lists(lists, targets_map, user_ids, catalog_size, ks=(10, 50)):
    """lists 与 user_ids 对齐；targets_map: {int user_id: set(video_ids)}。

    输出：recall@K、ndcg@10、hit_rate@K（≥1 命中用户占比）、coverage@maxK。
    """
    rec = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    ndcg = []
    for ranked, uid in zip(lists, user_ids):
        tg = targets_map.get(int(uid), set())
        for k in ks:
            rec[k].append(recall_at_k(ranked, tg, k))
            hit[k].append(1.0 if set(ranked[:k]) & tg else 0.0)
        ndcg.append(ndcg_at_k(ranked, tg, 10))
    out = {f"recall@{k}": float(np.mean(rec[k])) for k in ks}
    out.update({f"hit_rate@{k}": float(np.mean(hit[k])) for k in ks})
    out["ndcg@10"] = float(np.mean(ndcg))
    out[f"coverage@{max(ks)}"] = float(coverage_at_k(lists, max(ks), catalog_size))
    return out
