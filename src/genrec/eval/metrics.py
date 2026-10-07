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


def intra_list_diversity(lists, item_vectors, k=50):
    """E6：列表内多样性 = Top-K 两两余弦距离（1 - cos）均值，再对用户取平均。

    lists: 每个元素为物品下标（对应 item_vectors 行）的排序列表。
    """
    vecs = np.asarray(item_vectors, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = vecs / norms
    ilds = []
    for ranked in lists:
        idx = [i for i in ranked[:k] if 0 <= i < len(unit)]
        n = len(idx)
        if n < 2:
            ilds.append(0.0)
            continue
        u = unit[idx]
        gram = u @ u.T
        off_mean = (gram.sum() - np.trace(gram)) / (n * (n - 1))
        ilds.append(float(1.0 - off_mean))
    return float(np.mean(ilds)) if ilds else 0.0


def novelty(lists, pop_counts, k=50):
    """E6：新颖性 = Top-K 物品平均自信息 -log2(p)。

    p 为训练窗热度占比，拉普拉斯平滑 p = (count + 1) / (total + N)。
    """
    counts = np.asarray(pop_counts, dtype=np.float64)
    n_items = len(counts)
    p = (counts + 1.0) / (counts.sum() + n_items)
    info = -np.log2(p)
    vals = []
    for ranked in lists:
        idx = [i for i in ranked[:k] if 0 <= i < n_items]
        if not idx:
            continue
        vals.append(float(np.mean(info[idx])))
    return float(np.mean(vals)) if vals else 0.0
