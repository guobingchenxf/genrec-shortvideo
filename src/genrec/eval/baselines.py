"""非神经基线：流行度与 ItemCF。

"""

import json

import numpy as np
import pandas as pd
from scipy import sparse


def build_cache(cfg):
    """构建并缓存 (pop 计数, ItemCF 相似度 CSR)。已存在则直接复用。"""
    processed = cfg.path("paths", "processed_dir")
    cache_path = processed / "baseline_cache.npz"
    if cache_path.exists():
        return cache_path

    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    t0 = float(manifest["time_split"]["t0"])
    vocab = np.load(processed / "video_vocab.npz")["video_ids"]
    n_items = len(vocab)
    index_of = {int(v): i for i, v in enumerate(vocab)}

    raw_dir = cfg.path("paths", "raw_dir") / cfg["data"]["dir_name"]
    df = pd.read_csv(
        raw_dir / "big_matrix.csv",
        usecols=["user_id", "video_id", "timestamp"],
        dtype={"user_id": "int32", "video_id": "int32", "timestamp": "float64"})
    df = df[df["timestamp"] <= t0].dropna(subset=["timestamp"])
    df["vi"] = df["video_id"].map(index_of)
    df = df.dropna(subset=["vi"])
    df["vi"] = df["vi"].astype("int32")

    pop = np.bincount(df["vi"].to_numpy(), minlength=n_items).astype(np.float32)

    cap = int(cfg["models"]["itemcf"]["max_items_per_user"])
    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    rows_list, cols_list = [], []
    for _uid, g in df.groupby("user_id", sort=False):
        arr = g["vi"].to_numpy()[-cap:]
        a = np.unique(arr)
        if len(a) < 2:
            continue
        iu, ju = np.triu_indices(len(a), 1)
        rows_list.append(a[iu])
        cols_list.append(a[ju])
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    co = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_items, n_items)).tocsr()
    co = co + co.T
    coo = co.tocoo()
    scale = np.sqrt(pop[coo.row].astype(np.float64)
                    * pop[coo.col].astype(np.float64)) + 1e-9
    sim = sparse.coo_matrix(
        (coo.data / scale, (coo.row, coo.col)),
        shape=(n_items, n_items)).tocsr()
    # 对角线置零：排除"推荐用户刚看过的同一个视频"这种平凡解（否则指标虚增）
    sim.setdiag(0.0)
    sim.eliminate_zeros()

    np.savez_compressed(
        cache_path, pop=pop,
        sim_data=sim.data, sim_indices=sim.indices, sim_indptr=sim.indptr,
        sim_shape=np.asarray(sim.shape))
    return cache_path


def load_cache(cfg):
    path = build_cache(cfg)
    z = np.load(path)
    sim = sparse.csr_matrix((z["sim_data"], z["sim_indices"], z["sim_indptr"]),
                            shape=tuple(z["sim_shape"]))
    return {"pop": z["pop"], "sim": sim}


def pop_lists(pop, video_ids, bans, top_k=50):
    """流行度 Top-K；bans: 每个用户的排除集合（其已看视频）。"""
    order = np.argsort(-pop)
    lists = []
    for ban in bans:
        ranked = []
        for i in order:
            v = int(video_ids[i])
            if v in ban:
                continue
            ranked.append(v)
            if len(ranked) >= top_k:
                break
        lists.append(ranked)
    return lists


def itemcf_lists(cfg, contexts, mask, video_ids, bans, top_k=50):
    """contexts: (U, L) 视频 id；用每用户最近 top_context_items 个行为打分；
    已看物品（bans）置为 -inf 后取 Top-K。"""
    cache = load_cache(cfg)
    sim = cache["sim"]
    index_of = {int(v): i for i, v in enumerate(video_ids)}
    limit = int(cfg["models"]["itemcf"]["top_context_items"])
    lists = []
    for u in range(len(contexts)):
        idxs = np.nonzero(mask[u])[0]
        vs = contexts[u][idxs][-limit:]
        ii = [index_of[int(v)] for v in vs]
        if not ii:
            score = cache["pop"].copy()
        else:
            score = np.asarray(sim[ii].sum(axis=0)).ravel()
        for v in bans[u]:
            j = index_of.get(int(v))
            if j is not None:
                score[j] = -1e9
        if len(score) > top_k:
            top = np.argpartition(-score, top_k - 1)[:top_k]
            top = top[np.argsort(-score[top])]
        else:
            top = np.argsort(-score)
        lists.append([int(video_ids[i]) for i in top])
    return lists
