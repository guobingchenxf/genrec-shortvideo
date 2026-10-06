"""标准协议评估（C1/C3）：留一法 + 100 负采样，指标 HR@10 / NDCG@10。

协议（对齐 SASRec/BERT4Rec 一脉的常见做法）：
- 数据：**大矩阵**。每用户按时间排序，最后一条交互为目标（target），
  其前最近 max_context 条（含真实 watch_ratio 行为 token）为 context。
- 负样本：100 个"不在该用户历史中出现过"的物品（固定种子采样）。
- 打分（同一候选集，全部在 101 个候选内排名）：
  * 生成模型：教师强制序列似然——序列 = [BOS] + context token + 候选物品 token，
    候选得分 = 其 token 序列的对数概率之和（逐候选批处理，末尾 padding
    不影响因果注意力；只对候选 token 位置计算输出头）。
  * 流行度：训练窗口交互计数；ItemCF：同一 context（最近 ≤20 行为）的相似度和。
- 指标：HR@10（目标进前 10）、NDCG@10（单相关项，ndcg = 1/log2(rank+1)）。

诚实性说明（写入结果 JSON）：
- 生成模型训练于前 90% 时间窗，而目标多为窗口末端交互 → 近似"未来测试"，
  比文献"训练见完整历史"的留一法更严格；本表用于【同候选集同协议】的方法间对照，
  绝对值不与任何论文数字直接比较。
- 目标若在用户历史中重复出现（重看），仍作为目标保留；负样本保证不在历史中。
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from genrec.data.preprocess import bucket_action
from genrec.eval import baselines
from genrec.eval.run_eval import SID_FILES
from genrec.models.seqgen import BOS, NextTokenLM, RawTokenizer, SidTokenizer

DEFAULT_METHODS = ["random", "pop", "itemcf", "gen-sid", "gen-sid-v2",
                   "gen-raw", "gru-raw"]


# ----------------------------------------------------------------------
# 数据构建（纯函数部分可单测）
# ----------------------------------------------------------------------
def sample_candidates(seq_videos, all_videos, n_neg, rng):
    """给定一个用户的时间序物品序列，返回 (target, negatives)。

    负样本从"不在该用户历史中"的物品里无放回采样；池子不足时返回 None。
    """
    target = int(seq_videos[-1])
    hist = {int(v) for v in seq_videos}
    pool = np.asarray(sorted(all_videos - hist), dtype=np.int64)
    if len(pool) < n_neg:
        return None, None
    negatives = rng.choice(pool, size=n_neg, replace=False).astype(np.int64)
    return target, negatives


def load_loo_data(cfg, max_context=20, n_neg=100, seed=42, max_users=None):
    processed = cfg.path("paths", "processed_dir")
    raw_dir = cfg.path("paths", "raw_dir") / cfg["data"]["dir_name"]
    buckets = cfg["preprocess"]["watch_ratio_buckets"]
    manifest = json.loads(
        (processed / "manifest.json").read_text(encoding="utf-8"))
    t0 = float(manifest["time_split"]["t0"])

    df = pd.read_csv(
        raw_dir / "big_matrix.csv",
        usecols=["user_id", "video_id", "timestamp", "watch_ratio"],
        dtype={"user_id": "int32", "video_id": "int32",
               "timestamp": "float64", "watch_ratio": "float32"})
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["user_id", "timestamp"], kind="stable")

    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    all_videos = {int(v) for v in video_ids}

    rng = np.random.default_rng(seed)
    users, contexts, context_actions, targets, negatives, target_ts = \
        [], [], [], [], [], []
    for uid, g in df.groupby("user_id", sort=False):
        seq = g["video_id"].to_numpy()
        wr = g["watch_ratio"].to_numpy()
        acts = bucket_action(wr, buckets)
        if len(seq) < 2:
            continue
        target, negs = sample_candidates(seq, all_videos, n_neg, rng)
        if target is None:
            continue
        ctx = seq[max(0, len(seq) - 1 - max_context):-1]
        ctx_a = acts[max(0, len(seq) - 1 - max_context):-1]
        users.append(int(uid))
        contexts.append(ctx.astype(np.int64))
        context_actions.append(ctx_a.astype(np.int64))
        targets.append(target)
        negatives.append(negs)
        target_ts.append(float(g["timestamp"].to_numpy()[-1]))
        if max_users and len(users) >= max_users:
            break
    data = {
        "users": np.asarray(users, dtype=np.int64),
        "contexts": contexts,
        "context_actions": context_actions,
        "targets": np.asarray(targets, dtype=np.int64),
        "negatives": np.stack(negatives),
        "target_ts": np.asarray(target_ts, dtype=np.float64),
        "t0": t0,
    }
    return data


# ----------------------------------------------------------------------
# 打分器
# ----------------------------------------------------------------------
def score_random(data, seed=43):
    rng = np.random.default_rng(seed)
    n, c = len(data["users"]), data["negatives"].shape[1] + 1
    return rng.standard_normal((n, c)).astype(np.float32)


def score_pop(cfg, data):
    processed = cfg.path("paths", "processed_dir")
    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    index_of = {int(v): i for i, v in enumerate(video_ids)}
    pop = baselines.load_cache(cfg)["pop"]
    n = len(data["users"])
    c = data["negatives"].shape[1] + 1
    scores = np.zeros((n, c), dtype=np.float32)
    for ui in range(n):
        cands = np.concatenate([[data["targets"][ui]], data["negatives"][ui]])
        idx = np.array([index_of[int(v)] for v in cands])
        scores[ui] = pop[idx]
    return scores


def score_itemcf(cfg, data, context_items=20):
    processed = cfg.path("paths", "processed_dir")
    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    index_of = {int(v): i for i, v in enumerate(video_ids)}
    sim = baselines.load_cache(cfg)["sim"]
    n = len(data["users"])
    c = data["negatives"].shape[1] + 1
    scores = np.zeros((n, c), dtype=np.float32)
    for ui in range(n):
        ctx = data["contexts"][ui][-context_items:]
        ctx_idx = [index_of[int(v)] for v in ctx]
        full = np.asarray(sim[ctx_idx].sum(axis=0)).ravel()
        cands = np.concatenate([[data["targets"][ui]], data["negatives"][ui]])
        scores[ui] = full[[index_of[int(v)] for v in cands]]
    return scores


@torch.no_grad()
def score_candidates(model, tok, prefixes, candidates, L):
    """对候选集做序列似然打分（核心纯函数，便于单测）。

    prefixes: list[B]，每个元素的 token 列表（[BOS] + context token）
    candidates: list[B] -> list[C]，每个用户的目标+负样本（视频 id，目标在第 0 位）
    L: 每个候选占用的 token 数（SID=层数；raw=1）
    返回 np.ndarray (B, C)：候选 token 序列的 log-prob 之和。
    """
    B = len(prefixes)
    C = len(candidates[0])
    rows, metas = [], []
    for bi in range(B):
        for ci in range(C):
            toks = prefixes[bi] + tok.encode_target(int(candidates[bi][ci]))
            rows.append(toks)
            metas.append((bi, ci, len(toks)))
    P = max(len(r) for r in rows)
    x = torch.zeros(len(rows), P, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r)
    pred_pos = torch.zeros(len(rows), L, dtype=torch.long)
    tok_idx = torch.zeros(len(rows), L, dtype=torch.long)
    for i, (_bi, _ci, ntok) in enumerate(metas):
        base = ntok - L
        pred_pos[i] = torch.arange(base - 1, base - 1 + L)
        tok_idx[i] = x[i, base:base + L]
    logits = model(x, select_positions=pred_pos)          # (R, L, V)
    lp = torch.log_softmax(logits.float(), dim=-1)
    sc = lp.gather(2, tok_idx.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    scores = np.zeros((B, C), dtype=np.float32)
    for i, (bi, ci, _n) in enumerate(metas):
        scores[bi, ci] = float(sc[i])
    return scores


@torch.no_grad()
def score_generator(cfg, variant, data, users_chunk=4, cands_chunk=32,
                    device="cpu"):
    """教师强制序列似然打分：候选得分 = 其 token 序列 log-prob 之和。"""
    processed = cfg.path("paths", "processed_dir")
    ckpt_path = cfg.root / "results" / "models" / f"gen_{variant}.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    is_sid = bool(ckpt["is_sid"])
    if is_sid:
        z = np.load(processed / f"{SID_FILES[variant]}.npz")
        tok = SidTokenizer(z["video_ids"], z["codes"])
    else:
        video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
        tok = RawTokenizer(video_ids)
    max_items = int(ckpt["max_items"])
    block = (tok.n_levels + 1) if is_sid else 2
    tgt_len = tok.n_levels if is_sid else 1
    max_len = int(ckpt.get("max_len") or (1 + max_items * block + tgt_len + 8))
    model = NextTokenLM(
        ckpt["vocab_size"],
        d_model=int(ckpt["config"]["d_model"]),
        n_layers=int(ckpt["config"]["n_layers"]),
        n_heads=int(ckpt["config"].get("n_heads", 4)),
        dropout=0.0, max_len=max_len,
        backbone=ckpt["config"]["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    L = tgt_len

    n = len(data["users"])
    c = data["negatives"].shape[1] + 1
    scores = np.full((n, c), -1e9, dtype=np.float32)
    for u0 in range(0, n, users_chunk):
        u1 = min(u0 + users_chunk, n)
        prefixes, cands = [], []
        for ui in range(u0, u1):
            ctx = data["contexts"][ui][-max_items:]
            ctx_a = data["context_actions"][ui][-max_items:]
            toks = [BOS]
            for v, a in zip(ctx, ctx_a):
                toks.extend(tok.encode_item(int(v), int(a)))
            prefixes.append(toks)
            cands.append(np.concatenate(
                [[data["targets"][ui]], data["negatives"][ui]]))
        for c0 in range(0, c, cands_chunk):
            c1 = min(c0 + c, c)
            block_cands = [list(cc[c0:c1]) for cc in cands]
            sub = score_candidates(model, tok, prefixes, block_cands, L)
            scores[u0:u1, c0:c1] = sub
    return scores


# ----------------------------------------------------------------------
# 指标与主流程
# ----------------------------------------------------------------------
def hr_ndcg_at_10(scores):
    """目标位于第 0 列；其余列为候选。返回 (HR@10, NDCG@10)。

    NDCG 单相关项：rank 命中前 10 时 ndcg = 1/log2(rank+1)（rank 为 1 起）。
    """
    tgt = scores[:, 0]
    neg = scores[:, 1:]
    rank = 1 + (neg > tgt[:, None]).sum(axis=1)      # 1 起；并列按严格大于
    hr10 = float((rank <= 10).mean())
    ndcg10 = float(np.where(rank <= 10, 1.0 / np.log2(rank + 1), 0.0).mean())
    return hr10, ndcg10


def run(cfg, methods=None, max_users=None, n_neg=100, seed=42):
    methods = methods or DEFAULT_METHODS
    t_start = time.perf_counter()
    data = load_loo_data(cfg, n_neg=n_neg, seed=seed, max_users=max_users)
    n = len(data["users"])
    t0 = data["t0"]
    future_share = float((data["target_ts"] > t0).mean())
    print(f"[sampled] users={n}, n_neg={n_neg}, seed={seed}, "
          f"target 时间在训练窗之后的占比={future_share:.2%}")

    experiments = cfg.root / "results" / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_users is None else f"_n{max_users}"
    results = {}
    for m in methods:
        t0_ = time.perf_counter()
        if m == "random":
            scores = score_random(data, seed=seed + 1)
        elif m == "pop":
            scores = score_pop(cfg, data)
        elif m == "itemcf":
            scores = score_itemcf(cfg, data)
        elif m in ("gen-sid", "gen-sid-v2", "gen-raw", "gru-raw"):
            variant = {"gen-sid": "sid", "gen-sid-v2": "sid-v2",
                       "gen-raw": "raw", "gru-raw": "raw-gru"}[m]
            scores = score_generator(cfg, variant, data)
        else:
            raise ValueError(f"unknown method: {m}")
        hr10, ndcg10 = hr_ndcg_at_10(scores)
        wall = time.perf_counter() - t0_
        results[m] = {
            "hr@10": round(hr10, 4),
            "ndcg@10": round(ndcg10, 4),
            "wall_seconds": round(wall, 1),
            "ms_per_user": round(wall * 1000 / max(1, n), 2),
        }
        print(f"[sampled:{m}] hr@10={hr10:.4f} ndcg@10={ndcg10:.4f} "
              f"({wall:.1f}s)")

    payload = {
        "protocol": "leave-one-out + 100 sampled negatives (big matrix)",
        "n_users": int(n),
        "n_neg": int(n_neg),
        "seed": int(seed),
        "max_context": 20,
        "targets_after_train_window_share": round(future_share, 4),
        "note": ("生成模型训练于前 90% 时间窗；目标多为窗口末端交互（近似未来测试）；"
                 "本表用于同候选集同协议的方法间对照，绝对值不与论文数字直接比较。"),
        "results": results,
        "total_wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    out = experiments / f"sampled_protocol{suffix}.json"
    if out.exists():
        # 合并进已有结果：允许分批评不同方法（n_users/协议参数须一致）
        prev = json.loads(out.read_text(encoding="utf-8"))
        if prev.get("n_users") == int(n):
            prev["results"].update(results)
            prev["total_wall_seconds"] = round(
                prev.get("total_wall_seconds", 0) + payload["total_wall_seconds"], 1)
            payload["results"] = prev["results"]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[sampled] -> {out}")
    return payload
