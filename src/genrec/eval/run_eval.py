"""统一评估入口：在全观测协议上评估各方法，产出主表与排序列表存档。

方法清单：
  pop         - 流行度（训练窗口交互计数 Top-50）
  itemcf      - ItemCF（训练窗口共现相似度，context 最近 50 行为打分）
  gen-sid     - 生成式模型（语义 ID v1：256 码本，受限解码 + beam）
  gen-sid-v2  - 改进点 C1：语义 ID v2（1024 码本，降低碰撞）
  gen-raw     - 对照：同架构直接生成原生 video token
  gru-raw     - 轻量基线 GRU4Rec-lite（GRU + 原生 video token）

另外：所有方法的 Top-K 排序列表存档到 results/experiments/lists_*.npz，
供 scripts/analyze_buckets.py 做长尾/冷启动分桶分析。
"""

import json
import time

import numpy as np
import torch

from genrec.eval import baselines
from genrec.eval.metrics import evaluate_lists
from genrec.models.seqgen import (NextTokenLM, RawTokenizer, SidTokenizer,
                                  beam_next_raw, beam_search_sid)
from genrec.utils.monitor import save_json

METHOD_VARIANT = {"gen-sid": "sid", "gen-sid-v2": "sid-v2",
                  "gen-raw": "raw", "gru-raw": "raw-gru"}
ALL_METHODS = ["pop", "itemcf", "gen-sid", "gen-sid-v2", "gen-raw", "gru-raw"]
SID_FILES = {"sid": "sid_codes", "sid-v2": "sid_codes_v2"}


def _load_eval(cfg, smoke):
    processed = cfg.path("paths", "processed_dir")
    suffix = "_smoke" if smoke else ""
    ev = np.load(processed / f"eval_contexts{suffix}.npz")
    tg_raw = json.loads(
        (processed / f"eval_targets{suffix}.json").read_text(encoding="utf-8"))
    targets = {int(k): set(int(v) for v in vs)
               for k, vs in tg_raw["targets"].items()}
    video_ids = np.load(processed / f"video_vocab{suffix}.npz")["video_ids"]
    n = len(ev["user_ids"])
    if smoke:
        n = min(n, 100)
    return (suffix, ev["user_ids"][:n], ev["context_videos"][:n],
            ev["context_actions"][:n], ev["context_mask"][:n],
            targets, video_ids)


def _generate_lists(cfg, variant, contexts, actions, mask, video_ids, suffix,
                    beam=None):
    processed = cfg.path("paths", "processed_dir")
    ckpt_path = cfg.root / "results" / "models" / f"gen_{variant}{suffix}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}; run train-gen first")
    ckpt = torch.load(ckpt_path, weights_only=False)
    is_sid = bool(ckpt["is_sid"])
    if is_sid:
        z = np.load(processed / f"{SID_FILES[variant]}{suffix}.npz")
        tok = SidTokenizer(z["video_ids"], z["codes"])
    else:
        tok = RawTokenizer(video_ids)
    max_items = int(ckpt["max_items"])
    # 位置嵌入长度必须与训练时一致（否则 state_dict 不匹配）；
    # 旧检查点没有 max_len 字段，用训练时的公式重建
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
    beam = int(beam or cfg["models"]["gen"]["beam_size"])
    batch_users = int(cfg["models"]["gen"]["eval_batch_users"])
    tok_contexts = [tok.encode_context(contexts[i], mask[i], actions[i],
                                       max_items)
                    for i in range(len(contexts))]
    if is_sid:
        return beam_search_sid(model, tok, tok_contexts, beam=beam,
                               batch_users=batch_users)
    return beam_next_raw(model, tok, tok_contexts, beam=beam,
                         batch_users=batch_users * 2)


def run(cfg, methods=None, smoke=False, beam=None):
    methods = methods or ALL_METHODS
    suffix, user_ids, contexts, actions, mask, targets, video_ids = \
        _load_eval(cfg, smoke)
    experiments = cfg.root / "results" / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    # 统一排除规则：所有方法在排序/生成后剔除用户 context 中的已看视频
    bans = [set(int(v) for v in contexts[i][mask[i]])
            for i in range(len(contexts))]

    results = {}
    for m in methods:
        t0 = time.perf_counter()
        if m == "pop":
            cache = baselines.load_cache(cfg)
            lists = baselines.pop_lists(cache["pop"], video_ids, bans)
        elif m == "itemcf":
            lists = baselines.itemcf_lists(cfg, contexts, mask, video_ids, bans)
        elif m in METHOD_VARIANT:
            raw_lists = _generate_lists(cfg, METHOD_VARIANT[m], contexts,
                                        actions, mask, video_ids, suffix,
                                        beam=beam)
            lists = [[v for v in ranked if v not in bans[i]][:50]
                     for i, ranked in enumerate(raw_lists)]
        else:
            raise ValueError(f"unknown method: {m}")
        wall = time.perf_counter() - t0
        metrics = evaluate_lists(lists, targets, user_ids, len(video_ids))
        key = m if beam is None else f"{m}@beam{beam}"
        fname = m if beam is None else f"{m}_beam{beam}"
        metrics.update({
            "n_users": int(len(user_ids)),
            "eval_wall_seconds": round(wall, 1),
            "ms_per_user": round(wall * 1000.0 / max(1, len(user_ids)), 2),
            "smoke": smoke,
        })
        results[key] = metrics
        # 排序列表存档（供分桶/长尾分析；-1 为填充位）
        maxlen = max((len(r) for r in lists), default=1)
        padded = np.full((len(lists), maxlen), -1, dtype=np.int64)
        for i, r in enumerate(lists):
            padded[i, :len(r)] = r
        np.savez_compressed(experiments / f"lists_{fname}{suffix}.npz",
                            padded=padded, user_ids=user_ids)
        save_json(metrics, experiments / f"eval_{fname}{suffix}.json")
        print(f"[eval:{key}] {json.dumps(metrics, ensure_ascii=False)}")

    out = experiments / f"main_table{suffix}.json"
    if beam is None and not smoke:
        save_json(results, out)
    print(f"[eval] main table -> {out}")
    return results
