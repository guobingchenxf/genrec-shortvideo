"""预处理流水线。产物全部写入 data/processed/，报告写入 data/reports/。

"""

import numpy as np
import pandas as pd

from genrec.utils.monitor import StepTimer, save_json

BIG_DTYPES = {"user_id": "int32", "video_id": "int32",
              "timestamp": "float64", "watch_ratio": "float32"}
MATRIX_COLS = ["user_id", "video_id", "watch_ratio", "timestamp"]


# ----------------------------------------------------------------------
# 纯函数（tests/ 直接测试这些函数）
# ----------------------------------------------------------------------
def bucket_action(watch_ratio, buckets):
    """watch_ratio -> 行为 token。buckets 升序边界，返回 0..len(buckets)。"""
    buckets = np.asarray(buckets, dtype=np.float32)
    wr = np.asarray(watch_ratio, dtype=np.float32)
    return np.searchsorted(buckets, wr, side="right").astype(np.int8)


def time_split(df, quantile):
    """全局时间切分：返回 (train_df, val_df, t0)。t0 为时间戳分位数。"""
    t0 = float(df["timestamp"].quantile(quantile))
    train = df[df["timestamp"] <= t0].copy()
    val = df[df["timestamp"] > t0].copy()
    return train, val, t0


def build_train_samples(df, seq_len, max_per_user, max_total, seed,
                        exclude_pairs=None):
    """逐用户构造训练样本。

    - context 只取 target 位置之前的行为（防泄漏：不含 target 本身）；
    - exclude_pairs：set[(user_id, video_id)]，命中则丢弃该样本
      （用于剔除评估目标，避免转导泄漏）；
    - 每用户最多 max_per_user 个（取最近的），全局再随机下采样到 max_total。
    返回 (arrays, stats)。
    """
    rng = np.random.default_rng(seed)
    contexts, targets, target_actions, users = [], [], [], []
    order = df.sort_values(["user_id", "timestamp"], kind="stable")
    for uid, g in order.groupby("user_id", sort=False):
        vids = g["video_id"].to_numpy(dtype=np.int32)
        acts = g["action"].to_numpy(dtype=np.int8)
        n = len(vids)
        if n < 2:
            continue
        pos = np.arange(1, n)
        if len(pos) > max_per_user:
            pos = pos[-max_per_user:]
        for t in pos:
            tgt = int(vids[t])
            if exclude_pairs is not None and (int(uid), tgt) in exclude_pairs:
                continue
            start = max(0, t - seq_len)
            contexts.append((vids[start:t], acts[start:t]))
            targets.append(tgt)
            target_actions.append(int(acts[t]))
            users.append(int(uid))

    n_candidates = len(targets)
    if n_candidates > max_total:
        idx = rng.choice(n_candidates, size=max_total, replace=False)
        contexts = [contexts[i] for i in idx]
        targets = [targets[i] for i in idx]
        target_actions = [target_actions[i] for i in idx]
        users = [users[i] for i in idx]

    n = len(targets)
    L = seq_len
    ctx = np.zeros((n, L), dtype=np.int32)
    ctx_act = np.zeros((n, L), dtype=np.int8)
    mask = np.zeros((n, L), dtype=bool)
    for i, (cv, ca) in enumerate(contexts):
        m = len(cv)
        if m:
            ctx[i, L - m:] = cv
            ctx_act[i, L - m:] = ca
            mask[i, L - m:] = True

    arrays = {
        "context_videos": ctx,
        "context_actions": ctx_act,
        "context_mask": mask,
        "target_videos": np.asarray(targets, dtype=np.int32),
        "target_actions": np.asarray(target_actions, dtype=np.int8),
        "user_ids": np.asarray(users, dtype=np.int32),
    }
    stats = {
        "candidates_before_subsample": int(n_candidates),
        "samples": int(n),
        "sampled": bool(n_candidates > max_total),
        "users_with_samples": len(set(users)) if users else 0,
    }
    return arrays, stats


def build_small_eval(df_small, seq_len, context_ratio):
    """小矩阵全观测评估协议。

    每用户按时间排序：前 context_ratio 行为做 context（右对齐取尾 seq_len 条），
    其后交互去重保序作为 targets；另生成 leave-last-one 协议。
    返回 (eval_arrays, leave_one_arrays, targets_map, exclude_pairs)。
    """
    users_out, ctx_list, ctx_act_list = [], [], []
    lo_users, lo_ctx, lo_act, lo_tgt = [], [], [], []
    targets_map = {}

    order = df_small.sort_values(["user_id", "timestamp"], kind="stable")
    for uid, g in order.groupby("user_id", sort=False):
        vids = g["video_id"].to_numpy(dtype=np.int32)
        acts = g["action"].to_numpy(dtype=np.int8)
        n = len(vids)
        cut = max(1, round(n * context_ratio))
        if cut >= n:
            continue
        targets = list(dict.fromkeys(vids[cut:].tolist()))
        targets_map[int(uid)] = [int(v) for v in targets]
        ctx_list.append(vids[max(0, cut - seq_len):cut])
        ctx_act_list.append(acts[max(0, cut - seq_len):cut])
        users_out.append(int(uid))
        lo_ctx.append(vids[:-1][-seq_len:])
        lo_act.append(acts[:-1][-seq_len:])
        lo_tgt.append(int(vids[-1]))
        lo_users.append(int(uid))

    def _pad_with_mask(seqs, dtype, fill=0):
        L = seq_len
        out = np.full((len(seqs), L), fill, dtype=dtype)
        mask = np.zeros((len(seqs), L), dtype=bool)
        for i, s in enumerate(seqs):
            m = len(s)
            if m:
                out[i, L - m:] = s
                mask[i, L - m:] = True
        return out, mask

    ctx_mat, ctx_mask = _pad_with_mask(ctx_list, np.int32)
    ctx_act_mat, _ = _pad_with_mask(ctx_act_list, np.int8)
    eval_arrays = {
        "context_videos": ctx_mat,
        "context_actions": ctx_act_mat,
        "context_mask": ctx_mask,
        "user_ids": np.asarray(users_out, dtype=np.int32),
    }
    lo_mat, lo_mask = _pad_with_mask(lo_ctx, np.int32)
    lo_act_mat, _ = _pad_with_mask(lo_act, np.int8)
    leave_one = {
        "context_videos": lo_mat,
        "context_actions": lo_act_mat,
        "context_mask": lo_mask,
        "target_videos": np.asarray(lo_tgt, dtype=np.int32),
        "user_ids": np.asarray(lo_users, dtype=np.int32),
    }
    exclude_pairs = set()
    for uid, tgts in targets_map.items():
        for v in tgts:
            exclude_pairs.add((uid, int(v)))
    return eval_arrays, leave_one, targets_map, exclude_pairs


def build_video_texts(data_dir, video_ids):
    """组装每个视频的内容文本（caption + 封面 + 话题 + 三级类目 + raw 类目）。

    返回 (ids, corpus, coverage)。TF-IDF 路线（本模块）与神经编码器路线
    （data/content_neural.py）共用此函数，保证 B2 特征消融的文本源完全一致。
    文本只在内容侧使用，不存在时间泄漏问题；对所有视频统一处理（转导）。
    """
    # 注意：C 引擎在该文件会报 Buffer overflow（实测），改用 python 引擎；
    # 另实测有 8 行存在字段缺失（被补为 NaN），统一置空串处理
    cap = pd.read_csv(data_dir / "kuairec_caption_category.csv",
                      dtype=str, keep_default_na=False, engine="python")
    cap = cap.fillna("")
    text_parts = []
    for col in ["caption", "manual_cover_text", "topic_tag",
                "first_level_category_name", "second_level_category_name",
                "third_level_category_name"]:
        if col in cap.columns:
            text_parts.append(cap[col].str.replace("#", " ", regex=False))
    cap_text = pd.Series("", index=cap.index)
    for part in text_parts:
        cap_text = cap_text.str.cat(part, sep=" ")
    cap_vid = pd.to_numeric(cap["video_id"], errors="coerce")
    cap_valid = cap_vid.notna()
    caption_map = dict(zip(cap_vid[cap_valid].astype("int64"),
                           cap_text[cap_valid]))
    has_caption = set(cap_vid[cap_valid][
        cap_text[cap_valid].str.strip() != ""].astype("int64").tolist())

    category_map, has_categories = {}, set()
    cats = pd.read_csv(data_dir / "video_raw_categories_multi.csv",
                       dtype=str, keep_default_na=False).fillna("")
    if "category_name" in cats.columns and "video_id" in cats.columns:
        cat_vid = pd.to_numeric(cats["video_id"], errors="coerce")
        valid = cat_vid.notna()
        frame = pd.DataFrame({
            "video_id": cat_vid[valid].astype("int64"),
            "name": cats.loc[valid, "category_name"],
        })
        grouped = (frame[frame["name"].str.strip() != ""]
                   .groupby("video_id")["name"].apply(" ".join))
        category_map = grouped.to_dict()
        has_categories = set(grouped.index.tolist())

    ids = np.asarray(sorted({int(v) for v in video_ids}), dtype=np.int32)
    corpus, coverage = [], {"with_caption": 0, "with_categories": 0, "empty": 0}
    for v in ids.tolist():
        t_cap = caption_map.get(v, "")
        t_cat = category_map.get(v, "")
        combined = (t_cap + " " + t_cat).strip()
        if v in has_caption:
            coverage["with_caption"] += 1
        if v in has_categories:
            coverage["with_categories"] += 1
        if not combined:
            coverage["empty"] += 1
        corpus.append(combined)
    coverage["total_videos"] = len(ids)
    return ids, corpus, coverage


def build_content_features(video_ids, data_dir, content_cfg, seed):
    """caption + 类目 -> 字符 n-gram TF-IDF -> SVD。

    返回 (video_ids sorted array, feats float32 [N, d], coverage dict, svd_evr)。
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids, corpus, coverage = build_video_texts(data_dir, video_ids)

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=tuple(content_cfg["tfidf_ngram_range"]),
        max_features=content_cfg["tfidf_max_features"],
        min_df=content_cfg["tfidf_min_df"],
    )
    x = vec.fit_transform(corpus)
    svd = TruncatedSVD(n_components=content_cfg["svd_dim"], random_state=seed)
    feats = svd.fit_transform(x).astype(np.float32)
    evr = float(svd.explained_variance_ratio_.sum())
    return ids, feats, coverage, evr


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run(cfg, smoke=False):
    timer = StepTimer()
    raw_dir = cfg.path("paths", "raw_dir")
    data_dir = raw_dir / cfg["data"]["dir_name"]
    processed_dir = cfg.path("paths", "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = cfg.path("paths", "reports_dir")

    suffix = "_smoke" if smoke else ""
    pre = cfg["preprocess"]
    seed = pre["seed"]
    seq_len = pre["seq_len"]
    buckets = pre["watch_ratio_buckets"]

    n_rows_big = 300_000 if smoke else None
    n_rows_small = 50_000 if smoke else None

    timer.start("load_matrices")
    big = pd.read_csv(data_dir / "big_matrix.csv", usecols=MATRIX_COLS,
                      dtype=BIG_DTYPES, nrows=n_rows_big)
    small = pd.read_csv(data_dir / "small_matrix.csv", usecols=MATRIX_COLS,
                        dtype=BIG_DTYPES, nrows=n_rows_small)
    # 显式剔除关键字段缺失行（官方数据特性：small 矩阵约 3.9% 行缺 timestamp，
    # 无法进行时间排序）；剔除量写入 manifest，不做静默处理
    key_cols = ["user_id", "video_id", "timestamp", "watch_ratio"]
    dropped = {"big": int(len(big) - len(big.dropna(subset=key_cols))),
               "small": int(len(small) - len(small.dropna(subset=key_cols)))}
    big = big.dropna(subset=key_cols).reset_index(drop=True)
    small = small.dropna(subset=key_cols).reset_index(drop=True)
    big["action"] = bucket_action(big["watch_ratio"], buckets)
    small["action"] = bucket_action(small["watch_ratio"], buckets)

    timer.start("time_split")
    big_train, big_val, t0 = time_split(big, pre["time_split_quantile"])

    timer.start("build_small_eval")
    eval_arrays, leave_one, targets_map, exclude_pairs = build_small_eval(
        small, seq_len, pre["eval"]["context_ratio"])

    timer.start("build_train_samples")
    train_arrays, train_stats = build_train_samples(
        big_train, seq_len, pre["max_samples_per_user"],
        pre["max_train_samples"], seed, exclude_pairs=exclude_pairs)

    timer.start("build_val_samples")
    val_arrays, val_stats = build_train_samples(
        big_val, seq_len, max_per_user=20, max_total=50_000,
        seed=seed + 1, exclude_pairs=exclude_pairs)

    timer.start("content_features")
    all_videos = sorted(set(big["video_id"].tolist()) |
                        set(small["video_id"].tolist()))
    content_ids, content_feats, coverage, svd_evr = build_content_features(
        all_videos, data_dir, pre["content"], seed)

    timer.start("save_artifacts")
    np.savez_compressed(processed_dir / f"train_samples{suffix}.npz",
                        **train_arrays)
    np.savez_compressed(processed_dir / f"val_samples{suffix}.npz", **val_arrays)
    np.savez_compressed(processed_dir / f"eval_contexts{suffix}.npz",
                        **eval_arrays)
    np.savez_compressed(processed_dir / f"eval_leave_last_one{suffix}.npz",
                        **leave_one)
    save_json({"targets": {str(k): v for k, v in targets_map.items()}},
              processed_dir / f"eval_targets{suffix}.json")
    np.savez_compressed(processed_dir / f"content_feats{suffix}.npz",
                        video_ids=content_ids, feats=content_feats)
    np.savez_compressed(processed_dir / f"video_vocab{suffix}.npz",
                        video_ids=content_ids)

    timer.stop()
    manifest = {
        "smoke": smoke,
        "note": ("smoke artifacts: pipeline validation only, "
                 "NOT for official experimental conclusions") if smoke else
                "official artifacts",
        "seed": seed,
        "seq_len": seq_len,
        "watch_ratio_buckets": buckets,
        "time_split": {"quantile": pre["time_split_quantile"], "t0": t0,
                       "t0_readable": str(pd.to_datetime(t0, unit="s"))},
        "big_matrix_rows_used": len(big),
        "small_matrix_rows_used": len(small),
        "rows_dropped_nan_key_fields": dropped,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "eval_users": len(eval_arrays["user_ids"]),
        "eval_target_pairs_excluded_from_train": len(exclude_pairs),
        "content_coverage": coverage,
        "content_svd_explained_variance": svd_evr,
        "video_vocab_size": len(content_ids),
        "timings_seconds": timer.report()["wall_seconds"],
        "memory": timer.report(),
    }
    save_json(manifest, processed_dir / f"manifest{suffix}.json")
    save_json(manifest, reports_dir / f"preprocess_report{suffix}.json")

    print("[prepare] train samples:", train_stats)
    print("[prepare] val samples:", val_stats)
    print(f"[prepare] eval users: {manifest['eval_users']}, "
          f"excluded pairs: {manifest['eval_target_pairs_excluded_from_train']}")
    print("[prepare] content coverage:", coverage)
    print(f"[prepare] video vocab: {manifest['video_vocab_size']}, "
          f"svd evr: {svd_evr:.3f}")
    print("[prepare] timings:", manifest["timings_seconds"])
    print(f"[prepare] manifest -> {processed_dir / f'manifest{suffix}.json'}")
    return manifest
