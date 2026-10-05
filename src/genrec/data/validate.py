"""原始数据校验。

- 校验对象：big/small matrix 的行数/用户数/视频数（对照官方 README 口径）、
  关键字段缺失与取值范围；caption/类目文件对视频的覆盖率；两矩阵交集。
- 不一致必须写入报告并以失败态返回（CLI 非零退出），不得静默通过。
- tests/ 仅用小型合成数据测试本模块的纯函数。
"""

from datetime import datetime, timezone

import pandas as pd

from genrec.utils.monitor import StepTimer, save_json

MATRIX_USECOLS = ["user_id", "video_id", "timestamp", "watch_ratio"]
MATRIX_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "timestamp": "float64",
    "watch_ratio": "float32",
}


def load_matrix(path):
    head = pd.read_csv(path, nrows=0)
    missing = [c for c in MATRIX_USECOLS if c not in head.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return pd.read_csv(path, usecols=MATRIX_USECOLS, dtype=MATRIX_DTYPES)


def matrix_stats(df, name):
    return {
        "name": name,
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "videos": int(df["video_id"].nunique()),
        "nan_user_id": int(df["user_id"].isna().sum()),
        "nan_video_id": int(df["video_id"].isna().sum()),
        "nan_timestamp": int(df["timestamp"].isna().sum()),
        "nan_watch_ratio": int(df["watch_ratio"].isna().sum()),
        "watch_ratio_min": float(df["watch_ratio"].min()),
        "watch_ratio_max": float(df["watch_ratio"].max()),
        "watch_ratio_p999": float(df["watch_ratio"].quantile(0.999)),
        "watch_ratio_gt_50": int((df["watch_ratio"] > 50).sum()),
        "like_ratio_wr_gt_2": float((df["watch_ratio"] > 2.0).mean()),
        "timestamp_min": str(pd.to_datetime(df["timestamp"].min(), unit="s")),
        "timestamp_max": str(pd.to_datetime(df["timestamp"].max(), unit="s")),
    }


def validate_frame(df, expected, name):
    """返回 (stats, errors, warnings)。

    errors：结构性问题（规模不符 / user_id、video_id、watch_ratio 缺失 / 负值）；
    warnings：数据特性（timestamp 缺失与 watch_ratio 极值都属于 KuaiRec 已知特性，
    前者在 preprocess 中显式剔除并计数，后者由分桶封顶，均不改变实验口径）。
    """
    stats = matrix_stats(df, name)
    errors, warnings = [], []
    if expected:
        for key in ("rows", "users", "videos"):
            want = expected.get(key)
            got = stats[key]
            if want is not None and got != want:
                errors.append(f"{name}.{key}: expected {want}, got {got}")
    for col in ("user_id", "video_id", "watch_ratio"):
        count = stats[f"nan_{col}"]
        if count > 0:
            errors.append(f"{name}: NaN in {col} ({count} rows)")
    if stats["nan_timestamp"] > 0:
        frac = stats["nan_timestamp"] / max(1, stats["rows"])
        warnings.append(
            f"{name}: {stats['nan_timestamp']} rows ({frac:.2%}) lack timestamp - "
            f"known data property; excluded in preprocess (counted in manifest)")
    if stats["watch_ratio_min"] < 0:
        errors.append(f"{name}: negative watch_ratio ({stats['watch_ratio_min']})")
    if stats["watch_ratio_gt_50"] > 0:
        warnings.append(
            f"{name}: {stats['watch_ratio_gt_50']} rows with watch_ratio>50 "
            f"(max {stats['watch_ratio_max']:.1f}) - known property (replay loops); "
            f"behavior bucketing caps at the top bucket")
    return stats, errors, warnings


def _numeric_video_ids(series):
    return set(
        pd.to_numeric(series, errors="coerce").dropna().astype("int64").tolist())


def content_coverage(caption_path, raw_categories_path, target_videos):
    """caption / 类目 对目标视频集合的覆盖率（用于评估内容特征可用性）。"""
    # 该文件存在超长行，pandas C 引擎会报 Buffer overflow（2026-10-05 实测），
    # 改用 python 引擎（实测读取正常）
    cap = pd.read_csv(caption_path, engine="python")
    if "caption" not in cap.columns:
        raise ValueError(f"{caption_path}: missing column caption")
    nonempty = cap["caption"].fillna("").astype(str).str.strip() != ""
    caption_videos = _numeric_video_ids(cap.loc[nonempty, "video_id"])
    try:
        cats = pd.read_csv(raw_categories_path)
        category_videos = _numeric_video_ids(cats["video_id"])
    except Exception as exc:  # 明确记录而不是静默：该文件是 SID 的特征来源之一
        category_videos = set()
        print(f"[validate][warn] raw categories unreadable: {exc}")
    target = set(int(v) for v in target_videos)
    return {
        "target_videos": len(target),
        "with_caption": len(target & caption_videos),
        "with_categories": len(target & category_videos),
        "with_either": len(target & (caption_videos | category_videos)),
        "with_neither": len(target - caption_videos - category_videos),
    }


def run(cfg):
    timer = StepTimer()
    raw_dir = cfg.path("paths", "raw_dir")
    reports_dir = cfg.path("paths", "reports_dir")
    data_dir = raw_dir / cfg["data"]["dir_name"]
    if not data_dir.exists():
        raise FileNotFoundError(
            f"raw data dir not found: {data_dir}; run download first")

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.relative_to(cfg.root)),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    timer.start("load_big")
    big = load_matrix(data_dir / "big_matrix.csv")
    timer.start("load_small")
    small = load_matrix(data_dir / "small_matrix.csv")

    timer.start("validate_frames")
    big_stats, errs, warns = validate_frame(big, cfg["validate"]["expected_big"], "big")
    report["checks"]["big_matrix"] = big_stats
    report["errors"] += errs
    report["warnings"] += warns
    small_stats, errs, warns = validate_frame(small, cfg["validate"]["expected_small"], "small")
    report["checks"]["small_matrix"] = small_stats
    report["errors"] += errs
    report["warnings"] += warns

    timer.start("cross_checks")
    big_users = set(big["user_id"].unique().tolist())
    big_videos = set(big["video_id"].unique().tolist())
    small_users = set(small["user_id"].unique().tolist())
    small_videos = set(small["video_id"].unique().tolist())
    report["checks"]["overlap"] = {
        "small_users_in_big": len(small_users & big_users),
        "small_users_total": len(small_users),
        "small_videos_in_big": len(small_videos & big_videos),
        "small_videos_total": len(small_videos),
    }

    timer.start("content_coverage")
    report["checks"]["content_coverage"] = content_coverage(
        data_dir / "kuairec_caption_category.csv",
        data_dir / "video_raw_categories_multi.csv",
        big_videos | small_videos,
    )

    timer.stop()
    report["timings_seconds"] = timer.report()["wall_seconds"]
    report["ok"] = len(report["errors"]) == 0
    out = reports_dir / "validation_report.json"
    save_json(report, out)

    print(f"[validate] big:   {report['checks']['big_matrix']}")
    print(f"[validate] small: {report['checks']['small_matrix']}")
    print(f"[validate] overlap: {report['checks']['overlap']}")
    print(f"[validate] coverage: {report['checks']['content_coverage']}")
    for w in report["warnings"]:
        print(f"[validate][warn] {w}")
    for e in report["errors"]:
        print(f"[validate][ERROR] {e}")
    print(f"[validate] {'OK' if report['ok'] else 'FAILED'} -> {out}")
    return report["ok"]
