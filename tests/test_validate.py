"""小型合成数据测试校验纯函数。

"""

import pandas as pd

from genrec.data.validate import matrix_stats, validate_frame


def _make_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "video_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "timestamp": [1000.0 + i for i in range(10)],
        "watch_ratio": [0.5, 1.0, 1.5, 2.5, 0.1, 0.3, 4.0, 0.9, 1.2, 0.7],
    })


def test_validate_frame_ok():
    df = _make_df()
    stats, errors, warnings = validate_frame(df, None, "syn")
    assert errors == []
    assert warnings == []
    assert stats["rows"] == 10
    assert stats["users"] == 5
    assert stats["videos"] == 10


def test_validate_frame_expected_mismatch():
    df = _make_df()
    _, errors, _ = validate_frame(df, {"rows": 999}, "syn")
    assert any("rows" in e for e in errors)


def test_validate_frame_nan_policies():
    # watch_ratio 缺失或为负 -> 错误
    df = _make_df()
    df.loc[0, "watch_ratio"] = float("nan")
    df.loc[1, "watch_ratio"] = -0.5
    _, errors, _ = validate_frame(df, None, "syn")
    assert any("watch_ratio" in e for e in errors)
    assert any("negative" in e for e in errors)

    # timestamp 缺失 -> 警告而非错误（KuaiRec 已知数据特性）
    df2 = _make_df()
    df2.loc[0, "timestamp"] = float("nan")
    _, errors2, warnings2 = validate_frame(df2, None, "syn")
    assert errors2 == []
    assert any("timestamp" in w for w in warnings2)


def test_matrix_stats_like_ratio():
    df = _make_df()
    stats = matrix_stats(df, "syn")
    # watch_ratio > 2.0 的样本: 2.5 与 4.0 共 2/10
    assert abs(stats["like_ratio_wr_gt_2"] - 0.2) < 1e-6
