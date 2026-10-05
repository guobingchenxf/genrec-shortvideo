"""合成数据测试时间切分与全观测评估协议。

合成数据仅用于测试代码正确性。
"""

import numpy as np
import pandas as pd

from genrec.data.preprocess import build_small_eval, time_split


def test_time_split_partition():
    df = pd.DataFrame({
        "timestamp": [float(i) for i in range(100)],
        "user_id": [1] * 100,
        "video_id": list(range(100)),
        "watch_ratio": [1.0] * 100,
    })
    train, val, t0 = time_split(df, 0.9)
    assert len(train) + len(val) == len(df)
    assert train["timestamp"].max() <= t0
    assert val["timestamp"].min() > t0


def test_eval_protocol_partition():
    df = pd.DataFrame({
        "user_id": [1] * 10,
        "video_id": list(range(100, 110)),
        "timestamp": [float(i) for i in range(10)],
        "watch_ratio": [1.0] * 10,
        "action": np.zeros(10, dtype=np.int8),
    })
    eval_arrays, leave_one, targets_map, exclude_pairs = build_small_eval(
        df, seq_len=5, context_ratio=0.8)

    # 前 80% = 8 条, 后 20% 为 targets
    assert targets_map[1] == [108, 109]
    ctx = eval_arrays["context_videos"][0]
    assert ctx.tolist() == [103, 104, 105, 106, 107]   # 前 8 条的右对齐尾 5 条
    assert not (set(ctx.tolist()) & set(targets_map[1]))

    # leave-last-one 协议
    assert leave_one["target_videos"].tolist() == [109]
    assert leave_one["context_videos"][0].tolist() == [104, 105, 106, 107, 108]

    # 泄漏对
    assert exclude_pairs == {(1, 108), (1, 109)}


def test_eval_protocol_skips_single_interaction_users():
    df = pd.DataFrame({
        "user_id": [1],
        "video_id": [5],
        "timestamp": [0.0],
        "watch_ratio": [1.0],
        "action": np.zeros(1, dtype=np.int8),
    })
    eval_arrays, _, targets_map, _ = build_small_eval(df, seq_len=5,
                                                      context_ratio=0.8)
    assert targets_map == {}
    assert len(eval_arrays["user_ids"]) == 0
