"""合成序列测试：行为分桶、训练样本构造（防泄漏/padding/排除对）。

合成数据仅用于测试代码正确性。
"""

import numpy as np
import pandas as pd

from genrec.data.preprocess import bucket_action, build_train_samples


def _df(seq):
    return pd.DataFrame({
        "user_id": [u for u, _ in seq],
        "video_id": [v for _, v in seq],
        "timestamp": [float(i) for i in range(len(seq))],
        "action": np.zeros(len(seq), dtype=np.int8),
    })


def test_bucket_action_boundaries():
    buckets = [0.2, 0.5, 1.0, 2.0]
    wr = np.array([0.0, 0.19, 0.2, 0.5, 1.0, 1.99, 2.0, 5.0],
                  dtype=np.float32)
    assert bucket_action(wr, buckets).tolist() == [0, 0, 1, 2, 3, 3, 4, 4]


def test_train_samples_no_leakage_and_padding():
    df = _df([(1, 10), (1, 11), (1, 12), (1, 13)])
    arrays, stats = build_train_samples(
        df, seq_len=2, max_per_user=10, max_total=100, seed=0)
    assert stats["samples"] == 3
    ctx = arrays["context_videos"]
    tgt = arrays["target_videos"]
    assert ctx[0].tolist() == [0, 10]      # 右对齐 padding
    assert ctx[1].tolist() == [10, 11]
    assert ctx[2].tolist() == [11, 12]
    assert tgt.tolist() == [11, 12, 13]
    mask = arrays["context_mask"]
    assert mask[0].tolist() == [False, True]
    # 防泄漏：target 不出现在同一样本的 context 中（本测试视频号唯一）
    for i in range(len(tgt)):
        assert int(tgt[i]) not in ctx[i][mask[i]].tolist()


def test_exclude_pairs():
    df = _df([(1, 10), (1, 11), (1, 12)])
    arrays, stats = build_train_samples(
        df, seq_len=5, max_per_user=10, max_total=100, seed=0,
        exclude_pairs={(1, 12)})
    assert stats["samples"] == 1
    assert arrays["target_videos"].tolist() == [11]


def test_global_subsample_is_deterministic():
    seq = [(1, v) for v in range(50)]
    df = _df(seq)
    a1, s1 = build_train_samples(df, 5, 100, 10, seed=7)
    a2, s2 = build_train_samples(df, 5, 100, 10, seed=7)
    assert s1["samples"] == s2["samples"] == 10
    assert a1["target_videos"].tolist() == a2["target_videos"].tolist()
