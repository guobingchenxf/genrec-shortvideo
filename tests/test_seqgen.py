"""合成小数据测试：tokenizer / trie 受限解码 / beam / 指标 / 训练序列构建。

合成数据仅用于测试代码正确性。
"""

import numpy as np

from genrec.eval.metrics import ndcg_at_k, recall_at_k
from genrec.models.seqgen import (NextTokenLM, RawTokenizer, SidTokenizer,
                                  beam_next_raw, beam_search_sid,
                                  build_training_sequences)


def _sid_tok():
    video_ids = np.array([10, 11, 12, 13])
    codes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    return SidTokenizer(video_ids, codes, codebook_size=256)


def test_sid_trie():
    tok = _sid_tok()
    assert sorted(tok.allowed_codes(())) == [0, 1]
    assert sorted(tok.allowed_codes((0, 0))) == [0, 1]
    assert tok.allowed_codes((0, 1)) == [0]
    assert tok.videos_for_sid((0, 0, 0)) == [0]
    assert tok.videos_for_sid((1, 0, 0)) == [3]
    assert tok.videos_for_sid((1, 1, 1)) == []


def test_sid_encode_tokens():
    tok = _sid_tok()
    toks = tok.encode_context(np.array([10, 11]), np.array([True, True]),
                              np.array([1, 2]), max_items=5)
    # 每个行为 = 3 个 SID token + 1 个行为 token
    # code_token(lvl, c) = 2 + lvl*256 + c; action_token(a) = 2 + 3*256 + a
    assert toks == [1, 2, 258, 514, 771, 2, 258, 515, 772]


def test_build_training_sequences():
    tok = _sid_tok()
    ctx = np.array([[10, 11]])
    mask = np.array([[True, True]])
    acts = np.array([[1, 2]])
    tg = np.array([12])
    x, y = build_training_sequences(tok, ctx, mask, acts, tg,
                                    max_items=4, is_sid=True)
    assert x.shape == (1, 1 + 4 * 4 + 3)
    # 左对齐布局：BOS(1) + 2 个行为(8) + 目标(3)，目标块位于位置 9..11
    # 目标视频 12 的码为 (0,1,0) -> token [2, 259, 514]
    assert x[0, 9:12].tolist() == [2, 259, 514]
    assert y[0, 8:11].tolist() == [2, 259, 514]
    # labels 为右移一位（同一序列内）
    assert (y[0, :-1] == x[0, 1:]).all()
    # 序列尾部 pad 区域 label = 0（忽略）
    assert (y[0, -1] == 0)


def test_sid_beam_search_returns_valid_unique_videos():
    tok = _sid_tok()
    model = NextTokenLM(tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                        max_len=64)
    ctx = tok.encode_context(np.array([10]), np.array([True]), np.array([0]),
                             max_items=3)
    lists = beam_search_sid(model, tok, [ctx], beam=8, batch_users=1)
    assert len(lists) == 1
    ranked = lists[0]
    assert len(ranked) == len(set(ranked))          # 无重复
    assert all(v in {10, 11, 12, 13} for v in ranked)  # 只生成合法视频


def test_raw_tokenizer_roundtrip():
    video_ids = np.array([10, 11, 12])
    tok = RawTokenizer(video_ids)
    assert tok.decode_video_token(tok.video_token(11)) == 11
    toks = tok.encode_context(np.array([10, 12]), np.array([True, True]),
                              np.array([1, 0]), max_items=5)
    assert toks == [1, 2, 6, 4, 5]


def test_raw_beam_excludes_non_video_tokens():
    tok = RawTokenizer(np.array([10, 11, 12]))
    model = NextTokenLM(tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                        max_len=32)
    ctx = tok.encode_context(np.array([10]), np.array([True]), np.array([0]),
                             max_items=3)
    lists = beam_next_raw(model, tok, [ctx], beam=10, batch_users=1)
    assert all(v in {10, 11, 12} for v in lists[0])


def test_metrics_known_values():
    assert recall_at_k([5, 6, 7, 8], {6}, 2) == 1.0
    assert recall_at_k([5, 6, 7, 8], {6}, 1) == 0.0
    assert abs(ndcg_at_k([5, 6], {5, 6}, 2) - 1.0) < 1e-9
    assert abs(ndcg_at_k([6, 5], {5}, 2) - 1.0 / np.log2(3)) < 1e-9


def test_evaluate_lists_hit_rate():
    from genrec.eval.metrics import evaluate_lists
    lists = [[1, 2], [3, 4]]
    targets = {7: {2}, 8: {9}}
    out = evaluate_lists(lists, targets, [7, 8], catalog_size=10, ks=(1, 2))
    assert out["hit_rate@1"] == 0.0          # 命中在第 2 位
    assert out["hit_rate@2"] == 0.5          # 用户 7 命中，用户 8 未命中
    # 用户 7 的 recall@2 = 1/1，用户 8 = 0；均值 0.5
    assert abs(out["recall@2"] - 0.5) < 1e-9
