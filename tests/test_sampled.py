"""合成小数据测试标准协议评估（C1/C3）。

- 负采样：排除历史、固定种子可复现、池子不足返回 None；
- 指标：HR@10 / NDCG@10 已知值；
- 序列似然打分：批量实现与"手工逐候选"参考实现一致；
- select_positions 前向与全量前向的 gather 结果一致。
"""

import numpy as np
import torch

from genrec.eval.sampled import hr_ndcg_at_10, sample_candidates, score_candidates
from genrec.models.seqgen import NextTokenLM, RawTokenizer, SidTokenizer


def _tiny_sid_tok():
    video_ids = np.array([10, 11, 12])
    codes = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    return SidTokenizer(video_ids, codes, codebook_size=8)


def _tiny_model(vocab_size, max_len=64):
    torch.manual_seed(0)
    model = NextTokenLM(vocab_size, d_model=16, n_layers=1, n_heads=2,
                        max_len=max_len)
    model.eval()
    return model


def test_sample_candidates_excludes_history_and_is_deterministic():
    seq = np.array([10, 11, 12, 10])
    all_videos = {10, 11, 12, 13, 14, 15}
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    target, negs = sample_candidates(seq, all_videos, n_neg=2, rng=rng1)
    assert target == 10
    assert set(negs.tolist()) <= {13, 14, 15}     # 历史 {10,11,12} 被排除
    _, negs2 = sample_candidates(seq, all_videos, n_neg=2, rng=rng2)
    assert negs.tolist() == negs2.tolist()        # 同种子可复现
    # 池子不足 -> None
    target, negs = sample_candidates(seq, {10, 11, 12, 13}, n_neg=5,
                                     rng=np.random.default_rng(0))
    assert target is None and negs is None


def test_hr_ndcg_known_values():
    # 101 列（目标在第 0 列）：行 0 目标排第 1，行 1 排第 3，行 2 排第 11
    m = np.zeros((3, 101))
    m[0, 0] = 10.0
    m[1, 0] = 5.0
    m[1, 1:3] = 6.0
    m[2, 0] = 5.0
    m[2, 1:11] = 6.0
    hr, ndcg = hr_ndcg_at_10(m)
    assert abs(hr - 2 / 3) < 1e-9
    # rank=1 -> ndcg 1；rank=3 -> 1/log2(4)=0.5；rank=11 -> 0
    assert abs(ndcg - (1.0 + 1.0 / np.log2(4) + 0.0) / 3) < 1e-6


def test_select_positions_matches_full_forward():
    tok = _tiny_sid_tok()
    model = _tiny_model(tok.vocab_size)
    row = [1, 5, 7, 9, 11, 13, 15, 0, 0]
    x = torch.tensor([row, row])              # batch = 2
    sel = torch.tensor([[3, 5], [4, 6]])
    full = model(x)
    got = model(x, select_positions=sel)
    idx = torch.arange(2)[:, None].expand(2, 2)
    expect = full[idx, sel]
    assert torch.allclose(got, expect)


def test_score_candidates_matches_manual():
    tok = _tiny_sid_tok()
    model = _tiny_model(tok.vocab_size)
    prefix = [1] + tok.encode_item(10, 0)
    candidates = [[11, 12, 10]]
    L = tok.n_levels
    batched = score_candidates(model, tok, [prefix], candidates, L)

    # 手工参考：逐候选跑全量前向，取候选 token 序列的 log-prob 之和
    manual = []
    for cand in candidates[0]:
        toks = prefix + tok.encode_target(cand)
        x = torch.tensor([toks])
        with torch.no_grad():
            logits = model(x)[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        base = len(toks) - L
        s = sum(float(lp[base - 1 + i, toks[base + i]]) for i in range(L))
        manual.append(s)
    assert np.allclose(batched[0], np.array(manual), atol=1e-5)


def test_score_candidates_raw_tokenizer():
    video_ids = np.array([10, 11, 12])
    tok = RawTokenizer(video_ids)
    model = _tiny_model(tok.vocab_size)
    prefix = [1] + tok.encode_item(10, 0)
    candidates = [[11, 12]]
    batched = score_candidates(model, tok, [prefix], candidates, L=1)
    toks = prefix + [tok.video_token(11)]
    with torch.no_grad():
        logits = model(torch.tensor([toks]))[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    manual = float(lp[len(toks) - 2, toks[-1]])
    assert abs(batched[0, 0] - manual) < 1e-5
