"""C2：SASRec-lite 的合成数据测试（padding 方案 / 因果性 / 可训练性）。"""

import numpy as np
import torch

from genrec.models.sasrec import SASRecLM, encode_last, pad_left, recent_items


def test_recent_items_respects_mask_and_order():
    vids = np.array([90, 91, 92, 93, 94])
    mask = np.array([True, False, True, True, False])
    assert recent_items(vids, mask, 2) == [92, 93]
    assert recent_items(vids, mask, 10) == [90, 92, 93]


def test_pad_left_scheme():
    ids, pos, pad = pad_left([[11, 12], [21, 22, 23]], 4)
    assert ids.tolist() == [[0, 0, 11, 12], [0, 21, 22, 23]]
    assert pos.tolist() == [[0, 0, 3, 4], [0, 2, 3, 4]]
    assert pad.tolist() == [[True, True, False, False],
                            [True, False, False, False]]


def test_future_items_do_not_leak_to_earlier_positions():
    torch.manual_seed(0)
    m = SASRecLM(10, d_model=16, n_layers=1, n_heads=2, dropout=0.0,
                 max_len=8)
    m.eval()
    ids_a, pos_a, pad_a = pad_left([[1, 2, 3, 4]], 4)
    ids_b, pos_b, pad_b = pad_left([[1, 2, 3, 9]], 4)
    with torch.no_grad():
        h_a = m(torch.from_numpy(ids_a), torch.from_numpy(pos_a),
                torch.from_numpy(pad_a))
        h_b = m(torch.from_numpy(ids_b), torch.from_numpy(pos_b),
                torch.from_numpy(pad_b))
    # 末位变化不影响前三个位置的隐状态（因果掩码）
    assert torch.allclose(h_a[:, :3], h_b[:, :3], atol=1e-6)
    # 但会影响末位自身
    assert not torch.allclose(h_a[:, 3], h_b[:, 3], atol=1e-4)


def test_batch_composition_invariance():
    """padding 掩盖正确：同一样本与更长样本同批时，输出不受批内他者影响。"""
    torch.manual_seed(0)
    m = SASRecLM(10, d_model=16, n_layers=1, n_heads=2, dropout=0.0,
                 max_len=8)
    h1 = encode_last(m, [[5, 6, 7]], 5)
    h2 = encode_last(m, [[5, 6, 7], [1, 2, 3, 4, 5]], 5)
    assert np.allclose(h1[0], h2[0], atol=1e-6)


def test_next_item_learnable_on_tiny_data():
    """小样本过拟合检查：模型应能学出"1→2、5→6、9→10"的模式。"""
    torch.manual_seed(0)
    m = SASRecLM(20, d_model=32, n_layers=2, n_heads=2, dropout=0.0,
                 max_len=8)
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    seqs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    pos = torch.tensor([[1, 2, 3, 4]] * 3)
    pad = torch.zeros(3, 4, dtype=torch.bool)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)          # 小张量下单线程更快（项目已知坑）
    for _ in range(300):
        h = m(seqs, pos, pad)
        loss = 0.0
        for t in range(1, 4):
            logits = h[:, t - 1, :] @ m.item.weight.T   # 前一位置 -> 当前物品
            loss = loss + torch.nn.functional.cross_entropy(
                logits, seqs[:, t]) / 3
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        h = m(seqs, pos, pad)
        pred = (h[:, 0, :] @ m.item.weight.T).argmax(-1).tolist()
    torch.set_num_threads(prev_threads)
    assert pred == [2, 6, 10]
