"""E6：列表内多样性 / 新颖性指标测试（纯函数，合成数据）。"""

import numpy as np

from genrec.eval.metrics import intra_list_diversity, novelty


def test_ild_identical_vs_orthogonal():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # 两个相同向量：cos=1 -> 距离 0
    assert intra_list_diversity([[0, 1]], vecs) == 0.0
    # 两个正交向量：cos=0 -> 距离 1
    assert abs(intra_list_diversity([[0, 2]], vecs) - 1.0) < 1e-9
    # 混合：cos 均值 (1+0+0)/3 -> 距离 1 - 1/3
    assert abs(intra_list_diversity([[0, 1, 2]], vecs) - 2 / 3) < 1e-9


def test_ild_short_lists_and_truncation():
    vecs = np.eye(3)
    assert intra_list_diversity([[0]], vecs) == 0.0      # 单元素列表
    assert intra_list_diversity([[]], vecs) == 0.0
    # k 截断只看前 k 个
    assert intra_list_diversity([[0, 1, 2]], vecs, k=1) == 0.0


def test_novelty_known_values():
    counts = np.array([0, 0, 0, 1], dtype=np.float64)
    # p = (c + 1) / (total + N) = [1/5, 1/5, 1/5, 2/5]
    info0 = -np.log2(1 / 5)
    info3 = -np.log2(2 / 5)
    assert abs(novelty([[0, 1]], counts) - info0) < 1e-9
    assert abs(novelty([[3]], counts) - info3) < 1e-9
    assert abs(novelty([[0, 3]], counts) - (info0 + info3) / 2) < 1e-9


def test_novelty_rare_items_more_novel():
    counts = np.array([10, 1])
    assert novelty([[1]], counts) > novelty([[0]], counts)
