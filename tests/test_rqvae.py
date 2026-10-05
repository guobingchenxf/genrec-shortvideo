"""RQ-VAE 关键性质测试（合成小数据）。

重点回归：直通估计（STE）必须让重建梯度到达编码器——
没有 STE 时码本会坍缩（实测：10,728 视频只生成 593 个唯一 SID）。
"""

import torch

from genrec.models.rqvae import RQVAE, VectorQuantizer


def test_ste_passes_gradient_to_encoder():
    torch.manual_seed(0)
    model = RQVAE(in_dim=8, hidden=16, latent_dim=4, levels=2,
                  codebook_size=16)
    x = torch.randn(64, 8)
    out = model(x)
    out["recon"].backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, "encoder 未收到任何梯度"
    assert any(g.abs().sum() > 0 for g in grads), \
        "encoder 梯度全零：STE 可能失效"


def test_quantizer_returns_straight_through_tensor():
    torch.manual_seed(0)
    vq = VectorQuantizer(dim=4, codebook_size=8)
    z = torch.randn(3, 4, requires_grad=True)
    idx, q_st, _loss = vq(z)
    assert q_st.shape == z.shape
    # STE 张量：前向值等于码本向量（数值上 q_st == q）
    q = vq.embedding(idx)
    assert torch.allclose(q_st, q)
    # 反向：q_st 对 z 的梯度为 1（直通）
    q_st.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_codes_diversity_improves_with_training():
    """小规模可学习结构上做短训练：码字使用数应显著多于"全坍缩"。

    实测：lr=0.01 下约 600 次迭代收敛（8 个簇 -> 9~10 个码字）；
    200 次迭代时尚未收敛，不作为回归基准。
    """
    torch.manual_seed(0)
    centers = torch.randn(8, 8) * 5.0
    labels = torch.randint(0, 8, (512,))
    x = centers[labels] + torch.randn(512, 8) * 0.3
    model = RQVAE(in_dim=8, hidden=32, latent_dim=8, levels=1,
                  codebook_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(600):
        out = model(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    with torch.no_grad():
        codes = model.encode_codes(x)
    assert len(set(codes[:, 0].tolist())) >= 8
