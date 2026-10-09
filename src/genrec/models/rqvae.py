"""RQ-VAE：把内容特征量化为层次语义 ID（Semantic ID）。

"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from genrec.utils.monitor import save_json


class VectorQuantizer(nn.Module):
    """单级量化器：最近邻码本查找 + VQ 损失（码本项 + 承诺项）。"""

    def __init__(self, dim, codebook_size, commit_weight=0.25):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / codebook_size, 1.0 / codebook_size)
        self.commit_weight = commit_weight

    def forward(self, z):
        # z: (B, dim)
        dist = (z.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * z @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1))
        idx = dist.argmin(dim=1)                 # (B,)
        q = self.embedding(idx)                  # (B, dim)
        # 直通估计（STE）：量化不可导，把 q 的梯度原样传给 z，
        # 否则编码器收不到重建梯度，只会被承诺损失拉向少数码字（码本坍缩）
        q_st = z + (q - z).detach()
        codebook_loss = F.mse_loss(z.detach(), q)
        commit_loss = F.mse_loss(z, q.detach())
        return idx, q_st, codebook_loss + self.commit_weight * commit_loss


class RQVAE(nn.Module):
    def __init__(self, in_dim, hidden=128, latent_dim=32, levels=3,
                 codebook_size=256, commit_weight=0.25):
        super().__init__()
        self.levels = levels
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim))
        self.quantizers = nn.ModuleList([
            VectorQuantizer(latent_dim, codebook_size, commit_weight)
            for _ in range(levels)])
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim))

    def encode_codes(self, x):
        """返回 (B, levels) 的码字下标（int64）。"""
        residual = self.encoder(x)
        codes = []
        for layer in self.quantizers:
            idx, q, _ = layer(residual)
            codes.append(idx)
            residual = residual - q
        return torch.stack(codes, dim=1)

    def forward(self, x):
        residual = self.encoder(x)
        quantized = 0.0
        vq_loss = 0.0
        for layer in self.quantizers:
            _idx, q, loss = layer(residual)
            residual = residual - q
            quantized = quantized + q
            vq_loss = vq_loss + loss
        x_hat = self.decoder(quantized)
        recon = F.mse_loss(x_hat, x)
        return {"loss": recon + vq_loss, "recon": recon, "vq": vq_loss,
                "x_hat": x_hat}


def _restart_dead_codes(model, x, codebook_size, threshold=1):
    """把使用次数低于 threshold 的码字重置为随机样本的残差（防码本坍缩）。"""
    model.eval()
    n_restarted = 0
    with torch.no_grad():
        residual = model.encoder(x)
        for layer in model.quantizers:
            idx, q, _ = layer(residual)
            used = torch.bincount(idx, minlength=codebook_size)
            dead = (used < threshold).nonzero().flatten()
            if len(dead) > 0:
                pool = residual[torch.randint(0, len(residual), (len(dead),))]
                layer.embedding.weight.data[dead] = pool
                n_restarted += len(dead)
            residual = residual - q
    model.train()
    return n_restarted


def train_rqvae(feats, rqvae_cfg, device="cpu", log=None):
    """训练 RQ-VAE 并返回 (model, train_log, scaler)。

    对内容特征做标准化（零均值/单位方差）后再训练：
    TF-IDF+SVD 特征量纲小，直接 MSE 会导致码本坍缩（实测 smoke 训练）。
    scaler 随检查点保存，保证编码口径可复现。
    """
    torch.manual_seed(rqvae_cfg["seed"])
    np.random.seed(rqvae_cfg["seed"])
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    normed = ((feats - mu) / sd).astype(np.float32)
    scaler = {"mean": mu.astype(np.float32), "std": sd.astype(np.float32)}

    n, in_dim = normed.shape
    model = RQVAE(
        in_dim=in_dim,
        hidden=rqvae_cfg["hidden"],
        latent_dim=rqvae_cfg["latent_dim"],
        levels=rqvae_cfg["levels"],
        codebook_size=rqvae_cfg["codebook_size"],
        commit_weight=rqvae_cfg["commit_weight"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rqvae_cfg["lr"])
    x = torch.from_numpy(np.ascontiguousarray(normed))
    bs = rqvae_cfg["batch_size"]
    train_log = []
    for epoch in range(1, rqvae_cfg["epochs"] + 1):
        perm = torch.randperm(n)
        tot = rec = vq = 0.0
        nb = 0
        for s in range(0, n, bs):
            batch = x[perm[s:s + bs]].to(device)
            out = model(batch)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot += out["loss"].item()
            rec += out["recon"].item()
            vq += out["vq"].item()
            nb += 1
        train_log.append({
            "epoch": epoch,
            "loss": round(tot / nb, 6),
            "recon": round(rec / nb, 6),
            "vq": round(vq / nb, 6),
        })
        if log is not None and (epoch % max(1, rqvae_cfg["epochs"] // 8) == 0
                                or epoch == rqvae_cfg["epochs"]):
            log(f"[rqvae] epoch {epoch}/{rqvae_cfg['epochs']} "
                f"loss={train_log[-1]['loss']:.6f} "
                f"(recon={train_log[-1]['recon']:.6f}, vq={train_log[-1]['vq']:.6f})")
        restart_every = int(rqvae_cfg.get("restart_every", 0))
        if restart_every and epoch % restart_every == 0 and epoch < rqvae_cfg["epochs"]:
            n_dead = _restart_dead_codes(model, x, rqvae_cfg["codebook_size"])
            train_log[-1]["restarted_codes"] = n_dead
            if log is not None:
                log(f"[rqvae] epoch {epoch}: restarted {n_dead} dead codes")
    return model, train_log, scaler


def encode_all(model, feats, device="cpu", batch_size=4096):
    """对全部特征编码，返回 (N, levels) int16 码字。"""
    model.eval()
    codes = []
    with torch.no_grad():
        for s in range(0, len(feats), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(feats[s:s + batch_size], dtype=np.float32)).to(device)
            codes.append(model.encode_codes(batch).cpu().numpy())
    return np.concatenate(codes, axis=0).astype(np.int16)


def build_sid_table(video_ids, codes, popularity=None):
    """由码字构建 SID 表与碰撞统计。

    - sid_to_videos: {code 元组 -> [视频下标]}（组内按热度降序，默认按出现顺序）
    - 碰撞（多个视频共享同一 SID）不拼接额外位，改为"解码时展开候选组"，
      匹配口径在 evaluate 中统一处理并如实报告碰撞率。
    """
    video_ids = np.asarray(video_ids)
    groups = defaultdict(list)
    for i, key in enumerate(map(tuple, codes.tolist())):
        groups[key].append(i)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    sid_to_videos = {}
    for key, idxs in groups.items():
        if popularity is not None:
            idxs = sorted(idxs, key=lambda i: -float(popularity[i]))
        sid_to_videos[key] = idxs
    stats = {
        "n_videos": len(video_ids),
        "levels": int(codes.shape[1]),
        "codebook_size": 256,
        "unique_sids": len(groups),
        "collision_groups": len(collisions),
        "collided_videos": int(sum(len(v) for v in collisions.values())),
        "max_group_size": int(max(len(v) for v in groups.values())),
        "collision_rate": float(sum(len(v) for v in collisions.values()) / len(video_ids)),
    }
    return sid_to_videos, stats


def run(cfg, smoke=False, tag=None):
    """完整流程：读取内容特征 -> 训练 RQ-VAE -> 导出 SID 表与统计。

    tag：实验标签（如 "v2" 表示扩大码本的改进版本），用于区分产物文件名；
    配置取 models.rqvae 并按 models.rqvae_<tag> 覆盖（若存在）。
    """

    from genrec.utils.monitor import StepTimer

    timer = StepTimer()
    processed = cfg.path("paths", "processed_dir")
    experiments = cfg.root / "results" / "experiments"
    models_dir = cfg.root / "results" / "models"
    experiments.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tag_part = f"_{tag}" if tag else ""
    smoke_part = "_smoke" if smoke else ""

    rqvae_cfg = dict(cfg["models"]["rqvae"])
    if tag:
        rqvae_cfg.update(cfg["models"].get(f"rqvae_{tag}", {}))
    if smoke:
        rqvae_cfg["epochs"] = 30  # 干跑只减轮数；保持全部视频与正式流程同构

    timer.start("load_features")
    feats_file = rqvae_cfg.get("features_file", f"content_feats{smoke_part}.npz")
    data = np.load(processed / feats_file)
    feats = data["feats"]
    video_ids = data["video_ids"]

    timer.start("train_rqvae")
    model, train_log, scaler = train_rqvae(feats, rqvae_cfg)

    timer.start("encode_and_table")
    normed = ((feats - scaler["mean"]) / scaler["std"]).astype(np.float32)
    codes = encode_all(model, normed)
    _, sid_stats = build_sid_table(video_ids, codes)
    sid_stats["codebook_size"] = int(rqvae_cfg["codebook_size"])
    sid_stats["code_usage_per_level"] = [
        len(set(codes[:, lv].tolist())) for lv in range(codes.shape[1])]
    sid_stats["feature_normalization"] = "standardized (mean/std saved in checkpoint)"
    sid_stats["tag"] = tag

    timer.start("save")
    ckpt = {
        "state_dict": model.state_dict(),
        "config": rqvae_cfg,
        "video_ids": video_ids,
        "scaler": scaler,
    }
    torch.save(ckpt, models_dir / f"rqvae{tag_part}{smoke_part}.pt")
    np.savez_compressed(processed / f"sid_codes{tag_part}{smoke_part}.npz",
                        video_ids=video_ids, codes=codes)
    stats_payload = dict(sid_stats)
    stats_payload["smoke"] = smoke
    stats_payload["train_log_tail"] = train_log[-3:]
    stats_payload["timings_seconds"] = timer.report()["wall_seconds"]
    save_json(stats_payload, experiments / f"sid_stats{tag_part}{smoke_part}.json")

    timer.stop()
    print(f"[rqvae] tag={tag} final loss = {train_log[-1]['loss']:.6f} "
          f"(recon={train_log[-1]['recon']:.6f})")
    print(f"[rqvae] SID stats: {sid_stats}")
    print(f"[rqvae] timings: {timer.report()}")
    print(f"[rqvae] checkpoint -> {models_dir / f'rqvae{tag_part}{smoke_part}.pt'}")
    return sid_stats
