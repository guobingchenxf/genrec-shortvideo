"""C2：SASRec-lite 基线（自注意力序列推荐，采样 softmax 训练版）。

参考：Kang & McAuley, "Self-Attentive Sequential Recommendation"（ICDM 2018）。
CPU 友好的 lite 实现，与生成式模型保持同一数据预算（150k 序列样本、上下文窗口 20）；
与论文实现的差异（如实声明，写入检查点 config）：
- 损失用**全词表交叉熵**（与生成式模型同损失族；首版按路线图用 K=256 采样 softmax，
  实测信号偏弱、SASRec 明显欠训——采样版 HR@10 仅 0.22——遂改为全词表）；
- 左 padding + 窗口内绝对位置编码（padding 位置为 0），因果掩码；pre-LN 编码块；
- 输出头与物品嵌入共享权重。

序列格式与流水线一致：用户时间序最近 <=max_items 个物品；训练时追加 target
（全部有效位置监督"预测下一物品"），评估时取最后位置隐状态对候选打分。
"""

import time

import numpy as np
import torch
from torch import nn

from genrec.utils.monitor import save_json

PAD = 0


def recent_items(videos_row, mask_row, max_items):
    """用户时间序中最近 max_items 个有效物品（保持时间顺序）。"""
    idx = np.nonzero(mask_row)[0]
    if len(idx) > max_items:
        idx = idx[-max_items:]
    return [int(v) for v in videos_row[idx]]


def pad_left(seqs, width):
    """list[list[int]] -> (ids, pos, pad_mask)。

    右对齐 / 左 padding；位置 = 窗口内绝对序号 1..width（padding 位置置 0）。
    """
    ids = np.zeros((len(seqs), width), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = list(s)[-width:]
        if s:
            ids[i, width - len(s):] = s
    pos = np.tile(np.arange(1, width + 1), (len(seqs), 1))
    pos[ids == PAD] = 0
    return ids, pos, (ids == PAD)


class SASRecLM(nn.Module):
    """因果自注意力编码器 + 共享物品嵌入输出头。"""

    def __init__(self, n_items, d_model=64, n_layers=2, n_heads=2,
                 dropout=0.1, max_len=24):
        super().__init__()
        self.n_items = int(n_items)
        self.item = nn.Embedding(self.n_items + 1, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len + 1, d_model, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        # 与原版 SASRec 一致的小方差初始化：默认 N(0,1) 会让点积 logits 过大
        # （实测起始 CE≈21，正常应≈log K），继续训练会发散出 NaN。
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, ids, pos, pad_mask):
        _, T = ids.shape
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        h = self.drop(self.item(ids) + self.pos(pos))
        return self.backbone(h, mask=causal, src_key_padding_mask=pad_mask)


def load_sasrec(cfg, ckpt_name="sasrec.pt"):
    """加载检查点并重建模型（评估侧统一入口）。"""
    ckpt_path = cfg.root / "results" / "models" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}; run train-sasrec first")
    ckpt = torch.load(ckpt_path, weights_only=False)
    conf = ckpt["config"]
    model = SASRecLM(int(ckpt["n_items"]), d_model=int(conf["d_model"]),
                     n_layers=int(conf["n_layers"]),
                     n_heads=int(conf["n_heads"]), dropout=0.0,
                     max_len=int(ckpt["max_len"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def encode_last(model, ctx_seqs, max_items, batch=256):
    """每个 context 最后位置的隐状态 (N, d) numpy float32。"""
    model.eval()
    outs = []
    for s in range(0, len(ctx_seqs), batch):
        chunk = [list(c)[-max_items:] for c in ctx_seqs[s:s + batch]]
        ids, pos, pad = pad_left(chunk, max_items)
        h = model(torch.from_numpy(ids), torch.from_numpy(pos),
                  torch.from_numpy(pad))
        outs.append(h[:, -1, :].float().cpu())
    if not outs:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(outs).numpy()


@torch.no_grad()
def rank_all(model, ctx_seqs, max_items, topk=50, batch=256):
    """全库排序：返回 list[list[int]]（物品索引 0..n_items-1，对应 video_vocab 顺序）。"""
    model.eval()
    weights = model.item.weight[1:].detach().float()      # (n_items, d)
    lists = []
    for s in range(0, len(ctx_seqs), batch):
        chunk = [list(c)[-max_items:] for c in ctx_seqs[s:s + batch]]
        ids, pos, pad = pad_left(chunk, max_items)
        h = model(torch.from_numpy(ids), torch.from_numpy(pos),
                  torch.from_numpy(pad))
        logits = h[:, -1, :].float() @ weights.T          # (B, n_items)
        k = min(topk, logits.shape[1])
        top = torch.topk(logits, k, dim=1).indices.cpu().numpy()
        lists.extend(row.tolist() for row in top)
    return lists


@torch.no_grad()
def score_item_candidates(model, ctx_seqs, cand_idx, max_items, batch=128):
    """候选打分：cand_idx 为物品索引（1..n_items，目标在第 0 位）-> (N, C)。"""
    model.eval()
    n = len(ctx_seqs)
    scores = np.full((n, len(cand_idx[0])), -1e9, dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        chunk = [list(c)[-max_items:] for c in ctx_seqs[s:e]]
        ids, pos, pad = pad_left(chunk, max_items)
        h = model(torch.from_numpy(ids), torch.from_numpy(pos),
                  torch.from_numpy(pad))
        h = h[:, -1, :].float()
        sub = np.stack([np.asarray(c, dtype=np.int64)
                        for c in cand_idx[s:e]])
        emb = model.item(torch.from_numpy(sub))           # (b, C, d)
        sc = torch.einsum("bd,bcd->bc", h, emb)
        scores[s:e] = sc.float().cpu().numpy()
    return scores


def run(cfg, smoke=False, seed=None, epochs=None, tag="", log_every=100):
    """训练 SASRec-lite。参数语义与 train.run_gen 保持一致（供战役复用）。"""
    base = dict(cfg["models"]["sasrec"])
    seed_eff = int(seed if seed is not None else base["seed"])
    torch.manual_seed(seed_eff)
    np.random.seed(seed_eff)
    torch.set_num_threads(int(base.get("num_threads", 4)))

    processed = cfg.path("paths", "processed_dir")
    suffix = "_smoke" if smoke else ""
    tr = np.load(processed / f"train_samples{suffix}.npz")
    vids = np.load(processed / f"video_vocab{suffix}.npz")["video_ids"]
    n_items = len(vids)
    max_items = int(base["max_context_items"])
    width = max_items + 1
    max_len = int(base.get("max_len", width + 3))
    assert width <= max_len, "max_len 必须 >= max_context_items + 1"

    index_of = {int(v): i + 1 for i, v in enumerate(vids)}
    # 注意：npz 的每个 key 每次取值都会重新解压整个数组（含 CRC 校验），
    # 必须先物化到普通数组再逐行循环（曾因逐行取值导致训练实质上"永不结束"）。
    ctx_videos = tr["context_videos"]
    ctx_mask = tr["context_mask"]
    targets = tr["target_videos"]
    seqs = [[index_of[int(v)] for v in
             recent_items(ctx_videos[i], ctx_mask[i], max_items)]
            + [index_of[int(targets[i])]]
            for i in range(len(targets))]
    ids_np, pos_np, pad_np = pad_left(seqs, width)

    n = len(seqs)
    bs = int(base["batch_size"])
    epochs_eff = int(epochs if epochs is not None else base["epochs"])
    if smoke:
        n = min(n, 2048)
        bs = min(bs, 64)
        epochs_eff = 1
    ids = torch.from_numpy(ids_np[:n])
    pos = torch.from_numpy(pos_np[:n])
    pad = torch.from_numpy(pad_np[:n])

    model = SASRecLM(n_items, d_model=int(base["d_model"]),
                     n_layers=int(base["n_layers"]),
                     n_heads=int(base["n_heads"]),
                     dropout=float(base.get("dropout", 0.1)), max_len=max_len)
    opt = torch.optim.Adam(model.parameters(), lr=float(base["lr"]))

    print(f"[sasrec{tag}] samples={n}, width={width}, n_items={n_items}, "
          f"seed={seed_eff}, epochs={epochs_eff}")
    train_log = []
    for epoch in range(1, epochs_eff + 1):
        model.train()
        perm = torch.randperm(n)
        t0 = time.perf_counter()
        tot, nb = 0.0, 0
        total_batches = (n + bs - 1) // bs
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            x_ids, x_pos, x_pad = ids[idx], pos[idx], pad[idx]
            h = model(x_ids, x_pos, x_pad)
            valid = ~x_pad
            # 至少有一个前驱的有效位置才监督（覆盖中间位置与末尾 target 位置）
            sup = valid & (valid.to(torch.long).cumsum(1) >= 2)
            if int(sup.sum()) == 0:
                continue
            b_idx, t_idx = sup.nonzero(as_tuple=True)
            h_rows = h[b_idx, t_idx - 1, :]          # 前一位置隐状态
            labels = x_ids[b_idx, t_idx] - 1         # 词表下标（去掉 pad 行）
            logits = h_rows @ model.item.weight[1:].T
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss_v = float(loss.item())
            if not np.isfinite(loss_v):
                # 数值发散保护：跳过该批，不给 Adam 喂 NaN（否则状态被永久污染）
                opt.zero_grad()
                print(f"[sasrec{tag}]   WARNING: non-finite loss at "
                      f"batch {nb + 1}, skipped")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss_v
            nb += 1
            if log_every and nb % log_every == 0:
                print(f"[sasrec{tag}]   batch {nb}/{total_batches} "
                      f"loss~{tot / nb:.4f}")
        dt = time.perf_counter() - t0
        entry = {"epoch": epoch, "loss": round(tot / max(1, nb), 6),
                 "wall_seconds": round(dt, 1), "batches": nb,
                 "samples_per_second": round(n / dt, 1)}
        train_log.append(entry)
        print(f"[sasrec{tag}] epoch {epoch}/{epochs_eff} "
              f"loss={entry['loss']:.5f} ({dt:.1f}s, "
              f"{entry['samples_per_second']:.0f} samples/s)")

    results = cfg.root / "results"
    (results / "models").mkdir(parents=True, exist_ok=True)
    (results / "experiments").mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "n_items": n_items,
            "max_items": max_items, "max_len": max_len, "config": base,
            "seed": seed_eff, "smoke": smoke}
    ckpt_path = results / "models" / f"sasrec{tag}{suffix}.pt"
    torch.save(ckpt, ckpt_path)
    log = {"variant": "sasrec", "tag": tag, "smoke": smoke, "samples": int(n),
           "n_items": n_items, "max_items": max_items, "seed": seed_eff,
           "epochs": epochs_eff, "config": base, "epochs_log": train_log}
    save_json(log, results / "experiments" / f"train_sasrec{tag}{suffix}.json")
    print(f"[sasrec{tag}] checkpoint -> {ckpt_path}")
    return log
