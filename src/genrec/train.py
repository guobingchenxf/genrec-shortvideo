"""训练入口：RQ-VAE（语义 ID）与生成式序列模型（变体清单见 VARIANTS）。

运行方式见 cli.py：`python -m genrec.cli train-rqvae` / `train-gen --variant sid`。
所有检查点写入 results/models/（不入库），训练日志写入 results/experiments/（入库）。
"""

import time

import numpy as np
import torch

from genrec.models import rqvae as rqvae_mod
from genrec.models.seqgen import (
    NextTokenLM,
    RawTokenizer,
    SidTokenizer,
    build_training_sequences,
)
from genrec.utils.monitor import save_json

VARIANTS = {"sid": True, "sid-v2": True, "sid-b1": True, "sid-b3": True,
            "sid-b3-na": True, "raw": False, "raw-gru": False}
SID_FILES = {"sid": "sid_codes", "sid-v2": "sid_codes_v2",
             "sid-b1": "sid_codes_b1", "sid-b3": "sid_codes_b3",
             "sid-b3-na": "sid_codes_b3"}


def train_rqvae_main(cfg, smoke=False, tag=None):
    return rqvae_mod.run(cfg, smoke=smoke, tag=tag)


def _load_processsed(cfg, smoke, variant):
    processed = cfg.path("paths", "processed_dir")
    suffix = "_smoke" if smoke else ""
    train_npz = np.load(processed / f"train_samples{suffix}.npz")
    sid_file = SID_FILES.get(variant, "sid_codes")
    sid_npz = np.load(processed / f"{sid_file}{suffix}.npz")
    return suffix, train_npz, sid_npz["video_ids"], sid_npz["codes"]


def _build_tokenizer(variant, video_ids, codes, use_actions=True):
    if VARIANTS[variant]:  # is_sid：sid 与 sid-v2 都走语义 ID 词表
        return SidTokenizer(video_ids, codes, use_actions=use_actions)
    return RawTokenizer(video_ids)


def run_gen(cfg, variant, smoke=False, seed=None, epochs=None, tag="",
            val_eval=True, log_every=100):
    """训练生成式变体。

    扩展参数（供过夜战役等使用，默认行为不变）：
    - seed / epochs：覆盖配置中的随机种子与训练轮数；
    - tag：附加到检查点与日志文件名（如 "-e3" / "-s43"）；
    - val_eval：每轮结束后在 val_samples 上计算验证损失（写进日志）；
    - log_every：每 N 个 batch 打印一次进度（0 = 关闭）。
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}, expect {list(VARIANTS)}")
    is_sid = VARIANTS[variant]
    base = dict(cfg["models"]["gen"])
    merged = {**base, **dict(base["variants"][variant])}
    merged.pop("variants", None)
    seed_eff = int(seed if seed is not None else merged["seed"])
    torch.manual_seed(seed_eff)
    np.random.seed(seed_eff)
    torch.set_num_threads(int(merged.get("num_threads", 8)))
    device = "cpu"

    suffix, train_npz, video_ids, codes = _load_processsed(cfg, smoke, variant)
    use_actions = bool(merged.get("use_actions", True))
    tok = _build_tokenizer(variant, video_ids, codes, use_actions=use_actions)
    max_items = int(merged["max_context_items"])

    inputs, labels = build_training_sequences(
        tok,
        train_npz["context_videos"], train_npz["context_mask"],
        train_npz["context_actions"], train_npz["target_videos"],
        max_items, is_sid)
    x = torch.from_numpy(inputs)
    y = torch.from_numpy(labels)
    n = len(x)
    bs = int(merged["batch_size"])
    epochs_eff = int(epochs if epochs is not None else merged["epochs"])
    if smoke:
        n = min(n, 2048)
        bs = min(bs, 64)
        epochs_eff = 1

    # 验证集（收敛诊断用；文件缺失时跳过，不报错）
    val_x = val_y = None
    if val_eval:
        val_path = cfg.path("paths", "processed_dir") / f"val_samples{suffix}.npz"
        if val_path.exists():
            va = np.load(val_path)
            vx, vy = build_training_sequences(
                tok, va["context_videos"], va["context_mask"],
                va["context_actions"], va["target_videos"], max_items, is_sid)
            val_x, val_y = torch.from_numpy(vx), torch.from_numpy(vy)

    model = NextTokenLM(
        tok.vocab_size,
        d_model=int(merged["d_model"]),
        n_layers=int(merged["n_layers"]),
        n_heads=int(merged.get("n_heads", 4)),
        dropout=float(merged.get("dropout", 0.1)),
        max_len=int(inputs.shape[1] + 8),
        backbone=merged["backbone"])
    opt = torch.optim.Adam(model.parameters(), lr=float(merged["lr"]))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    print(f"[gen:{variant}{tag}] samples={n}, seq_len={inputs.shape[1]}, "
          f"vocab={tok.vocab_size}, backbone={merged['backbone']}, "
          f"seed={seed_eff}, epochs={epochs_eff}, "
          f"val_samples={0 if val_x is None else len(val_x)}")
    train_log = []
    for epoch in range(1, epochs_eff + 1):
        perm = torch.randperm(n)
        t0 = time.perf_counter()
        tot, nb = 0.0, 0
        total_batches = (n + bs - 1) // bs
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            logits = model(x[idx])
            loss = loss_fn(logits.reshape(-1, tok.vocab_size),
                           y[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
            if log_every and nb % log_every == 0:
                print(f"[gen:{variant}{tag}]   batch {nb}/{total_batches} "
                      f"loss~{tot / nb:.4f}")
        dt = time.perf_counter() - t0
        entry = {"epoch": epoch, "loss": round(tot / max(1, nb), 6),
                 "wall_seconds": round(dt, 1), "batches": nb,
                 "samples_per_second": round(n / dt, 1)}
        if val_x is not None:
            model.eval()
            tot_v, nb_v = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(val_x), 512):
                    logits = model(val_x[s:s + 512])
                    vl = loss_fn(logits.reshape(-1, tok.vocab_size),
                                 val_y[s:s + 512].reshape(-1))
                    tot_v += vl.item()
                    nb_v += 1
            entry["val_loss"] = round(tot_v / max(1, nb_v), 6)
            model.train()
        train_log.append(entry)
        val_part = (f"val={entry['val_loss']:.5f} "
                    if "val_loss" in entry else "")
        print(f"[gen:{variant}{tag}] epoch {epoch}/{epochs_eff} "
              f"loss={entry['loss']:.5f} {val_part}({dt:.1f}s, "
              f"{entry['samples_per_second']:.0f} samples/s)")

    results = cfg.root / "results"
    (results / "models").mkdir(parents=True, exist_ok=True)
    (results / "experiments").mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "variant": variant,
        "is_sid": is_sid,
        "config": merged,
        "vocab_size": tok.vocab_size,
        "max_items": max_items,
        "max_len": int(inputs.shape[1] + 8),
        "seed": seed_eff,
        "device": device,
        "use_actions": use_actions,
    }
    ckpt_path = results / "models" / f"gen_{variant}{tag}{suffix}.pt"
    torch.save(ckpt, ckpt_path)
    log = {"variant": variant, "tag": tag, "smoke": smoke, "samples": int(n),
           "seq_len": int(inputs.shape[1]), "vocab_size": tok.vocab_size,
           "max_items": max_items, "device": device, "seed": seed_eff,
           "epochs": epochs_eff,
           "val_samples": 0 if val_x is None else len(val_x),
           "config": merged, "epochs_log": train_log}
    save_json(log, results / "experiments"
              / f"train_gen_{variant}{tag}{suffix}.json")
    print(f"[gen:{variant}{tag}] checkpoint -> {ckpt_path}")
    return log
