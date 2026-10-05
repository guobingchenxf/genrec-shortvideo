"""训练入口：RQ-VAE（语义 ID）与生成式序列模型（SID / raw / raw-gru 三个变体）。

运行方式见 cli.py：`python -m genrec.cli train-rqvae` / `train-gen --variant sid`。
所有检查点写入 results/models/（不入库），训练日志写入 results/experiments/（入库）。
"""

import time

import numpy as np
import torch

from genrec.models import rqvae as rqvae_mod
from genrec.models.seqgen import (NextTokenLM, RawTokenizer, SidTokenizer,
                                  build_training_sequences)
from genrec.utils.monitor import save_json

VARIANTS = {"sid": True, "sid-v2": True, "raw": False, "raw-gru": False}
SID_FILES = {"sid": "sid_codes", "sid-v2": "sid_codes_v2"}


def train_rqvae_main(cfg, smoke=False, tag=None):
    return rqvae_mod.run(cfg, smoke=smoke, tag=tag)


def _load_processsed(cfg, smoke, variant):
    processed = cfg.path("paths", "processed_dir")
    suffix = "_smoke" if smoke else ""
    train_npz = np.load(processed / f"train_samples{suffix}.npz")
    sid_file = SID_FILES.get(variant, "sid_codes")
    sid_npz = np.load(processed / f"{sid_file}{suffix}.npz")
    return suffix, train_npz, sid_npz["video_ids"], sid_npz["codes"]


def _build_tokenizer(variant, video_ids, codes):
    if VARIANTS[variant]:  # is_sid：sid 与 sid-v2 都走语义 ID 词表
        return SidTokenizer(video_ids, codes)
    return RawTokenizer(video_ids)


def run_gen(cfg, variant, smoke=False):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}, expect {list(VARIANTS)}")
    is_sid = VARIANTS[variant]
    base = dict(cfg["models"]["gen"])
    merged = {**base, **dict(base["variants"][variant])}
    merged.pop("variants", None)
    seed = int(merged["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(merged.get("num_threads", 8)))
    device = "cpu"

    suffix, train_npz, video_ids, codes = _load_processsed(cfg, smoke, variant)
    tok = _build_tokenizer(variant, video_ids, codes)
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
    epochs = int(merged["epochs"])
    if smoke:
        n = min(n, 2048)
        bs = min(bs, 64)
        epochs = 1

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

    print(f"[gen:{variant}] samples={n}, seq_len={inputs.shape[1]}, "
          f"vocab={tok.vocab_size}, backbone={merged['backbone']}")
    train_log = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        t0 = time.perf_counter()
        tot, nb = 0.0, 0
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
        dt = time.perf_counter() - t0
        entry = {"epoch": epoch, "loss": round(tot / max(1, nb), 6),
                 "wall_seconds": round(dt, 1), "batches": nb,
                 "samples_per_second": round(n / dt, 1)}
        train_log.append(entry)
        print(f"[gen:{variant}] epoch {epoch}/{epochs} "
              f"loss={entry['loss']:.5f} ({dt:.1f}s, "
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
        "seed": seed,
        "device": device,
    }
    ckpt_path = results / "models" / f"gen_{variant}{suffix}.pt"
    torch.save(ckpt, ckpt_path)
    log = {"variant": variant, "smoke": smoke, "samples": int(n),
           "seq_len": int(inputs.shape[1]), "vocab_size": tok.vocab_size,
           "max_items": max_items, "device": device, "seed": seed,
           "config": merged, "epochs_log": train_log}
    save_json(log, results / "experiments" / f"train_gen_{variant}{suffix}.json")
    print(f"[gen:{variant}] checkpoint -> {ckpt_path}")
    return log
