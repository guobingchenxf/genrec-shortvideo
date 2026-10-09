import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config
from genrec.models.seqgen import (
    NextTokenLM,
    RawTokenizer,
    SidTokenizer,
    build_training_sequences,
)


def time_train(model, x, y, vocab, bs, batches, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=0)
    perm = torch.randperm(len(x))
    for i in range(2):  # 预热
        idx = perm[i * bs:(i + 1) * bs]
        logits = model(x[idx])
        loss = lossf(logits.reshape(-1, vocab), y[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for i in range(batches):
        idx = perm[i * bs:(i + 1) * bs]
        logits = model(x[idx])
        loss = lossf(logits.reshape(-1, vocab), y[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    dt = time.perf_counter() - t0
    return batches * bs / dt


def main():
    cfg = load_config()
    torch.set_num_threads(int(cfg["models"]["gen"]["num_threads"]))
    processed = cfg.path("paths", "processed_dir")
    tr = np.load(processed / "train_samples.npz")
    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    sid_path = processed / "sid_codes.npz"
    if sid_path.exists():
        z = np.load(sid_path)
        sid_ids, codes, sid_src = z["video_ids"], z["codes"], "真实 SID"
    else:
        rng = np.random.default_rng(0)
        sid_ids = video_ids
        codes = rng.integers(0, 256, size=(len(video_ids), 3)).astype(np.int16)
        sid_src = "随机码（正式 SID 未生成时的结构等效计时）"
    n = 20000

    for name, tok, is_sid, max_items in [
        ("sid  (ctx20)", SidTokenizer(sid_ids, codes), True, 20),
        ("sid  (ctx10)", SidTokenizer(sid_ids, codes), True, 10),
        ("raw  (ctx20)", RawTokenizer(video_ids), False, 20),
        ("raw  (ctx10)", RawTokenizer(video_ids), False, 10),
    ]:
        x, y = build_training_sequences(
            tok, tr["context_videos"][:n], tr["context_mask"][:n],
            tr["context_actions"][:n], tr["target_videos"][:n], max_items, is_sid)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        model = NextTokenLM(tok.vocab_size, d_model=128, n_layers=2, n_heads=4,
                            max_len=x.shape[1] + 8, backbone="transformer")
        sps = time_train(model, x, y, tok.vocab_size, bs=256, batches=8)
        print(f"[speed] transformer {name}: seq={x.shape[1]}, "
              f"{sps:.0f} samples/s | 100k 样本 1 epoch ≈ {100000/sps/60:.1f} min")

    tok = RawTokenizer(video_ids)
    x, y = build_training_sequences(
        tok, tr["context_videos"][:n], tr["context_mask"][:n],
        tr["context_actions"][:n], tr["target_videos"][:n], 10, False)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    model = NextTokenLM(tok.vocab_size, d_model=64, n_layers=1,
                        max_len=x.shape[1] + 8, backbone="gru")
    sps = time_train(model, x, y, tok.vocab_size, bs=256, batches=8, lr=2e-3)
    print(f"[speed] gru raw (ctx10): seq={x.shape[1]}, {sps:.0f} samples/s | "
          f"100k 样本 1 epoch ≈ {100000/sps/60:.1f} min")
    print(f"[speed] SID 来源: {sid_src}")


if __name__ == "__main__":
    main()
