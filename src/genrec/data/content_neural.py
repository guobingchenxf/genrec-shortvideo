"""B1：神经内容编码器（bge-small-zh-v1.5）→ 64 维内容特征。

- 文本组装复用 `preprocess.build_video_texts`（与 TF-IDF 路线完全同源，
  保证 B2 特征消融只有"编码器"这一个变量）；
- 编码：CLS 池化 + L2 归一化（bge 的官方用法）；CPU 批量推理；
- 降维：PCA 到 64 维（与 TF-IDF+SVD 路线的 64 维对齐）；
- 产物：data/processed/content_feats_bge.npz（video_ids + feats 64d）
  与 results/experiments/content_features_bge.json（含实测覆盖率与耗时）。

模型文件放在 data/models/bge-small-zh-v1.5/（不入库），来源见 docs/论文与出处.md。
"""

import time

import numpy as np

from genrec.data.preprocess import build_video_texts
from genrec.utils.monitor import save_json

ENCODER_NAME = "BAAI/bge-small-zh-v1.5"


def load_encoder(encoder_dir):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(encoder_dir))
    model = AutoModel.from_pretrained(str(encoder_dir))
    model.eval()
    torch.set_num_threads(4)
    return tok, model


def encode_texts(corpus, tok, model, batch_size=64, max_length=128):
    """CLS 池化 + L2 归一化，返回 (N, dim) float32。"""
    import torch
    import torch.nn.functional as F

    embs = []
    with torch.no_grad():
        for s in range(0, len(corpus), batch_size):
            batch = corpus[s:s + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            out = model(**enc)
            cls = out.last_hidden_state[:, 0]
            cls = F.normalize(cls, dim=-1)
            embs.append(cls.numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def run(cfg, out_name="content_feats_bge.npz", pca_dim=64, batch_size=64,
        max_length=128):
    from sklearn.decomposition import PCA

    processed = cfg.path("paths", "processed_dir")
    raw_dir = cfg.path("paths", "raw_dir") / cfg["data"]["dir_name"]
    encoder_dir = cfg.root / "data" / "models" / "bge-small-zh-v1.5"
    if not encoder_dir.exists():
        raise FileNotFoundError(
            f"encoder not found: {encoder_dir}; download per docs/论文与出处.md")

    video_ids = np.load(processed / "video_vocab.npz")["video_ids"]
    ids, corpus, coverage = build_video_texts(raw_dir, video_ids)
    print(f"[bge] videos={len(ids)}, coverage={coverage}")

    t0 = time.perf_counter()
    tok, model = load_encoder(encoder_dir)
    embs = encode_texts(corpus, tok, model, batch_size=batch_size,
                        max_length=max_length)
    enc_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pca = PCA(n_components=pca_dim, random_state=0)
    feats = pca.fit_transform(embs).astype(np.float32)
    pca_seconds = time.perf_counter() - t1

    np.savez_compressed(processed / out_name, video_ids=ids, feats=feats)
    stats = {
        "encoder": ENCODER_NAME,
        "encoder_dir": str(encoder_dir.relative_to(cfg.root)),
        "encoding": "CLS pooling + L2 normalize (bge official usage)",
        "embedding_dim": int(embs.shape[1]),
        "pca_dim": int(pca_dim),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_videos": len(ids),
        "coverage": coverage,
        "encode_seconds": round(enc_seconds, 1),
        "pca_seconds": round(pca_seconds, 1),
        "batch_size": batch_size,
        "max_length": max_length,
        "device": "cpu",
    }
    save_json(stats, cfg.root / "results" / "experiments"
              / "content_features_bge.json")
    print(f"[bge] embedding {embs.shape} in {enc_seconds:.0f}s; "
          f"PCA->{pca_dim}d evr={stats['pca_explained_variance']:.3f}")
    print(f"[bge] -> {processed / out_name}")
    return stats
