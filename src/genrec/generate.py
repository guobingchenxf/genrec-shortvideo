"""单用户生成 demo：加载训练好的 SID 模型，对指定评估用户生成 Top-K 推荐。

用法：python -m genrec.cli generate --user-id 0 --topk 10
"""

import numpy as np
import torch

from genrec.models.seqgen import NextTokenLM, SidTokenizer, beam_search_sid


def run(cfg, user_id, topk=10, beam=10):
    import sys
    try:  # Windows 控制台默认 GBK，无法显示中文标题时切换为 UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001, S110 - 仅影响显示编码，失败则保持默认
        pass
    processed = cfg.path("paths", "processed_dir")
    ev = np.load(processed / "eval_contexts.npz")
    z = np.load(processed / "sid_codes.npz")

    ckpt_path = cfg.root / "results" / "models" / "gen_sid.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    tok = SidTokenizer(z["video_ids"], z["codes"],
                       use_actions=bool(ckpt.get("use_actions", True)))
    max_items = int(ckpt["max_items"])
    block = tok.n_levels + (1 if tok.use_actions else 0)
    max_len = int(ckpt.get("max_len")
                  or (1 + max_items * block + tok.n_levels + 8))
    model = NextTokenLM(
        ckpt["vocab_size"],
        d_model=int(ckpt["config"]["d_model"]),
        n_layers=int(ckpt["config"]["n_layers"]),
        n_heads=int(ckpt["config"].get("n_heads", 4)),
        dropout=0.0, max_len=max_len,
        backbone=ckpt["config"]["backbone"])
    model.load_state_dict(ckpt["state_dict"])

    uids = ev["user_ids"].tolist()
    if user_id not in uids:
        raise SystemExit(f"user {user_id} not in eval users "
                         f"(range: {min(uids)}..{max(uids)})")
    i = uids.index(user_id)
    ctx = ev["context_videos"][i]
    act = ev["context_actions"][i]
    mask = ev["context_mask"][i]
    n_hist = int(np.count_nonzero(mask))
    toks = tok.encode_context(ctx, mask, act, ckpt["max_items"])
    ranked = beam_search_sid(model, tok, [toks], beam=beam, batch_users=1)[0]
    ranked = ranked[:topk]

    import pandas as pd
    raw_dir = cfg.path("paths", "raw_dir") / cfg["data"]["dir_name"]
    cap = pd.read_csv(raw_dir / "kuairec_caption_category.csv", dtype=str,
                      keep_default_na=False, engine="python").fillna("")
    title = dict(zip(
        pd.to_numeric(cap["video_id"], errors="coerce").dropna().astype(int),
        cap.loc[pd.to_numeric(cap["video_id"], errors="coerce").notna(), "caption"]))

    print(f"user {user_id}: history={n_hist} behaviors, "
          f"model context tokens={len(toks)}")
    for rank, v in enumerate(ranked, 1):
        t = str(title.get(int(v), ""))[:42]
        print(f"  {rank:2d}. video {int(v):5d}  {t}")
    return ranked
