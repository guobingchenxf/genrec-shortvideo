"""D1 补充：同条件对照实验（同一进程、同一时段）。

1) gen-sid-b3 @ beam50：前 128 用户 naive vs cached 背靠背计时（真实速度比）；
2) cached 模式下重测 beam {1,2,5,10,20} 全量（同会话 Pareto，写回 beam_pareto.json）。

输出：results/experiments/d1_latency_control.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config
from genrec.eval import run_eval
from genrec.models.seqgen import (
    NextTokenLM,
    SidTokenizer,
    beam_search_sid,
    beam_search_sid_naive,
)

N_USERS = 128
BEAMS = [1, 2, 5, 10, 20]


def load_b3(cfg):
    proc = cfg.path("paths", "processed_dir")
    ckpt = torch.load(cfg.root / "results" / "models" / "gen_sid-b3.pt",
                      weights_only=False)
    z = np.load(proc / "sid_codes_b3.npz")
    tok = SidTokenizer(z["video_ids"], z["codes"],
                       use_actions=bool(ckpt.get("use_actions", True)))
    model = NextTokenLM(ckpt["vocab_size"],
                        d_model=int(ckpt["config"]["d_model"]),
                        n_layers=int(ckpt["config"]["n_layers"]),
                        n_heads=int(ckpt["config"].get("n_heads", 4)),
                        dropout=0.0, max_len=int(ckpt["max_len"]),
                        backbone=ckpt["config"]["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, tok, ckpt


def main():
    cfg = load_config()
    torch.set_num_threads(4)
    _, _, contexts, actions, mask, _, _ = run_eval._load_eval(cfg, smoke=False)
    model, tok, ckpt = load_b3(cfg)
    ctx_lists = [tok.encode_context(contexts[i], mask[i], actions[i],
                                    ckpt["max_items"])
                 for i in range(N_USERS)]

    t0 = time.perf_counter()
    a = beam_search_sid_naive(model, tok, ctx_lists, beam=50, batch_users=32)
    t1 = time.perf_counter()
    b = beam_search_sid(model, tok, ctx_lists, beam=50, batch_users=32)
    t2 = time.perf_counter()
    eq = sum(1 for x, y in zip(a, b) if x == y)
    naive_ms = (t1 - t0) * 1000.0 / N_USERS
    cached_ms = (t2 - t1) * 1000.0 / N_USERS

    out = {
        "note": "同进程/同段对照：naive=beam_search_sid_naive，cached=D1 前缀缓存解码",
        "method": "gen-sid-b3", "beam": 50, "n_users": N_USERS,
        "naive_ms_per_user": round(naive_ms, 2),
        "cached_ms_per_user": round(cached_ms, 2),
        "speedup": round(naive_ms / cached_ms, 2),
        "lists_identical": f"{eq}/{N_USERS}",
        "beam_points_cached_full": {},
    }
    print(f"[ctl] {N_USERS} users: naive={naive_ms:.1f}ms cached={cached_ms:.1f}ms "
          f"speedup={naive_ms / cached_ms:.2f}x identical={eq}/{N_USERS}")

    for bm in BEAMS:
        res = run_eval.run(cfg, ["gen-sid-b3"], beam=bm)
        m = res[f"gen-sid-b3@beam{bm}"]
        out["beam_points_cached_full"][str(bm)] = {
            "ms_per_user": m["ms_per_user"],
            "eval_wall_seconds": m["eval_wall_seconds"]}
        print(f"[ctl] beam{bm}: {m['ms_per_user']} ms/user")

    exp = cfg.root / "results" / "experiments"
    (exp / "d1_latency_control.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    pj = json.loads((exp / "beam_pareto.json").read_text(encoding="utf-8"))
    for bm, v in out["beam_points_cached_full"].items():
        pj["points"][bm]["ms_per_user"] = v["ms_per_user"]
        pj["points"][bm]["eval_wall_seconds"] = v["eval_wall_seconds"]
    pj["note"] = (pj["note"].split("；同会话重测")[0]
                  + "；同会话重测（d1_latency_control.json）")
    (exp / "beam_pareto.json").write_text(
        json.dumps(pj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ctl] -> {exp / 'd1_latency_control.json'}")
    print("[ctl] DONE")


if __name__ == "__main__":
    main()
