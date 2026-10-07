"""E3 + D2 战役：行为 token 消融（E3）+ beam 成本-质量 Pareto（D2），串行跑完。

D2（先跑，快）：gen-sid-b3 在 beam ∈ {1,2,5,10,20} 下逐一做稠密全观测评估；
  beam=50 参考点引用主表（同机同协议、另一会话运行，延迟可比性已在 JSON 注明）。
  结果增量写 results/experiments/beam_pareto.json。
E3（后跑，慢）：训练 sid-b3-na（关闭 watch_ratio 行为 token，其余同 sid-b3，seed42 × 1 epoch），
  随后双协议评估（稠密 1,411 用户 + 留一 100 负采样 2,000 用户），与 sid-b3 对照；
  结果增量写 results/experiments/e3_behavior_token_ablation.json。

进程存活期间请求系统保持唤醒（Windows SetThreadExecutionState，不改系统设置）；
单个步骤失败会记录错误并继续后续步骤。
"""

import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec import train as trainer
from genrec.config import load_config
from genrec.eval import run_eval, sampled
from genrec.eval.metrics import evaluate_lists
from genrec.utils.monitor import save_json

BEAMS = [1, 2, 5, 10, 20]
E3_VARIANT = "sid-b3-na"


def _keep_awake():
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required)
        print(f"[e3d2] keep-awake: {'OK' if ok else 'FAILED'}")
    except Exception as exc:  # noqa: BLE001 - 非 Windows 平台忽略即可
        print(f"[e3d2] keep-awake skipped: {exc}")


def _dense_metrics(cfg, variant, ckpt_name):
    (suffix, user_ids, contexts, actions, mask, targets, video_ids
     ) = run_eval._load_eval(cfg, smoke=False)
    bans = [{int(v) for v in contexts[i][mask[i]]}
            for i in range(len(contexts))]
    raw_lists = run_eval._generate_lists(
        cfg, variant, contexts, actions, mask, video_ids, suffix,
        ckpt_name=ckpt_name)
    lists = [[v for v in r if v not in bans[i]][:50]
             for i, r in enumerate(raw_lists)]
    m = evaluate_lists(lists, targets, user_ids, len(video_ids))
    return {"recall@10": round(m["recall@10"], 6),
            "recall@50": round(m["recall@50"], 6),
            "ndcg@10": round(m["ndcg@10"], 6),
            "n_users": len(user_ids)}


def _sampled_metrics(cfg, variant, ckpt_name, max_users=2000):
    data = sampled.load_loo_data(cfg, n_neg=100, seed=42, max_users=max_users)
    scores = sampled.score_generator(cfg, variant, data, ckpt_name=ckpt_name)
    hr, ndcg = sampled.hr_ndcg_at_10(scores)
    return {"hr@10": round(hr, 6), "ndcg@10": round(ndcg, 6),
            "n_users": len(data["users"])}


def _run_d2(cfg, exp_dir):
    out_path = exp_dir / "beam_pareto.json"
    state = {
        "note": ("D2：gen-sid-b3 的 beam 成本-质量 Pareto。各点在同一会话/同线程数"
                 "下实测；beam=50 参考点来自主表（另一会话，延迟可比性注明）。"),
        "method": "gen-sid-b3",
        "protocol": "dense full-observation (1,411 users)",
        "threads": int(cfg["models"]["gen"]["num_threads"]),
        "points": {},
        "total_wall_seconds": None,
    }

    def flush():
        save_json(state, out_path)

    flush()
    t0 = time.perf_counter()
    for beam in BEAMS:
        print(f"\n[d2] beam={beam} @{time.strftime('%H:%M:%S')}")
        try:
            res = run_eval.run(cfg, ["gen-sid-b3"], beam=beam)
            m = res[f"gen-sid-b3@beam{beam}"]
            state["points"][str(beam)] = {
                "recall@10": m["recall@10"], "recall@50": m["recall@50"],
                "ndcg@10": m["ndcg@10"], "coverage@50": m["coverage@50"],
                "ms_per_user": m["ms_per_user"],
                "eval_wall_seconds": m["eval_wall_seconds"],
                "source": "this_run"}
        except Exception as exc:  # noqa: BLE001 - 记录并继续
            state["points"][str(beam)] = {"error": repr(exc)}
            print(f"[d2] beam={beam} FAILED: {exc}")
        state["total_wall_seconds"] = round(time.perf_counter() - t0, 1)
        flush()
    mt = json.loads((exp_dir / "main_table.json").read_text(encoding="utf-8"))
    ref = mt["gen-sid-b3"]
    state["points"]["50"] = {
        "recall@10": ref["recall@10"], "recall@50": ref["recall@50"],
        "ndcg@10": ref["ndcg@10"], "coverage@50": ref["coverage@50"],
        "ms_per_user": ref["ms_per_user"],
        "eval_wall_seconds": ref["eval_wall_seconds"],
        "source": "main_table（beam=50 主实验，另一会话）"}
    state["total_wall_seconds"] = round(time.perf_counter() - t0, 1)
    flush()
    print(f"[d2] DONE in {state['total_wall_seconds']:.0f}s")


def _run_e3(cfg, exp_dir):
    out_path = exp_dir / "e3_behavior_token_ablation.json"
    state = {
        "note": ("E3：行为 token 消融。sid-b3-na = 关闭 watch_ratio 行为 token，"
                 "其余与 sid-b3 完全一致（configs/default.yaml）。"),
        "variant": E3_VARIANT,
        "base_variant": "sid-b3",
        "training": {},
        "eval": {},
        "reference_sid-b3": {},
        "total_wall_seconds": None,
    }

    def flush():
        save_json(state, out_path)

    flush()
    t0 = time.perf_counter()
    print(f"\n[e3] train {E3_VARIANT} @{time.strftime('%H:%M:%S')}")
    try:
        log = trainer.run_gen(cfg, E3_VARIANT, seed=42, epochs=1, tag="")
        state["training"] = {
            "seed": 42, "epochs": 1,
            "seq_len": log["seq_len"], "vocab_size": log["vocab_size"],
            "use_actions": bool(log["config"].get("use_actions", True)),
            "epochs_log": log["epochs_log"]}
    except Exception as exc:  # noqa: BLE001 - 记录并继续
        state["training"] = {"error": repr(exc),
                             "traceback": traceback.format_exc()[-800:]}
        print(f"[e3] train FAILED: {exc}")
    flush()

    ref_dense = json.loads(
        (exp_dir / "main_table.json").read_text(encoding="utf-8"))["gen-sid-b3"]
    ref_sampled = json.loads(
        (exp_dir / "sampled_protocol_n2000.json"
         ).read_text(encoding="utf-8"))["results"]["gen-sid-b3"]
    state["reference_sid-b3"] = {
        "dense": {k: ref_dense[k]
                  for k in ("recall@10", "recall@50", "ndcg@10")},
        "sampled": {"hr@10": ref_sampled["hr@10"],
                    "ndcg@10": ref_sampled["ndcg@10"]}}

    ckpt_name = f"gen_{E3_VARIANT}.pt"
    for proto, fn in (("dense", _dense_metrics), ("sampled", _sampled_metrics)):
        print(f"\n[e3] {proto} eval @{time.strftime('%H:%M:%S')}")
        try:
            state["eval"][proto] = fn(cfg, E3_VARIANT, ckpt_name)
        except Exception as exc:  # noqa: BLE001 - 记录并继续
            state["eval"][proto] = {"error": repr(exc)}
            print(f"[e3] {proto} eval FAILED: {exc}")
        flush()

    state["total_wall_seconds"] = round(time.perf_counter() - t0, 1)
    flush()
    print(f"[e3] DONE in {state['total_wall_seconds']:.0f}s")


def main():
    _keep_awake()
    cfg = load_config()
    torch.set_num_threads(int(cfg["models"]["gen"]["num_threads"]))
    exp_dir = cfg.root / "results" / "experiments"
    t_start = time.perf_counter()
    _run_d2(cfg, exp_dir)
    _run_e3(cfg, exp_dir)
    total = round(time.perf_counter() - t_start, 1)
    print(f"\n[e3d2] ALL DONE in {total:.0f}s")


if __name__ == "__main__":
    main()
