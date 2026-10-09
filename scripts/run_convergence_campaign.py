import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec import train as trainer
from genrec.config import load_config
from genrec.eval import run_eval, sampled
from genrec.eval.metrics import evaluate_lists
from genrec.utils.monitor import save_json

VARIANTS = ["sid-b3", "raw"]


def _keep_awake():
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required)
        print(f"[campaign] keep-awake: {'OK' if ok else 'FAILED'}")
    except Exception as exc:  # noqa: BLE001 - 非 Windows 平台忽略即可
        print(f"[campaign] keep-awake skipped: {exc}")


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


def main():
    _keep_awake()
    t_start = time.perf_counter()
    cfg = load_config()
    out_path = (cfg.root / "results" / "experiments"
                / "convergence_campaign.json")
    state = {
        "note": ("P0：A1 收敛（-e3，seed42×3 epoch）+ E4 种子方差（-s43/-s44）；"
                 "seed42×1ep 基线引用主表（gen_sid-b3 / gen_raw 主检查点）"),
        "training_logs": {},
        "eval": {},
        "summary": {},
    }

    def flush():
        save_json(state, out_path)

    plan = [(v, "-e3", 42, 3) for v in VARIANTS]
    for seed in (43, 44):
        plan += [(v, f"-s{seed}", seed, 1) for v in VARIANTS]

    # ---- 训练 ----
    for variant, tag, seed, epochs in plan:
        key = f"{variant}{tag}"
        print(f"\n[campaign] train {key} (seed={seed}, epochs={epochs}) "
              f"@{time.strftime('%H:%M:%S')}")
        try:
            log = trainer.run_gen(cfg, variant, seed=seed, epochs=epochs,
                                  tag=tag)
            state["training_logs"][key] = {
                "seed": seed, "epochs": epochs,
                "epochs_log": log["epochs_log"]}
        except Exception as exc:  # noqa: BLE001 - 记录并继续
            state["training_logs"][key] = {
                "error": repr(exc), "traceback": traceback.format_exc()[-800:]}
            print(f"[campaign] train {key} FAILED: {exc}")
        flush()

    # ---- 评估 ----
    for variant, tag, _seed, _epochs in plan:
        key = f"{variant}{tag}"
        ckpt_name = f"gen_{variant}{tag}.pt"
        for proto, fn in (("dense", _dense_metrics),
                          ("sampled", _sampled_metrics)):
            print(f"\n[campaign] {proto} eval {key} @{time.strftime('%H:%M:%S')}")
            try:
                state["eval"].setdefault(key, {})[proto] = fn(
                    cfg, variant, ckpt_name)
            except Exception as exc:  # noqa: BLE001 - 记录并继续
                state["eval"].setdefault(key, {})[proto] = {"error": repr(exc)}
                print(f"[campaign] {proto} eval {key} FAILED: {exc}")
            flush()

    # ---- seed42×1ep 基线（引用已有主表结果） ----
    main_table = json.loads(
        (cfg.root / "results" / "experiments" / "main_table.json"
         ).read_text(encoding="utf-8"))
    sampled_tbl = json.loads(
        (cfg.root / "results" / "experiments" / "sampled_protocol_n2000.json"
         ).read_text(encoding="utf-8"))["results"]
    ref = {
        "sid-b3": {
            "dense": {k: main_table["gen-sid-b3"][k]
                      for k in ("recall@10", "recall@50", "ndcg@10")},
            "sampled": {"hr@10": sampled_tbl["gen-sid-b3"]["hr@10"],
                        "ndcg@10": sampled_tbl["gen-sid-b3"]["ndcg@10"]}},
        "raw": {
            "dense": {k: main_table["gen-raw"][k]
                      for k in ("recall@10", "recall@50", "ndcg@10")},
            "sampled": {"hr@10": sampled_tbl["gen-raw"]["hr@10"],
                        "ndcg@10": sampled_tbl["gen-raw"]["ndcg@10"]}},
    }
    state["seed42_epoch1_reference"] = ref

    # ---- 汇总：三种子配对差（sid-b3 − raw） ----
    deltas = {"dense_recall@50": [], "dense_ndcg@10": [], "sampled_hr@10": []}
    for tag in (None, "-s43", "-s44"):
        if tag is None:
            b3, raw = ref["sid-b3"], ref["raw"]
        else:
            b3 = state["eval"].get(f"sid-b3{tag}", {})
            raw = state["eval"].get(f"raw{tag}", {})
        if "dense" not in b3 or "dense" not in raw \
                or "sampled" not in b3 or "sampled" not in raw:
            continue
        deltas["dense_recall@50"].append(
            round(b3["dense"]["recall@50"] - raw["dense"]["recall@50"], 6))
        deltas["dense_ndcg@10"].append(
            round(b3["dense"]["ndcg@10"] - raw["dense"]["ndcg@10"], 6))
        deltas["sampled_hr@10"].append(
            round(b3["sampled"]["hr@10"] - raw["sampled"]["hr@10"], 6))
    state["summary"] = {
        "delta_definition": "sid-b3 减 raw（同种子、同预算；负值=SID 落后）",
        "seeds": [42, 43, 44],
        "deltas": deltas,
        "delta_mean": {k: round(float(np.mean(v)), 6) for k, v in deltas.items()}
        if all(deltas.values()) else {},
        "delta_min": {k: round(float(np.min(v)), 6) for k, v in deltas.items()}
        if all(deltas.values()) else {},
        "delta_max": {k: round(float(np.max(v)), 6) for k, v in deltas.items()}
        if all(deltas.values()) else {},
    }
    state["total_wall_seconds"] = round(time.perf_counter() - t_start, 1)
    flush()
    print(f"\n[campaign] DONE in {state['total_wall_seconds']:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
