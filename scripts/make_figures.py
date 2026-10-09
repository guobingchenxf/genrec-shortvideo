import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genrec.config import load_config

FONTS = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]


def _draw_box(ax, x, y, w, h, fc, ec, title, lines, tcolor):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
        linewidth=1.5, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    ax.text(x + 22, y + 14, title, fontsize=13, fontweight="bold",
            color=tcolor, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 22, y + 44 + i * 19, line, fontsize=10.2,
                color="#475569", va="top")


def _draw_arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops={"arrowstyle": "-|>", "color": "#94a3b8",
                                "lw": 2.0, "shrinkA": 0, "shrinkB": 0})


def plot_architecture(cfg):
    out = cfg.root / "docs" / "assets" / "architecture.png"
    plt.rcParams["font.sans-serif"] = FONTS
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(12.4, 8.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1240)
    ax.set_ylim(800, 0)
    ax.axis("off")

    _draw_box(ax, 40, 30, 1160, 84, "#f8fafc", "#cbd5e1",
              "KuaiRec 数据集（快手短视频，CIKM 2022）",
              [("大矩阵 12.53M 交互 / 7,176 用户 / 10,728 视频（训练）｜"
               "小矩阵 99.6% 全观测（无偏评估，1,411 用户）"),
               ("预处理：全局时间切分 · 150k 序列样本 + 50k 验证 · "
               "watch_ratio 5 档行为 token · 防泄漏剔除 898,914 对")],
              "#0f172a")
    _draw_box(ax, 40, 150, 560, 104, "#eff6ff", "#93c5fd",
              "内容特征 → 语义 ID（RQ-VAE）",
              ["内容特征：TF-IDF+SVD64 或 bge-small-zh（PCA64）",
               "RQ-VAE：3 级 × 1024 码本（STE + 死码重启）",
               "b3：碰撞额外码位 → 10,728 个唯一 SID（碰撞率 39.2% → 0%）"],
              "#1e40af")
    _draw_box(ax, 640, 150, 560, 104, "#f0f9ff", "#7dd3fc",
              "序列样本（因果 LM 格式）",
              ["[BOS] + Σ（SID token + 行为 token）+ 目标 SID",
               "上下文 ≤ 20 行为 · 目标即下一行为（防泄漏）",
               "消融：行为 token on/off（b3-na）· 自回归误差诊断"],
              "#0369a1")
    _draw_box(ax, 40, 290, 700, 104, "#eef2ff", "#a5b4fc",
              "生成式序列模型（因果 Transformer，全词表 CE，CPU）",
              [("SID 变体：sid / v2 / b1 / b3 / b3-na ｜ 原生 ID 对照：gen-raw ｜ "
               "轻量基线：gru-raw"),
               ("受控实验：唯一变量链 · 3 epoch 收敛对照 · "
               "三种子方差（同预算配对差）"),
               "训练预算：150k 样本 × 1 epoch 主表（4 线程实测，无 GPU）"],
              "#4338ca")
    _draw_box(ax, 780, 290, 420, 104, "#f5f3ff", "#c4b5fd",
              "判别式基线",
              ["GRU4Rec-lite（gru-raw，1 epoch）",
               "SASRec-lite（同数据预算，阴性结果 +",
               "回声陷阱诊断链，详见实验报告 §6.9）"],
              "#6d28d9")
    _draw_box(ax, 40, 430, 1160, 84, "#ecfeff", "#67e8f9",
              "推理与解码",
              [("trie 受限解码 + beam search ｜ "
               "D1：context 前缀 K/V 缓存解码（与参考实现逐元素等价验证）"),
               ("beam 成本-质量 Pareto（beam 1 → 50：top-10 质量在 beam ≈ 20 "
               "饱和，延迟降约一半）")],
              "#0e7490")
    _draw_box(ax, 40, 550, 1160, 104, "#fffbeb", "#fcd34d",
              "双协议评估（方法学核心）",
              [("① 稠密全观测：1,411 用户 · recall@10/50 · ndcg@10 · coverage · "
               "ILD / 新颖性（E6）"),
               ("② 标准留一 + 100 负采样：2,000 用户 · HR@10 / NDCG@10"
               "（教师强制序列似然打分）"),
               "③ 分桶与“上新物品”口径（E5）· 种子方差 · 收敛/预算对照"],
              "#92400e")
    _draw_box(ax, 40, 690, 1160, 76, "#f0fdf4", "#86efac",
              "研究闭环（全部如实记录，含阴性结果）",
              [("负结果 → 碰撞诊断 → b3 修复 → 双协议追平、3 epoch 反超 ｜ "
               "长尾结论自我修正 ｜ SASRec 阴性诊断 ｜ 行为 token 消融为负")],
              "#166534")

    for y1, y2 in [(116, 146), (256, 286), (396, 426), (516, 546),
                   (656, 686)]:
        _draw_arrow(ax, 620, y1, y2)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


def main():
    cfg = load_config()
    plot_architecture(cfg)
    exp = cfg.root / "results" / "experiments"
    out_dir = cfg.root / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    mt = json.loads((exp / "main_table.json").read_text(encoding="utf-8"))
    raw = mt["gen-raw"]["recall@50"]

    def collision(fname):
        return json.loads((exp / fname).read_text(encoding="utf-8"))[
            "collision_rate"]

    points = [
        ("gen-sid（v1）", collision("sid_stats.json"), mt["gen-sid"]["recall@50"]),
        ("gen-sid-v2", collision("sid_stats_v2.json"), mt["gen-sid-v2"]["recall@50"]),
        ("gen-sid-b1", collision("sid_stats_b1.json"), mt["gen-sid-b1"]["recall@50"]),
        ("gen-sid-b3", 0.0, mt["gen-sid-b3"]["recall@50"]),
    ]
    xs = [p[1] * 100 for p in points]
    ys = [p[2] / raw * 100 for p in points]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=200)
    ax.plot(xs, ys, "-o", color="#2563eb", lw=2.4, ms=9, zorder=3)
    ax.axhline(100, color="#9ca3af", ls="--", lw=1.2, zorder=1)
    ax.text(26, 101.8, "gen-raw 对照基线（100%）", color="#6b7280", fontsize=9)
    offsets = [(10, -6), (10, -6), (10, -6), (12, 10)]
    for (name, _c, _r), x, y, off in zip(points, xs, ys, offsets):
        ax.annotate(f"{name}（{y:.1f}%）", (x, y), textcoords="offset points",
                    xytext=off, fontsize=9.5, color="#1e3a8a")
    ax.set_xlabel("语义 ID 碰撞率（%）", fontsize=11)
    ax.set_ylabel("稠密 recall@50 / gen-raw（%）", fontsize=11)
    ax.set_title("碰撞率 → 质量：四点剂量-响应（KuaiRec 全观测评估）",
                 fontsize=12.5)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xlim(-3, 46)
    ax.set_ylim(30, 110)
    fig.tight_layout()
    dst = out_dir / "collision_curve.png"
    fig.savefig(dst)
    print(f"-> {dst}")
    print("points:", [(p[0], round(x, 1), round(y, 1)) for p, x, y in
                      zip(points, xs, ys)])


if __name__ == "__main__":
    main()
