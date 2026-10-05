# GenRec-ShortVideo：基于语义 ID 的短视频生成式推荐（KuaiRec）

一个**面向推荐算法实习面试、可完整复现**的生成式推荐研究原型：
把视频编成**层次语义 ID**（内容特征 → RQ-VAE），用轻量 seq2seq 在用户行为序列上
**生成**候选（trie 受限解码 + beam search），并在快手 **KuaiRec 全观测稠密矩阵**上
做**无偏评估与受控对照实验**。全流程纯 CPU，可在普通笔记本上跑通。

> 所有实验数字均来自本仓库实测存档（`results/experiments/*.json`），未运行项明确标注。
> 详细版见 [`docs/实验报告.md`](docs/实验报告.md)；面试讲稿与追问见 [`docs/面试材料.md`](docs/面试材料.md)。

---

## 1. 项目背景与问题定义

生成式推荐（TIGER 路线）：把物品编码为**语义 ID**、把推荐重构为**序列生成**。
它宣称解决 item 词表过大与冷启动问题，并在多个基准上超过判别式方法——
但"**在什么条件下成立、瓶颈在哪**"，值得用一个受控实验去回答。本项目实际回答三个问题：

| 问题 | 本项目的答案（见 §7） |
|---|---|
| Q1：同架构、同预算下，SID 生成 vs 原生 ID 生成谁更强？ | **当前设置下原生 ID 更强**（recall@50 0.0383 vs 0.0248）——负结果 |
| Q2：碰撞率是不是 SID 弱势的主因？ | **是主因之一但不唯一**：唯一变量（码本 256→1024）使 recall@50 **+64%** |
| Q3：什么场景下 SID 生成有独有优势？ | **长尾/冷启动**：唯一能命中零曝光冷启动视频的方法（其他方法≈0） |

## 2. 数据集（KuaiRec）

- **来源**：Zenodo record `18164998`（"KuaiRec Dataset with Raw Features"）；GitHub: `chongminggao/KuaiRec`。
- **许可**：Zenodo 标注 **CC-BY-4.0**；GitHub 徽章标注 CC BY-SA 4.0（口径不一致，双标注）。
- **引用**：Gao et al., *KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems*, CIKM 2022。
- **规模**：大矩阵 7,176 用户 × 10,728 视频 / 12,530,806 条（16.3%，训练用）；
  小矩阵 1,411 × 3,327 / 4,676,570 条（**99.6% 稠密，全观测**，无偏评估用）。
- **字段**：交互 = user_id / video_id / play_duration / video_duration / time / date / timestamp / **watch_ratio（=标签）**；
  内容 = caption（中文标题）/ 封面文字 / 话题标签 / 三级类目。
- **已知数据特性**（校验证实）：
  - small 矩阵 3.9% 行缺 timestamp → 预处理显式剔除并计数；
  - watch_ratio 有回放循环极值（>50 占 0.02%，最大 573）→ 行为分桶封顶；
  - caption 文件需 `engine="python"` 解析（C 引擎 Buffer overflow）；8 行字段缺失按空串处理；
  - 内容覆盖率 99.96%（10,728 个视频仅 4 个无标题且无类目）。
- **下载路线**（2026-10-05 本机实测）：官方整包 42 KB/s（不可行）→
  默认：big/small 矩阵走 **hf-mirror.com**（实测 455 KB/s，下载后按官方口径严格校验），
  caption/raw-categories 走 Zenodo 官方小文件；
  官方完整包手动方案：浏览器下载 `KuaiRec.zip` (432MB) → 解压后把 4 个 CSV 放入
  `data/raw/kuairec/` → 运行 `python -m genrec.cli validate`。

## 3. 方法与算法

### 3.1 语义 ID（RQ-VAE，`src/genrec/models/rqvae.py`）

内容特征标准化 → MLP 编码潜变量 z → 逐级残差量化：

```
r_1 = z;  第 l 级:  c_l = argmin_k ||r_l − e_k||²;  r_{l+1} = r_l − c_l
量化表示 q = Σ_l c_l;   重建 x̂ = Decoder(q)
损失 = ||x − x̂||² + Σ_l ( ||sg(r_l) − c_l||² + β ||r_l − sg(c_l)||² )
直通估计(STE):  q_st = z + (q − z).detach()
```

要点（均有实测依据）：**STE 缺失会导致码本坍缩**（本项目踩坑：唯一 SID 593 → 修复后 9,114）；
每 100 轮死码重启；3 级 × 1024 码本（v2）碰撞率 22.2%。

### 3.2 生成模型（`src/genrec/models/seqgen.py`）

- **Token 布局**：`PAD=0, BOS=1`；SID 码字 token = `2 + level*K + code`；
  行为 token = `2 + L*K + action`（watch_ratio 五档分桶）。每个行为 = L 个 SID token + 1 个行为 token。
- **训练**：因果语言模型，`[BOS] + 历史 token 序列 + 目标 SID`，下一个 token 交叉熵（PAD 忽略）。
- **推理**：对全部合法 SID 建 **trie**，beam search 每步只允许合法前缀；
  解码后同一 SID 的**碰撞组展开**并去重；raw 变体单步生成。

## 4. 环境安装

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/Scripts/python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install --no-deps -e .
```

## 5. 全流程命令

```bash
python -m genrec.cli download            # 下载 + 解压 + 尺寸校验（含 SHA256 清单）
python -m genrec.cli validate            # 官方口径校验（行数/用户/视频/覆盖率）
python -m genrec.cli prepare             # 预处理（150k 训练样本 + 全观测评估协议 + 内容特征）
python -m genrec.cli train-rqvae                 # 语义 ID v1（256 码本）
python -m genrec.cli train-rqvae --tag v2        # 语义 ID v2（1024 码本，改进版）
python -m genrec.cli train-gen --variant sid     # 生成模型（SID v1）
python -m genrec.cli train-gen --variant sid-v2  # 生成模型（SID v2）
python -m genrec.cli train-gen --variant raw     # 对照：原生 ID 生成
python -m genrec.cli train-gen --variant raw-gru # 基线：GRU4Rec-lite
python -m genrec.cli evaluate            # 六方法主表（results/experiments/main_table.json）
python -m genrec.cli evaluate --methods gen-sid,gen-sid-v2 --beam 10   # beam 消融
python -m genrec.cli generate --user-id 14 --topk 10   # 单用户生成 demo
python scripts/analyze_buckets.py        # 长尾/冷启动分桶分析
pytest -q                                # 24 项测试（仅用合成数据）
```

常用干跑（不产正式结论）：`prepare --smoke` / `train-rqvae --smoke` / `train-gen ... --smoke` /
`evaluate --smoke`（产物带 `_smoke` 后缀，物理隔离）。

## 6. 结果摘要（1,411 用户全观测无偏评估；实测）

| 方法 | recall@10 | recall@50 | ndcg@10 | coverage@50 | 延迟 ms/用户 |
|---|---|---|---|---|---|
| 流行度 | 0.00285 | 0.00820 | 0.1926 | 0.50% | 0.6 |
| ItemCF | **0.01484** | **0.05852** | **0.9469** | 0.83% | 1.5 |
| gen-sid（v1） | 0.00370 | 0.01515 | 0.2428 | **21.8%** | 203 |
| gen-sid-v2 | 0.00555 | 0.02483 | 0.3600 | 18.9% | 282 |
| gen-raw（对照） | 0.00791 | 0.03833 | 0.4966 | 14.8% | 3.5 |
| gru-raw（基线） | 0.00837 | 0.03713 | 0.5640 | 2.6% | 2.4 |

三条结论：**① 核心对照为负结果**（SID 生成暂弱于原生 ID）；
**② 碰撞是主因之一**（码本扩容唯一变量 → recall@50 +64%，但未全解释）；
**③ SID 生成的价值在长尾/冷启动**（唯一能命中零曝光冷启动物品；Top-50 长尾占比 5~8%，其他方法≈0）。
完整数据与归因见 [`docs/实验报告.md`](docs/实验报告.md)。

## 7. 目录结构与关键模块

```
configs/default.yaml      全部超参与路径（无硬编码）
data/raw/                 原始数据（不入库）     data/processed/ 预处理产物（不入库）
data/reports/             下载清单/校验报告      results/experiments/ 实验 JSON（入库）
src/genrec/
  cli.py                  命令行入口（9 个子命令）
  config.py               配置加载与路径解析
  data/download.py        多源下载（断点续传 + SHA256）
  data/validate.py        官方口径校验（errors/warnings 分离）
  data/preprocess.py      样本构建/评估协议/内容特征（防泄漏）
  models/rqvae.py         RQ-VAE（STE/死码重启/标准化）
  models/seqgen.py        Tokenizer/trie 受限解码/beam/因果 LM
  train.py                训练编排（四个变体）
  eval/metrics.py         指标（recall/ndcg/hit_rate/coverage）
  eval/baselines.py       流行度与 ItemCF（含缓存）
  eval/run_eval.py        统一评估与列表存档
  generate.py             单用户生成 demo
scripts/measure_speed.py  CPU 速度标定脚本      scripts/analyze_buckets.py 分桶分析
tests/                    24 项测试（合成数据，覆盖防泄漏/受限解码/RQ-VAE 关键性质）
docs/                     实验报告 / 面试材料 / 论文与出处
```

## 8. 硬件资源需求（本机实测）

| 环节 | 实测 |
|---|---|
| 预处理全流程 | 约 77s，峰值内存 1.39 GB |
| RQ-VAE 训练 | 416s（v1）/ 670s（v2），600 epochs |
| 生成模型训练（150k 样本 ×1 epoch） | GRU 843s；Transformer 853~1,588s |
| 评估（1,411 用户） | 基线 <5s；生成模型 286~398s（beam=50, CPU） |
| 训练吞吐 | **4 线程 206 samples/s；8 线程仅 19（线程超订）** |

无 GPU 依赖；内存峰值 < 1.5GB；磁盘约 3GB（数据 + 产物）。

## 9. 常见问题（FAQ）

1. **下载失败/太慢？** 见 §2 手动方案：把 4 个 CSV 放入 `data/raw/kuairec/` 后重跑 `validate`。
2. **为什么 8 线程更慢？** CPU 小模型线程超订（同步开销 > 计算收益），实测 4 线程快 10 倍。
3. **为什么指标与论文不可比？** 全观测稠密数据的任务形态不同（人均 3,000+ 交互），
   只做跨方法相对比较，绝对值不可与其他稀疏数据集数字对照。
4. **`_smoke` 产物能用于结论吗？** 不能。它们仅用于流水线验证，与正式产物物理隔离。
5. **怎么改评估/模型配置？** 全部在 `configs/default.yaml`，改动会被实验 JSON 记录。
6. **种子在哪？** 配置中统一 `seed: 42`（RQ-VAE 与生成模型各自记录）。
7. **如何只看快速冒烟？** 三个 `--smoke` 命令 + `pytest -q` 即可在几分钟内验证全链路。
8. **为什么没有 like 信号？** KuaiRec 无点赞字段，官方建议 `like ≈ watch_ratio > 2.0`（行为分桶已采用）。

## 10. 局限与未解决问题

- **离线研究原型**：无线上 A/B、无线上收益声明；标签为观看行为，非互动全信号；
- **轻量预算**：生成模型 1 epoch × 150k 样本；内容编码器为 TF-IDF+SVD（非神经编码器）；
- **碰撞率 22%（v2）**：仍是主要瓶颈之一；TIGER 式"碰撞追加位"未实现；
- **待运行**：行为 token 消融（on/off）、4 级 SID、多 epoch 收敛对比。

## 11. 引用与许可

数据集引用与论文清单（可核查）：见 [`docs/论文与出处.md`](docs/论文与出处.md)。
代码为本项目独立实现（未复制任何开源代码）；数据集使用遵循其许可（CC-BY-4.0 / CC BY-SA 4.0 以原文为准）。
