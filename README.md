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
但"**在什么条件下成立、瓶颈在哪**"，值得用一个受控实验去回答。本项目实际回答四个问题：

| 问题 | 本项目的答案（见 §7） |
|---|---|
| Q1：同架构、同预算下，SID 生成 vs 原生 ID 生成谁更强？ | **初版负结果 → 诊断修复后追平 → 收敛预算下反超**：修复碰撞与特征后 dense recall@50 达 raw 的 98.5%、采样 HR@10 达 97.3%；同预算 3 epoch 下四项指标全面反超（§6，单种子） |
| Q2：碰撞率是不是 SID 弱势的主因？ | **是主导因素**：碰撞率 39.2%→22.2%→9.3%→0% 对应 recall@50/raw 比值 40%→65%→76%→**98.5%**（四点剂量-响应） |
| Q3：什么场景下 SID 生成有独有优势？ | **修正后的答案**：早期的"长尾优势"是碰撞展开的配置副产物（消除碰撞后消失）；长尾覆盖需要显式的多样性/探索机制，不能指望 SID 自动带来 |
| Q4（C2 补充）：工业标准基线 SASRec 的表现？ | **阴性结果 + 完整诊断链**：SASRec-lite 两协议均弱于 GRU4Rec-lite 与生成式模型；根因锁定"训练期回看回声结构 × 短窗口点积打分"（回看率 28.1%→7.2%→3.4% 随时间衰减），非实现错误；结论仅限 lite 档（实验报告 §6.9） |

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
python scripts/build_extended_sid.py             # b3：碰撞额外码位（4 级全唯一）
python -m genrec.cli train-gen --variant sid     # 生成模型（SID v1）
python -m genrec.cli train-gen --variant sid-v2  # 生成模型（SID v2）
python -m genrec.cli train-gen --variant sid-b1  # 生成模型（神经特征 bge）
python -m genrec.cli train-gen --variant sid-b3  # 生成模型（碰撞额外码位）
python -m genrec.cli train-gen --variant sid-b3-na  # E3 消融：关闭行为 token
python -m genrec.cli train-gen --variant raw     # 对照：原生 ID 生成
python -m genrec.cli train-gen --variant raw-gru # 基线：GRU4Rec-lite
python -m genrec.cli train-sasrec           # C2：SASRec-lite 基线（全词表 CE）
python -m genrec.cli evaluate            # 主表：默认六方法（增量合并进 main_table.json）
python -m genrec.cli evaluate --methods gen-sid-b1,gen-sid-b3   # 补 B 组两行
python -m genrec.cli evaluate --methods gen-sid,gen-sid-v2 --beam 10   # beam 消融
python -m genrec.cli evaluate-sampled    # 标准协议（--methods 追加 gen-sid-b1,gen-sid-b3）
python -m genrec.cli generate --user-id 14 --topk 10   # 单用户生成 demo
python scripts/analyze_buckets.py        # 长尾/冷启动分桶分析
python scripts/analyze_coldstart.py      # E5 冷启动细化（首次曝光时间口径）
python scripts/analyze_diversity.py      # E6 列表多样性/新颖性
python scripts/run_convergence_campaign.py   # 收敛/种子战役：3 epoch + 种子方差（约 6.8h）
python scripts/run_e3_d2_campaign.py     # E3/D2 战役：行为 token 消融 + beam Pareto（约 77min）
python scripts/run_c2_sasrec.py           # C2 战役：SASRec-lite 训练 + 双协议评估（约 34min）
pytest -q                                # 45 项测试（仅用合成数据）
```

常用干跑（不产正式结论）：`prepare --smoke` / `train-rqvae --smoke` / `train-gen ... --smoke` /
`evaluate --smoke`（产物带 `_smoke` 后缀，物理隔离）。

## 6. 结果摘要（实测；完整数据见 [`docs/实验报告.md`](docs/实验报告.md)）

**稠密协议**（全观测小矩阵，1,411 用户）：

| 方法 | recall@10 | recall@50 | ndcg@10 | coverage@50 | 延迟 ms/用户 |
|---|---|---|---|---|---|
| 流行度 | 0.00285 | 0.00820 | 0.1926 | 0.50% | 0.6 |
| ItemCF | **0.01484** | **0.05852** | **0.9469** | 0.83% | 1.5 |
| gen-sid（v1，256 码本） | 0.00370 | 0.01515 | 0.2428 | 21.8% | 203 |
| gen-sid-v2（1024 码本） | 0.00555 | 0.02483 | 0.3600 | 18.9% | 282 |
| gen-sid-b1（bge 特征） | 0.00626 | 0.02898 | 0.3948 | 18.5% | 279 |
| **gen-sid-b3（碰撞额外码位）** | 0.00835 | **0.03775** | **0.5178** | 15.9% | 557 |
| gen-raw（对照） | 0.00791 | 0.03833 | 0.4966 | 14.8% | 3.5 |
| gru-raw（基线） | 0.00837 | 0.03713 | 0.5640 | 2.6% | 2.4 |
| sasrec（SASRec-lite） | 0.00619 | 0.02862 | 0.3948 | 19.5% | 0.77 |

**标准协议复验**（留一法 + 100 负采样，2,000 用户；序列似然打分，HR@10）：

| 方法 | random | pop | itemcf | gen-sid | gen-sid-v2 | gen-sid-b1 | gen-sid-b3 | gen-raw | gru-raw | sasrec |
|---|---|---|---|---|---|---|---|---|---|---|
| HR@10 | 0.092 | 0.159 | 0.517 | 0.292 | 0.411 | 0.426 | **0.470** | 0.483 | 0.501 | 0.334 |

**训练预算与种子方差（2026-10-07 实测；详见实验报告 §6.6）**：

- **3 epoch 同预算（seed 42）**：sid-b3 四项全面领先 raw-e3——dense recall@50 **0.03915 vs 0.03393（+15.4%）**、
  dense ndcg@10 **0.5714 vs 0.4580（+24.7%）**、采样 HR@10 0.5065 vs 0.4990（+1.5%）、采样 NDCG@10 0.2957 vs 0.2910（+1.6%）；
  且 sid-b3 随预算持续增益（1→3 epoch：+3.7% / +10.3% / +7.9% / +10.5%），raw 稠密两项回落（-11.5% / -7.8%）。
- **1 epoch 种子方差（seed 42/43/44，b3−raw 配对差）**：dense recall@50 **-0.046±0.083pp（在噪声内）**；
  dense ndcg@10 **+1.77±0.45pp（稳定为正）**；采样 HR@10 **-1.92±0.56pp（稳定小劣势）**——
  "未追平"只剩采样协议上一笔小额可测劣势，扩大预算后被反超。

**行为 token 消融与 beam Pareto（2026-10-07 实测；详见实验报告 §6.7/§6.8）**：

- **行为 token 消融（E3，新变体 sid-b3-na）**：关闭 watch_ratio 行为 token 后，五项指标全部方向为负但很小——
  稠密三项完全落在 b3 三种子分布内、采样两项略低于下沿 0.15~0.27pp → **未检测到收益**（序列 -19%、评估快 17%）。
- **beam 成本-质量 Pareto（D2）**：top-10 质量在 **beam≈20 饱和**（ndcg 0.5179 ≈ beam50 的 0.5178，延迟 256ms 仅 46%）；
  beam=10 保留 90% ndcg、延迟 24%；recall@50 受候选清单长度约束（beam50 0.0378 vs beam20 0.0151）；coverage 2.8%→15.9%。

**SASRec-lite 标准基线（C2，2026-10-07 实测；详见实验报告 §6.9）**：

- 同数据预算的因果自注意力基线（150k×6ep、全词表 CE、d64）：稠密 recall@50 **0.02862** / ndcg@10 0.3948、
  采样 HR@10 **0.3340** —— 两协议均弱于 GRU4Rec-lite（0.0371/0.501）与生成式模型（**阴性结果，如实记录**）。
- 诊断链：管线三重核验；**回看回声率 28.1%（训练窗）→ 7.2%（未来窗口）→ 3.4%（最终项）随数据集时间衰减**；
  LOO 上"末项回声"得分 ≥ 目标的比例 84.7%；容量 d64→d128 反使 LOO 0.334→0.208（回声被放大）。
- 结论仅限 lite 档配置；延迟 0.77ms/用户（全表最快）。

**评估维度扩展（E5/E6，2026-10-07 实测；详见实验报告 §6.10/§6.11）**：

- **冷启动细化（E5）**：按"首次曝光时间晚于训练窗"定义真·上新（词表 10.7%）；上新目标几乎不可命中
  （仅 v2 有 0.008 级微弱命中）；**只有 SID 家族会给上新物品曝光**（v1 5.39% → b3 0.16%，非 SID 方法全为 0）。
- **多样性与新颖性（E6）**：v1 新颖性最高（13.92）但消除碰撞后 b3 回落到 raw 同级（11.99 vs 11.95）；
  v1 的 ILD 极低（0.36）同样是碰撞展开后果——"SID 不自动带来多样性"证据链第三次闭合。

**五条结论**：**① 从负结果到追平**——初版 SID 生成弱于原生 ID；经诊断（碰撞处理 + 内容特征）
修复后，**b3 在两个协议上基本追平原生 ID**（dense recall@50 达 raw 的 98.5%、ndcg@10 反超 4%；
采样协议 HR@10 达 raw 的 97.3%）；
**② 碰撞率是主导因素**——碰撞率 39.2%→22.2%→9.3%→0% 对应 recall@50/raw 的
40%→65%→76%→**98.5%**（四点剂量-响应，跨协议一致）；
**③ 长尾发现的修正**——早期版本的长尾覆盖部分来自"碰撞展开任意取成员"的副产物，
消除碰撞后消失（8.0%→0.26%）；语义 ID 生成本身不自动带来长尾覆盖（详见实验报告 §6.5）；
**④ 方法论发现**——协议会改变方法排序（稠密协议下 ItemCF 断层第一，标准协议下与序列模型同档）；
**⑤ 训练预算与种子噪声**——1 epoch 下 dense recall@50 差距已在种子噪声内、采样协议剩 ~2pp 稳定小劣势；
同预算 3 epoch 下 sid-b3 全面反超 raw（dense 证据强 / 采样证据弱，单种子）。

## 7. 目录结构与关键模块

```
configs/default.yaml      全部超参与路径（无硬编码）
data/raw/                 原始数据（不入库）     data/processed/ 预处理产物（不入库）
data/reports/             下载清单/校验报告      results/experiments/ 实验 JSON（入库）
src/genrec/
  cli.py                  命令行入口（10 个子命令）
  config.py               配置加载与路径解析
  data/download.py        多源下载（断点续传 + SHA256）
  data/validate.py        官方口径校验（errors/warnings 分离）
  data/preprocess.py      样本构建/评估协议/内容特征（防泄漏）
  models/rqvae.py         RQ-VAE（STE/死码重启/标准化）
  models/seqgen.py        Tokenizer/trie 受限解码/beam/因果 LM
  models/sasrec.py        C2：SASRec-lite 基线（因果自注意力，全词表 CE）
  train.py                训练编排（四个变体）
  eval/metrics.py         指标（recall/ndcg/hit_rate/coverage/ILD/新颖性）
  eval/baselines.py       流行度与 ItemCF（含缓存）
  eval/run_eval.py        统一评估与列表存档
  generate.py             单用户生成 demo
scripts/measure_speed.py  CPU 速度标定脚本      scripts/analyze_buckets.py 分桶分析
scripts/build_extended_sid.py  b3 扩展 SID      scripts/run_convergence_campaign.py  收敛/种子战役
scripts/run_e3_d2_campaign.py  E3/D2 战役（行为 token 消融 + beam Pareto）
scripts/run_c2_sasrec.py  C2 战役（SASRec-lite 训练 + 双协议评估）
scripts/analyze_coldstart.py  E5 冷启动细化   scripts/analyze_diversity.py  E6 多样性/新颖性
tests/                    45 项测试（合成数据，覆盖防泄漏/受限解码/RQ-VAE/消融/多样性指标）
docs/                     实验报告 / 面试材料 / 论文与出处
```

## 8. 硬件资源需求（本机实测）

| 环节 | 实测 |
|---|---|
| 预处理全流程 | 约 77s，峰值内存 1.39 GB |
| RQ-VAE 训练 | 416s（v1）/ 670s（v2），600 epochs |
| 生成模型训练（150k 样本 ×1 epoch） | GRU 843s；Transformer 853~1,588s |
| 收敛/种子战役（6 次训练 + 12 次双协议评估） | 24,592s ≈ 6.8h |
| E3/D2 战役（5 点 beam 扫描 + 1 次训练 + 双协议评估） | 4,646s ≈ 77min |
| SASRec-lite 训练（150k × 6 epoch，d64） | 2,036s ≈ 34min（471 samples/s） |
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
- **轻量预算**：主表为 1 epoch × 150k 样本；已补 3 epoch 收敛对照与三种子方差（§6.6），未做超参搜索；
- **结论边界**：3 epoch 反超为单种子结论；采样协议上的领先证据弱于 dense 协议；b3 解码 557ms/用户（b1 279ms、raw 3.5ms），上线需 prefix 缓存等优化（待运行）；
- **消融与曲线**：行为 token 未检测到收益（E3 负结果，§6.7）；top-10 质量 beam≈20 饱和、直出场景延迟可减半（D2，§6.8）；
- **标准基线（阴性）**：SASRec-lite 两协议均低于 GRU4Rec-lite 与生成式模型；已定位为数据分布效应（回看回声率随时间衰减），非实现错误（§6.9）；
- **待运行**：完整版 SASRec（超参搜索——本版为 lite 档）、prefix 缓存推理优化。

## 11. 引用与许可

数据集引用与论文清单（可核查）：见 [`docs/论文与出处.md`](docs/论文与出处.md)。
代码为本项目独立实现（未复制任何开源代码）；数据集使用遵循其许可（CC-BY-4.0 / CC BY-SA 4.0 以原文为准）。
