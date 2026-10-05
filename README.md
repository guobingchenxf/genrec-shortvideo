# GenRec-ShortVideo：基于语义 ID 的短视频生成式推荐（KuaiRec）

面向推荐算法实习面试的生成式推荐项目：把物品（视频）编成**层次语义 ID**，
用轻量 seq2seq 模型在用户行为序列上**生成**下一批候选，而不是给候选打分排序。

> 状态：阶段 2（数据流水线）。模型/训练/评估代码在后续阶段交付。
> 所有实验结论以 `results/` 与 `docs/` 中的实测记录为准；未运行的实验一律标注"待运行"。

## 数据

- **KuaiRec**（快手短视频推荐数据集）：1,411 用户 × 3,327 视频的**全观测稠密矩阵**
  （99.6% 密度，用于**无偏评估**）+ 7,176 × 10,728 的稀疏大矩阵（训练用）。
- 官方来源：Zenodo record `18164998`（KuaiRec Dataset with Raw Features）；
  GitHub: `chongminggao/KuaiRec`。
- 引用：Gao et al. *KuaiRec: A Fully-observed Dataset and Insights for Evaluating
  Recommender Systems*, CIKM 2022。
- 许可：Zenodo 记录标注 **CC-BY-4.0**；GitHub 仓库徽章标注 CC BY-SA 4.0
  （两处口径不一致，此处双标注）。
- 下载路线（2026-10-05 本机实测制定）：
  - 官方 Zenodo 整包直链（432MB）本机实测约 **42 KB/s**（需数小时），不作为自动默认；
  - 默认自动路线：big/small 矩阵走 **hf-mirror.com 镜像**（同一数据集的非官方镜像，
    实测约 455 KB/s；下载后按官方统计口径严格校验后才用于实验）；
    caption / raw categories 走官方 Zenodo 小文件直链。

  ```bash
  python -m genrec.cli download
  ```

- 官方完整包（手动/备用）：浏览器打开 Zenodo 记录页下载 `KuaiRec.zip`（432 MB），
  解压后将 `big_matrix.csv`、`small_matrix.csv`、`kuairec_caption_category.csv`、
  `video_raw_categories_multi.csv` 放入 `data/raw/kuairec/`，
  再运行 `python -m genrec.cli validate`（会按官方口径校验）。
- 已知数据特性（2026-10-05 校验证实，见 `data/reports/validation_report.json`）：
  - small 矩阵约 **3.9%** 的行缺 `timestamp` → 预处理中显式剔除并把剔除量写入 manifest（不做静默兜底）；
  - `watch_ratio` 存在极值（回放循环导致：>50 约 0.02%，最大约 573）→ 行为分桶天然封顶；
  - caption 文件存在特殊引号行，pandas C 解析器会报 Buffer overflow → 以
    `engine="python"` 读取（实测 0.1s，行数 10,732 与视频数吻合）；其中 8 行存在
    字段缺失（读取后按空串处理）；
  - 内容覆盖率：10,728 个目标视频中 9,372 个有标题、10,683 个有类目、
    仅 **4 个两者皆无**（语义 ID 内容特征覆盖率 99.96%）。

## 环境

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/Scripts/python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install --no-deps -e .
```

## 常用命令

```bash
python -m genrec.cli download          # 下载 + 解压 + 尺寸校验
python -m genrec.cli validate          # 原始数据规模/字段/覆盖率校验
python -m genrec.cli prepare           # 预处理（正式产物 -> data/processed/）
python -m genrec.cli prepare --smoke   # 小规模 dry-run（产物带 _smoke 后缀）
pytest -q                              # 单元测试（仅用合成小数据）
```

## 模型与训练

- **语义 ID（RQ-VAE）**：内容特征（标题+类目 → 字符 n-gram TF-IDF → SVD 64 维）
  经标准化后训练 3 级 RQ-VAE（每级码本 256；直通估计 STE + 每 100 轮死码重启）。
  实测：10,728 个视频 → 7,699 个唯一 SID，三级码本均全激活；
  碰撞组（多视频共享同一 SID）在解码时按序展开，碰撞率如实记录在 `sid_stats.json`。
- **生成式序列模型（core）**：因果 Transformer 解码器（d=128、2 层、4 头）。
  Token 序列 = [BOS] + Σ[L 个 SID token + 1 个行为 token] + 目标物品 SID；
  训练目标 = 下一个 token 交叉熵（PAD 忽略）；推理 = **trie 受限解码 + beam search**，
  只生成合法 SID 前缀，beam 内展开碰撞组并按序去重。
- **基线与对照**：gen-raw（同架构直接生成原生 video token）、
  gru-raw（GRU4Rec-lite 风格单层 GRU）、流行度 Top-50、ItemCF。
- 训练速度实测（本机 8 核 CPU）：**4 线程约 200 samples/s**（150k 样本 1 epoch ≈ 12 分钟）；
  8 线程因线程超订降到 19 samples/s——参数已固化在 `configs/default.yaml` 注释中。

## 完整复现（命令序列）

```bash
python -m genrec.cli download && python -m genrec.cli validate
python -m genrec.cli prepare
python -m genrec.cli train-rqvae
python -m genrec.cli train-gen --variant sid
python -m genrec.cli train-gen --variant raw
python -m genrec.cli train-gen --variant raw-gru
python -m genrec.cli evaluate --methods pop,itemcf,gen-sid,gen-raw,gru-raw
python -m genrec.cli generate --user-id 0 --topk 10
```

主表输出：`results/experiments/main_table.json`；单方法结果：`eval_<method>.json`。

## 目录

```
configs/          配置（路径 / 预处理 / 评估协议参数）
data/raw/         原始数据（不提交）
data/processed/   预处理产物（不提交）
data/reports/     下载与校验记录（提交）
src/genrec/       代码：data/（本阶段）与后续阶段的 models/train/eval
tests/            纯合成数据测试
results/          实验记录（后续阶段，提交）
docs/             实验报告与面试材料（后续阶段）
```

## 评估协议（本项目核心方法学）

- **训练**：大矩阵按全局时间切分（前 90%），逐用户构造 (context → target) 样本，
  context 仅含 target 之前的行为（防泄漏）；并从训练集剔除
  "评估用户 × 评估目标视频"对（防转导泄漏）。
- **评估**：稠密小矩阵按时间切 80% / 20%——前 80% 行为作 context，
  后 20% 视频集合作 targets，在全量视频空间计算 Recall@K / NDCG@K。
  稠密矩阵的"全观测"性质使该评估**无曝光偏差**（与稀疏数据只能采样负样本有本质区别）。
- **候选排除**：所有方法统一排除用户 context 中的已看视频（ItemCF 相似度对角线置零）；
  NDCG 的分母 DCG 上限为 min(K, |targets|)（本数据集 targets 较大，约 600+ 视频/用户）。
- 行为 token：watch_ratio 五档分桶（官方建议 like ≈ watch_ratio > 2.0）。
- **两个协议的统计特性要如实理解**：该数据集用户密度极高（人均 3000+ 交互），
  "预测下一时段观看集合"对内容相似型方法（ItemCF）天然友好——
  指标绝对值不宜与其他稀疏数据集的论文数字直接比较，跨方法相对比较才是有意义的信号。
