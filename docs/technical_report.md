# A Controlled Study of Semantic-ID Generative Retrieval under Compute Constraints

*An offline, fully reproducible study on KuaiRec: when does semantic-ID generative retrieval work, where does it break, and what does a standard discriminative baseline do on the same budget.*

> This is the English technical-report companion to the Chinese experiment log
> ([`docs/实验报告.md`](实验报告.md)). All numbers are measured on a single consumer CPU
> laptop and archived under `results/experiments/*.json`; negative results and failed
> hypotheses are reported as-is. "待运行 / not run" items are marked explicitly.

---

## Abstract

Generative retrieval represents items as *semantic IDs* (hierarchical codebook
indices) and casts recommendation as sequence generation. We build a complete
controlled study on the fully-observed KuaiRec dense matrix (1,411 users,
99.6% density) with: (i) an RQ-VAE semantic-ID pipeline, (ii) a lightweight causal
Transformer over tokenized behavior sequences with trie-constrained beam decoding,
(iii) a chain of controlled ablations (collision handling, content encoder,
behavior tokens, training budget, seeds), and (iv) two evaluation protocols
(full-observation dense and standard leave-one-out + 100 negatives). We report a
complete research loop rather than a single point result: an initial *negative*
result (semantic-ID generation underperforms raw-ID generation),
a stepwise diagnosis (collision rate is the dominant factor; content encoder and
collision-free code assignment close the gap; larger training budget overtakes),
and two self-corrections (a "long-tail advantage" claim that turned out to be an
artifact of collision expansion, and a behavior-token ablation that finds no
detectable benefit). We additionally implement and evaluate a standard SASRec
baseline under the same data budget, obtaining a *negative* baseline result with
a full diagnosis (an "echo trap" induced by repeat-heavy training windows).

---

## 1. Introduction

Generative retrieval (TIGER, Rajput et al., NeurIPS 2023) replaces the item
lookup table with *semantic identifiers*: items are encoded from content,
quantized into a hierarchy of codebook indices (RQ-VAE), and recommendations are
produced by autoregressive sequence generation over the SID vocabulary. The
paradigm promises content-based cold-start identifiers and compact item spaces.
Published results, however, are obtained at budgets far beyond a single consumer
laptop, and it remains unclear *which* parts of the recipe carry the benefit.

This report studies the recipe under a deliberately small, fully reproducible
budget on the KuaiRec fully-observed dataset. We ask four questions:

- **Q1.** Under an identical architecture, dataset and training budget, does
  SID-based generation beat generation over raw item IDs?
- **Q2.** If it lags, is codebook *collision* the dominant factor?
- **Q3.** Under which conditions (if any) does SID generation have a unique
  advantage?
- **Q4.** How strong is a standard discriminative sequential baseline (SASRec)
  under the same budget and data?

Contributions:

1. A complete, reproducible pipeline (data → content features → RQ-VAE →
   causal-LM training → trie-constrained beam decoding → dual-protocol
   evaluation) that runs end-to-end on CPU.
2. A controlled chain of ablations (collision handling, content encoder,
   behavior tokens, training budget, seeds) with a four-point dose-response
   relating collision rate to retrieval quality.
3. Two evaluation protocols (fully-observed dense; standard leave-one-out with
   100 negatives) used to cross-check every conclusion.
4. Honest negative results and two self-corrections: an early "long-tail
   advantage" claim that proved to be an artifact of collision expansion, and a
   behavior-token ablation with no detectable benefit.
5. An engineering study of decoding cost (beam Pareto; prefix-cached decoding
   with element-wise equivalence checks).

## 2. Dataset, Preprocessing and Protocols

**Data.** KuaiRec (Gao et al., CIKM 2022) provides two user-video matrices from
Kuaishou: a large, sparse matrix (7,176 users × 10,728 videos, 12,530,806
interactions, 16.3% dense) used for training, and a small, *fully observed*
matrix (1,411 users × 3,327 videos, 4,676,570 interactions, 99.6% dense) used
for unbiased evaluation — every user-video pair is observed, so ranking metrics
carry no exposure bias. Content includes captions, cover text, hashtags and a
three-level category taxonomy.

**Preprocessing.** A global time split at the 90th timestamp percentile (t0)
defines the training window. Training samples are per-user windows: up to the
100 most recent behaviors per user, each sample being a context of up to 50
behaviors (we use the last 20) and the immediate next behavior as target;
150,000 windows are sampled globally plus 50,000 validation windows. Two
leakage guards are applied: contexts never contain the target, and 898,914
(eval-user × eval-target-video) pairs are excluded from training. `watch_ratio`
is bucketed into five action tokens. Known data caveats: 3.9% of small-matrix
rows lack timestamps (explicitly dropped); `watch_ratio` contains replay-loop
outliers (>50 for ~0.02%, max 573).

**Protocols.** *(a) Dense full-observation:* per user, the first 80% of
small-matrix behaviors form the context (we use the last 20) and the distinct
videos in the remaining 20% form the target set (≈637 videos/user on average).
All methods rank the full catalog; items already in the user's context are
excluded. Metrics: recall@10/50, NDCG@10, hit-rate@10, coverage@50, plus
intra-list diversity and novelty. *(b) Sampled standard protocol:* on the big
matrix, the last interaction of each user is the target, the preceding ≤20
behaviors the context, and 100 items absent from the user's history are sampled
as negatives (seed 42; 2,000 users). Generator scores are teacher-forced
sequence log-likelihoods; all methods share the candidate set. 95.5% of targets
lie after t0, making this a near-future test.

## 3. Method

### 3.1 Content features and semantic IDs

Item text combines caption, cover text, hashtags and the three-level category
string. The default encoder is TF-IDF (character 2–3 grams, 50k features, min
df 2) reduced by SVD to 64 dimensions (explained variance 0.350). The neural
ablation replaces it with `bge-small-zh-v1.5` CLS embeddings (512-d, L2
normalized) reduced by PCA to 64 dimensions (EV 0.648).

A residual-quantization VAE (RQ-VAE) maps features to hierarchical IDs:
standardize → MLP encoder to 32-d latent → three residual quantization levels
(codebook 256 for v1, 1024 for v2/b1) → MLP decoder. The loss is reconstruction
MSE + codebook loss + 0.25 × commitment; gradients pass through quantization via
a straight-through estimator (STE); dead codes are restarted every 100 epochs;
600 epochs on CPU (7–11 min). *Implementation note:* the first version omitted
the STE — the encoder received no reconstruction gradient and the codebook
collapsed (only 593 unique SIDs for 10,728 videos); this is now a regression
test.

Collision handling is explicit: v1/v2 produce 7,699 / 9,114 unique SIDs
(39.2% / 22.2% of videos share a code), decoded by expanding the collision
group; **b3** appends one extra digit (rank within the group by video id) so
that all 10,728 SIDs are unique (4 levels).

### 3.2 Generative sequence model

Token layout: `PAD=0, BOS=1`; SID token `2 + level*K + code`; action token
`2 + L*K + bucket` (watch_ratio bucketed at 0.2/0.5/1.0/2.0). Each historical
behavior contributes L SID tokens plus one action token; the training sequence
is `[BOS] + Σ(SID tokens + action token) + target SID tokens`, optimized with
next-token cross-entropy (padding ignored). The model is a 2-layer pre-LN
Transformer encoder (d=128, 4 heads, GELU) with learned positional embeddings
and a linear output head over the SID vocabulary.

Decoding is trie-constrained beam search: all legal SIDs form a prefix tree and
each level only expands legal code tokens; the final beams map back to videos
(collision groups are expanded and de-duplicated for v1/v2; b3 needs no
expansion). For the sampled protocol, candidates are scored by teacher-forced
sequence log-likelihood (the output head is applied only at the candidate
positions, making scoring cheap).

Budget and engineering: 150k samples, 1 epoch, batch 256, Adam lr 1e-3, 4 CPU
threads. (Measured: 8 threads were ~10× slower than 4 due to thread
oversubscription on the small model.)

### 3.3 Variants and baselines

Generative variants share the architecture and budget: **gen-sid** (v1),
**gen-sid-v2** (1024 codebooks), **gen-sid-b1** (bge features), **gen-sid-b3**
(extra-digit SIDs), **gen-sid-b3-na** (b3 without action tokens) and **gen-raw**
(raw video-token generation, vocab 10,735). Non-generative baselines:
**gru-raw** (GRU4Rec-lite: 1-layer GRU, d=64, lr 2e-3, same token stream),
**sasrec-lite** (causal self-attention over item-ID sequences, d=64, 2 layers,
full-vocabulary cross-entropy on every next-item position, N(0, 0.02)
initialization, 150k × 6 epochs), **itemcf** (train-window co-occurrence
similarity, top-50 by summed similarity to the last 50 context items) and
**pop** (train-window frequency). All ranking lists exclude items in the user's
context.

## 4. Experiments

### 4.1 Main results

Dense full-observation protocol (1,411 users; delay measured on CPU):

| method | recall@10 | recall@50 | ndcg@10 | coverage@50 | ms/user |
|---|---|---|---|---|---|
| pop | 0.00285 | 0.00820 | 0.1926 | 0.50% | 0.6 |
| itemcf | **0.01484** | **0.05852** | **0.9469** | 0.83% | 1.5 |
| gen-sid (v1) | 0.00370 | 0.01515 | 0.2428 | 21.8% | 203 |
| gen-sid-v2 | 0.00555 | 0.02483 | 0.3600 | 18.9% | 282 |
| gen-sid-b1 | 0.00626 | 0.02898 | 0.3948 | 18.5% | 279 |
| gen-sid-b3 | 0.00835 | 0.03775 | 0.5178 | 15.9% | 557 |
| gen-raw | 0.00791 | 0.03833 | 0.4966 | 14.8% | 3.5 |
| gru-raw | 0.00837 | 0.03713 | 0.5640 | 2.6% | 2.4 |
| sasrec-lite | 0.00619 | 0.02862 | 0.3948 | 19.5% | 0.77 |

Sampled standard protocol (2,000 users; HR@10):

| random | pop | itemcf | gen-sid | v2 | b1 | b3 | gen-raw | gru-raw | sasrec |
|---|---|---|---|---|---|---|---|---|---|
| 0.092 | 0.159 | 0.517 | 0.292 | 0.411 | 0.426 | **0.470** | 0.483 | 0.501 | 0.334 |

Three observations. **(i) The initial result is negative and only partially
recovered.** The first SID model loses to raw-ID generation by a wide margin
(recall@50 0.0152 vs 0.0383); after fixing collision handling and content
features, b3 reaches 98.5% of gen-raw's dense recall@50 and 97.3% of its
sampled HR@10 (and beats it by 4% on dense NDCG@10) — parity, not victory.
**(ii) The protocol changes the ranking.** ItemCF dominates the dense protocol
(0.0585, 2.4× the best SID model) but sits in the same band as GRU/gen-raw on
the sampled protocol — evaluation protocol is part of any conclusion.
**(iii) Generation does not beat the strong simple baselines** under this
budget: the best generator (gen-raw) trails ItemCF on dense and trails ItemCF
and gru-raw on sampled.

### 4.2 What closes the gap: codebook collisions

Collision rate and quality form a clean four-point dose-response (dense
recall@50 relative to gen-raw): 39.2% → 39.5%, 22.2% → 64.8%, 9.3% → 75.6%,
0% → **98.5%**. Two controlled steps produce the endpoints: (i) *encoder
ablation* (TF-IDF → bge-small-zh features, everything else fixed) raises
recall@50 by 17% and simultaneously reduces collisions 22.2% → 9.3%; (ii)
*collision handling* (v2's expansion → b3's extra digit) raises recall@50 by
52%. The mechanism is visible in how collisions are decoded: with group
expansion the model cannot choose *which* member of a collision group to
recommend (members are emitted in video-id order), so the code-to-item mapping
is ambiguous both for training (multiple items share a prefix) and decoding.
The extra digit resolves both: the same checkpoint family improves without any
change to data or architecture, and a per-user overlap check shows the b3 and
raw models share only ~40% of their top-10 lists (they rank differently, not
identically).

### 4.3 Training budget and seed variance

The main table trains every generative model for a single epoch. Two follow-up
runs test whether the parity result survives budget and noise:

**Convergence (3 epochs, seed 42).** sid-b3 and raw were retrained for 3 epochs.
The SID model keeps improving (dense recall@50 +3.7%, NDCG@10 +10.3%, sampled
HR@10 +7.9% from epoch 1 to 3) while the raw model's dense metrics *decline*
(recall@50 −11.5%, NDCG@10 −7.8%) — different budget-return curves. At 3 epochs
sid-b3 leads raw on all four metrics: dense recall@50 0.03915 vs 0.03393
(+15.4%), dense NDCG@10 0.5714 vs 0.4580 (+24.7%), sampled HR@10 0.5065 vs
0.4990 (+1.5%), sampled NDCG@10 +1.6% (single seed). Notably, teacher-forced
validation loss rises for both models while ranking keeps improving — loss is
not a substitute for downstream evaluation.

**Seed variance (1 epoch, seeds 42/43/44, paired b3−raw deltas).** Dense
recall@50 differs by −0.046 ± 0.083 pp (crosses zero — within seed noise);
dense NDCG@10 by +1.77 ± 0.45 pp (stable positive); sampled HR@10 by
−1.92 ± 0.56 pp (stable, small disadvantage of the SID model at 1 epoch, which
the 3-epoch run reverses). At 1 epoch the dense-recall gap is statistically
indistinguishable from zero; the sampled-protocol gap is small but real, and
budget-sensitive.

### 4.4 Inference cost: prefix-cached decoding

Naive trie-beam decoding re-runs the full sequence (context + generated codes)
for every beam row at every level: with 20 context items and beam 50, the
81-token prefix is recomputed up to ~50×4 times per user. We implement a
minimal prefix cache: per user the context K/V are computed once; each decode
step then forwards only the newly generated code tokens (≤3) with attention
spanning [cached prefix + tail]. To keep bit-compatibility with the naive
decoder, we replicated its padding-slot semantics exactly (a quirk that only
affects variable-length batches).

Verification: (i) unit tests assert element-wise identical outputs against the
retained reference implementation (including variable-length batches and the
GRU fallback path); (ii) a full re-evaluation of 4 models + 5 beam settings
(12,700 user lists) reproduces 12,696 lists exactly — two models flip 1–2 users
near rank 50 through float noise, and all protocol metrics are unchanged
(byte-identical ranking archives for gen-sid-b3); (iii) a same-process
controlled A/B (128 users, beam 50): 568.0 ms → 129.9 ms per user (**4.37×**).
On a quieter machine window the full 1,411-user evaluation falls from the
archived 556.7 ms to 106.4 ms per user for b3 (5.2×; cross-session absolute
latencies are load-sensitive — the controlled A/B is the authoritative
speedup).

### 4.5 What does not help: behavior tokens

b3-na removes the watch_ratio action tokens (sequence length 105 → 85 tokens).
All five metrics move slightly negative (−2.5% / −1.9% / −2.8% on dense
recall@10 / recall@50 / NDCG@10; −3.1% / −6.4% on sampled HR@10 / NDCG@10), but
every dense delta lies inside the seed-noise band and the sampled deltas clear
the noise floor by only 0.15–0.27 pp: **no detectable benefit for behavior
tokens at this budget**. Whether the effect is absent or needs more training
remains open — but the ablation is reported as a negative result.

### 4.6 Beam Pareto, cold-start definition and diversity

**Beam cost–quality curve** (gen-sid-b3, dense protocol, prefix-cached, single
session): beam 1 → 7.4 ms/user (NDCG@10 0.061); beam 5 → 10.3 ms (0.274);
beam 10 → 14.0 ms (0.466); beam 20 → 30.0 ms (0.518); beam 50 → 106.4 ms
(0.518). Top-10 quality saturates near beam 20 (its NDCG@10 equals beam 50 to
three decimals) while recall@50 keeps growing — recall@50 is candidate-budget
bound (a list shorter than 50 caps it). Direct top-10 serving can therefore
halve the latency at beam 20; "generate-then-rerank" pipelines want beam 50.

**Cold-start, refined definition (E5).** Defining "new items" by first
appearance later than t0 (1,144 / 10,728 = 10.7% of the catalog): no method
retrieves future-watched new items above the noise floor (only v2 records
0.008 recall@50 over 1,226 new targets) — but only the SID family ever
*exposes* new items in its top-50 (v1 5.39% of slots, v2 3.62%, b1 1.64%,
b3 0.16%; every non-SID method exactly 0). Content-addressed decoding is the
only mechanism here that can surface an item with no interaction history, and
its exposure share shrinks as collision artifacts are removed.

**Diversity and novelty (E6).** Intra-list diversity (mean pairwise content
cosine distance within top-50) and novelty (mean −log2 p of train-window
popularity): v1 has the lowest ILD (0.363) and the highest novelty (13.92) —
collision-driven again; after b3 both sit at gen-raw level (ILD 0.813 vs 0.861;
novelty 11.99 vs 11.95). This is the *third independent replication* of the
same correction: the apparent long-tail and diversity advantages of early SID
configurations were artifacts of collision expansion, not properties of
semantic-ID generation.

### 4.7 A standard discriminative baseline fails, and why (SASRec-lite)

We implemented SASRec (causal self-attention, shared item-embedding head,
all-position next-item supervision, full-vocabulary cross-entropy, N(0, 0.02)
initialization; same 150k-sample windows, 6 epochs — a faithful "lite" port)
and verified its coding and evaluation paths by hand (manual reproduction of
scores) and by deterministic re-runs (per-epoch losses and both protocol
metrics reproduce exactly). The result is negative: dense recall@50 0.0286 /
sampled HR@10 0.334 — below both gru-raw (0.0371 / 0.501) and gen-raw
(0.0383 / 0.4825).

Diagnosis chain (all measured): the model reaches HR@10 0.99 on its own
training windows, so the gap is generalization, not fit. The training
distribution is repeat-heavy — 28.1% of train-window targets re-appear among
the last 20 context items — while late-window and final-item evaluation sets
are not (7.2% / 3.4%): repeat viewing decays over the dataset's timeline. The
trained model prefers the *last watched item* over the true target in 84.7% of
sampled cases (mean score 5.59 vs 0.68) — an "echo policy" that is optimal in
the training window and miscalibrated at evaluation time. Doubling capacity
(d64 → d128) lowers training loss (4.90 → 4.58) yet further hurts sampled
HR@10 (0.334 → 0.208); the released model is the d64 variant. This is a
*lite-configuration* result — it does not rank full-scale SASRec — but it
documents that under this data and budget a standard discriminative sequential
baseline collapses into an echo trap driven by the training window's repeat
structure.

## 5. Discussion

**Q1.** Under the same architecture and budget, SID generation starts behind,
is repaired to parity (98.5% / 97.3% of raw on the two protocols, +4% dense
NDCG@10), and overtakes at 3 epochs (single seed) — a "not yet, and here is
what it takes" answer.

**Q2.** Collisions are the dominant controllable factor: a four-point
dose-response, plus a two-step endpoint repair (encoder, extra digit)
reproducible without touching the architecture.

**Q3.** Unique SID behavior appears in *exposure*, not retrieval: only
content-addressed decoding ever surfaces new items, and the early long-tail /
novelty advantages were collision artifacts — diversity has to be engineered
into decoding, it is not automatic.

**Q4.** On this data and budget, a faithful SASRec-lite does not beat the GRU
baseline — and the diagnosis (an echo policy induced by repeat-dominated
training windows) is itself a useful lesson about evaluating at dataset
boundaries.

## 6. Limitations

Lite budgets throughout: 1-epoch main models, a single seed in the 3-epoch
comparison, no hyper-parameter search; offline-only (no online A/B); the
SASRec result is configuration-specific; absolute latencies are load-sensitive
across sessions (a same-process controlled A/B is provided). Longer context,
larger models and longer training are natural extensions, all marked
"not run" in the Chinese experiment log.

## 7. Reproducibility

- Pipeline commands: README §5. Every number in this report maps to an archived
  JSON under `results/experiments/` (training logs, evaluations, ablations,
  ranking archives `lists_*.npz`).
- Tests: 45 unit tests on synthetic data (`pytest -q`), ruff-clean codebase.
- Hardware: single consumer CPU (4 threads), ~1.4 GB peak memory for
  preprocessing, no GPU; full main pipeline (including both protocols) runs on
  a laptop overnight or in hours.
- The Chinese experiment log (`docs/实验报告.md`, sections §6.1–§6.12) contains
  per-experiment details, including all negative results and "not run" items.

## References

1. Gao, C., et al. KuaiRec: A Fully-observed Dataset and Insights for
   Evaluating Recommender Systems. CIKM 2022.
2. Rajput, S., et al. Recommender Systems with Generative Retrieval (TIGER).
   NeurIPS 2023.
3. Lee, D., et al. Autoregressive Image Generation using Residual Quantization
   (RQ-VAE). CVPR 2022.
4. Kang, W.-C., McAuley, J. Self-Attentive Sequential Recommendation (SASRec).
   ICDM 2018.
5. Hidasi, B., et al. Session-based Recommendations with Recurrent Neural
   Networks (GRU4Rec). ICLR 2016.
