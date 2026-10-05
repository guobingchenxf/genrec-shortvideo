"""生成式序列模型：语义 ID（SID）/ 原生 ID 的词表化、训练与受限解码。

设计：
- Token 布局（SID 变体）: PAD=0, BOS=1；
  第 level 级码字 code 的 token = 2 + level*256 + code；
  行为 token = 2 + levels*256 + action。每个行为 = L 个 SID token + 1 个行为 token。
- Token 布局（raw 变体）: PAD=0, BOS=1；视频 token = 2 + video_idx；
  行为 token = 2 + n_videos + action。每个行为 = 1 个视频 token + 1 个行为 token。
- 训练：因果语言模型（预测下一个 token，PAD 忽略）；序列为
  [BOS] + 历史行为 token 序列 + 目标物品 token；pad 放序列尾部（不污染因果注意力）。
- 推理：SID 变体按 trie 做逐级受限解码 + beam search（只生成合法 SID）；
  raw 变体单步 beam。碰撞 SID 在解码时按热度展开候选组（口径统一、如实报告）。
"""

import numpy as np
import torch
import torch.nn as nn

PAD, BOS = 0, 1


def last_valid_items(videos, mask, actions, max_items):
    """取 context 中最后 max_items 个有效行为的 (视频, 行为 token)。"""
    idx = np.nonzero(mask)[0]
    if len(idx) > max_items:
        idx = idx[-max_items:]
    return videos[idx], actions[idx]


# ----------------------------------------------------------------------
# 词表 / 序列化
# ----------------------------------------------------------------------
class SidTokenizer:
    def __init__(self, video_ids, codes, n_actions=5, codebook_size=256,
                 popularity=None):
        self.video_ids = np.asarray(video_ids, dtype=np.int64)
        self.codes = np.asarray(codes).astype(np.int64)     # (N, levels)
        self.n_levels = int(self.codes.shape[1])
        self.codebook_size = codebook_size
        self.n_actions = n_actions
        self.vocab_size = 2 + self.n_levels * codebook_size + n_actions
        self.index_of = {int(v): i for i, v in enumerate(self.video_ids)}
        if popularity is None:
            self._order = np.arange(len(self.video_ids))
        else:
            self._order = np.argsort(-np.asarray(popularity, dtype=np.float64))
            rank = np.empty(len(self.video_ids), dtype=np.int64)
            rank[self._order] = np.arange(len(self.video_ids))
            self._order = rank
        self._trie = {}
        for i, key in enumerate(map(tuple, self.codes.tolist())):
            node = self._trie
            for c in key[:-1]:
                node = node.setdefault(c, {})
            node.setdefault(key[-1], []).append(i)

    def code_token(self, level, code):
        return 2 + level * self.codebook_size + code

    def action_token(self, action):
        return 2 + self.n_levels * self.codebook_size + int(action)

    def encode_item(self, video_id, action):
        i = self.index_of[int(video_id)]
        toks = [self.code_token(lv, int(c))
                for lv, c in enumerate(self.codes[i])]
        toks.append(self.action_token(action))
        return toks

    def encode_context(self, videos, mask, actions, max_items=20):
        vs, acs = last_valid_items(videos, mask, actions, max_items)
        toks = [BOS]
        for v, a in zip(vs, acs):
            toks.extend(self.encode_item(v, a))
        return toks

    def encode_target(self, target_video):
        i = self.index_of[int(target_video)]
        return [self.code_token(lv, int(c))
                for lv, c in enumerate(self.codes[i])]

    # ---- trie 查询 ----
    def allowed_codes(self, prefix):
        node = self._trie
        for c in prefix:
            node = node.get(c)
            if node is None:
                return []
        return list(node.keys()) if isinstance(node, dict) else []

    def videos_for_sid(self, prefix):
        node = self._trie
        for c in prefix:
            node = node.get(c)
            if node is None:
                return []
        if isinstance(node, list):
            return sorted(node, key=lambda i: self._order[i])
        return []


class RawTokenizer:
    def __init__(self, video_ids, n_actions=5):
        self.video_ids = np.asarray(video_ids, dtype=np.int64)
        self.n_videos = int(len(self.video_ids))
        self.n_actions = n_actions
        self.vocab_size = 2 + self.n_videos + n_actions
        self.index_of = {int(v): i for i, v in enumerate(self.video_ids)}

    def video_token(self, video_id):
        return 2 + self.index_of[int(video_id)]

    def decode_video_token(self, token):
        return int(self.video_ids[int(token) - 2])

    def action_token(self, action):
        return 2 + self.n_videos + int(action)

    def encode_item(self, video_id, action):
        return [self.video_token(video_id), self.action_token(action)]

    def encode_context(self, videos, mask, actions, max_items=20):
        vs, acs = last_valid_items(videos, mask, actions, max_items)
        toks = [BOS]
        for v, a in zip(vs, acs):
            toks.extend(self.encode_item(v, a))
        return toks

    def encode_target(self, target_video):
        return [self.video_token(target_video)]


# ----------------------------------------------------------------------
# 训练序列构建
# ----------------------------------------------------------------------
def build_training_sequences(tokenizer, ctx_videos, ctx_mask, ctx_actions,
                             target_videos, max_items, is_sid):
    """返回 (inputs, labels)，形状 (N, L)；pad 在序列尾部，label=PAD 处忽略。"""
    n = len(target_videos)
    block = (tokenizer.n_levels + 1) if is_sid else 2
    tgt_len = tokenizer.n_levels if is_sid else 1
    length = 1 + max_items * block + tgt_len
    inputs = np.zeros((n, length), dtype=np.int64)
    for i in range(n):
        toks = tokenizer.encode_context(ctx_videos[i], ctx_mask[i],
                                        ctx_actions[i], max_items)
        toks.extend(tokenizer.encode_target(int(target_videos[i])))
        inputs[i, :len(toks)] = toks
    labels = np.zeros_like(inputs)
    labels[:, :-1] = inputs[:, 1:]
    return inputs, labels


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class NextTokenLM(nn.Module):
    """因果 LM：embedding + 位置 + 骨干（transformer/gru）+ 输出投影。"""

    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4,
                 dropout=0.1, max_len=512, backbone="transformer"):
        super().__init__()
        self.backbone_name = backbone
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        if backbone == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, norm_first=True,
                activation="gelu")
            self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        elif backbone == "gru":
            self.backbone = nn.GRU(d_model, d_model, num_layers=n_layers,
                                   batch_first=True)
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(pos)[None, :, :]
        if self.backbone_name == "transformer":
            causal = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            pad_mask = (x == PAD)
            h = self.backbone(h, mask=causal, src_key_padding_mask=pad_mask)
        else:
            # GRU 为轻量基线：末尾 padding 不参与损失（label=0 忽略）
            h, _ = self.backbone(h)
        return self.head(h)


# ----------------------------------------------------------------------
# 受限解码 / beam search
# ----------------------------------------------------------------------
def _pad_rows(token_lists, device):
    length = max(len(t) for t in token_lists)
    x = torch.zeros(len(token_lists), length, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        x[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return x.to(device)


@torch.no_grad()
def beam_search_sid(model, tok, contexts, beam=50, batch_users=32, device="cpu"):
    """对每个 context（token 列表）做 L 步受限解码，返回排序后的视频 id 列表。"""
    model.eval()
    out_lists = []
    for s in range(0, len(contexts), batch_users):
        chunk = contexts[s:s + batch_users]
        states = [[{"prefix": (), "cum": 0.0, "seq": list(c)}] for c in chunk]
        for step in range(tok.n_levels):
            rows, owner = [], []
            for ui, beams in enumerate(states):
                for st in beams:
                    rows.append(st)
                    owner.append(ui)
            if not rows:
                break
            x = _pad_rows([st["seq"] for st in rows], device)
            logits = model(x)[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
            for ui in range(len(chunk)):
                cand = []
                for i, o in enumerate(owner):
                    if o != ui:
                        continue
                    st = rows[i]
                    allowed = tok.allowed_codes(st["prefix"])
                    if not allowed:
                        continue
                    tids = np.array([2 + step * tok.codebook_size + c
                                     for c in allowed])
                    scores = logp[i, tids] + st["cum"]
                    order = np.argsort(-scores)[:beam]
                    for k in order:
                        code = allowed[int(k)]
                        cand.append({
                            "prefix": st["prefix"] + (code,),
                            "cum": float(scores[k]),
                            "seq": st["seq"] + [2 + step * tok.codebook_size + code],
                        })
                cand.sort(key=lambda d: -d["cum"])
                states[ui] = cand[:beam]
        for ui in range(len(chunk)):
            ranked, seen = [], set()
            for st in states[ui]:
                if len(st["prefix"]) != tok.n_levels:
                    continue
                for vi in tok.videos_for_sid(st["prefix"]):
                    vid = int(tok.video_ids[vi])
                    if vid not in seen:
                        seen.add(vid)
                        ranked.append(vid)
            out_lists.append(ranked)
    return out_lists


@torch.no_grad()
def beam_next_raw(model, tok, contexts, beam=50, batch_users=64, device="cpu"):
    """raw 变体：单步 beam，直接生成视频 token。返回视频 id 列表。"""
    model.eval()
    out_lists = []
    for s in range(0, len(contexts), batch_users):
        chunk = contexts[s:s + batch_users]
        x = _pad_rows(chunk, device)
        logits = model(x)[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        logp[:, :2] = -1e9                        # PAD / BOS 不可推荐
        logp[:, 2 + tok.n_videos:] = -1e9         # 行为 token 不可推荐
        k = min(int(beam), tok.n_videos)
        top = np.argsort(-logp, axis=1)[:, :k]
        for row in top:
            out_lists.append([tok.decode_video_token(int(t)) for t in row])
    return out_lists
