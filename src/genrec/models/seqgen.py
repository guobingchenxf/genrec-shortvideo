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
- D1：SID 解码默认使用"context 前缀 K/V 缓存"实现（消除每步对整段前缀的重复计算）；
  beam_search_sid_naive 为等价性测试用的参考实现（语义与缓存版一致）。
- 行为 token 可关闭（use_actions=False，E3 消融）：每个行为只保留 SID token，
  目标物品 token 序列不变（raw 变体暂不支持该开关）。
"""

import math

import numpy as np
import torch
from torch import nn

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
    def __init__(self, video_ids, codes, n_actions=5, codebook_size=None,
                 use_actions=True):
        self.video_ids = np.asarray(video_ids, dtype=np.int64)
        self.codes = np.asarray(codes).astype(np.int64)     # (N, levels)
        self.n_levels = int(self.codes.shape[1])
        self.use_actions = bool(use_actions)
        # 码本大小从码字自动推断（2 的幂），保证训练与推理两端词表一致；
        # 显式传入仅用于测试等特殊场景
        if codebook_size is None:
            codebook_size = 2 ** int(np.ceil(np.log2(int(self.codes.max()) + 1)))
        self.codebook_size = int(codebook_size)
        self.n_actions = n_actions
        self.vocab_size = 2 + self.n_levels * self.codebook_size + n_actions
        self.index_of = {int(v): i for i, v in enumerate(self.video_ids)}
        self._order = np.arange(len(self.video_ids))
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
        if self.use_actions:
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
        self.n_videos = len(self.video_ids)
        self.use_actions = True
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
    block = (tokenizer.n_levels if is_sid else 1) + (1 if tokenizer.use_actions else 0)
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

    def forward(self, x, select_positions=None):
        """select_positions: 可选，(B, K) 位置索引；只对这些位置计算输出头。

        用于"序列似然打分"等场景：避免对整条序列 × 词表算 logits（省时省内存）。
        """
        _, T = x.shape
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
        if select_positions is not None:
            bidx = torch.arange(h.shape[0], device=h.device)[:, None]
            h = h[bidx, select_positions]          # (B, K, d)
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


def _cached_sa_block(attn, x, past_k, past_v, key_bias):
    """复刻 nn.MultiheadAttention 前向（batch_first），支持拼接历史 K/V。

    x: (B, T, d)；past_k/past_v: None 或 (B, H, T_past, d_h)；
    key_bias: None 或 (B, 1, T, T_past + T) 加性掩码（-inf 屏蔽）。返回 (out, k, v)。
    """
    b, t, e = x.shape
    h = attn.num_heads
    d_h = e // h
    q, k, v = nn.functional.linear(
        x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q.reshape(b, t, h, d_h).transpose(1, 2)
    k = k.reshape(b, t, h, d_h).transpose(1, 2)
    v = v.reshape(b, t, h, d_h).transpose(1, 2)
    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
    if key_bias is not None:
        scores = scores + key_bias
    weight = torch.softmax(scores, dim=-1)
    out = torch.matmul(weight, v).transpose(1, 2).reshape(b, t, e)
    out = nn.functional.linear(out, attn.out_proj.weight, attn.out_proj.bias)
    return out, k, v


def _cached_layer_forward(layer, x, past=None, key_bias=None):
    """复刻 nn.TransformerEncoderLayer（norm_first=True）前向，支持历史 K/V。"""
    sa_out, k_all, v_all = _cached_sa_block(
        layer.self_attn, layer.norm1(x),
        past[0] if past is not None else None,
        past[1] if past is not None else None,
        key_bias)
    x = x + layer.dropout1(sa_out)
    ff = layer.linear2(
        layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
    x = x + layer.dropout2(ff)
    return x, (k_all, v_all)


def _prefix_bias(pad):
    """前缀自注意力加性掩码 (B, 1, T, T)：因果（j <= i）且屏蔽 pad 键。"""
    t = pad.shape[1]
    idx = torch.arange(t, device=pad.device)
    allowed = (idx[None, :, None] >= idx[None, None, :]) & ~pad[:, None, :]
    bias = torch.zeros(pad.shape[0], t, t, device=pad.device)
    bias.masked_fill_(~allowed, float("-inf"))
    return bias.unsqueeze(1)


def _tail_logprobs(model, rows, owner, ctx_t, t_pre, pad, past, device,
                   row_chunk=256):
    """尾部（已生成码字 + 末位槽）log 概率：注意力横跨 [缓存前缀 + 尾部]。

    与原实现保持一致：短于批次最大长度的行取"padding 末位槽"的隐状态
    （该槽本身是 padding，但因果可见整行真实 token 的"幽灵查询"）。
    """
    r = len(rows)
    tails = [st["seq"][int(ctx_t[ui]):] for st, ui in zip(rows, owner)]
    t_tail = len(tails[0])
    owner_t = torch.tensor(owner, device=device)
    base = ctx_t[owner_t][:, None]
    # 尾部 tokens + 末位槽（tokens=0、位置=批次末位）
    tail_ids = torch.cat([
        torch.tensor(tails, device=device),
        torch.zeros(r, 1, dtype=torch.long, device=device)], dim=1)
    pos_tail = torch.cat([
        base + torch.arange(t_tail, device=device)[None, :],
        torch.full((r, 1), t_pre + t_tail - 1, dtype=torch.long,
                   device=device)], dim=1)                # (r, t_tail + 1)
    k_abs = torch.cat([
        torch.arange(t_pre, device=device)[None, :].expand(r, -1),
        base + torch.arange(t_tail, device=device)[None, :],
        pos_tail[:, -1:]], dim=1)                        # (r, t_pre+t_tail+1)
    pad_full = torch.cat(
        [pad[owner_t],
         torch.zeros(r, t_tail, dtype=torch.bool, device=device),
         torch.ones(r, 1, dtype=torch.bool, device=device)], dim=1)
    allowed = (k_abs[:, None, :] <= pos_tail[:, :, None]) & ~pad_full[:, None, :]
    bias = torch.zeros(r, t_tail + 1, t_pre + t_tail + 1, device=device)
    bias.masked_fill_(~allowed, float("-inf"))
    bias = bias.unsqueeze(1)                                 # (r,1,T,tk)
    emb = model.emb(tail_ids) + model.pos(pos_tail)
    # 行短于批次最大长度 → 取末位槽（末列）；否则取最后一个真实尾部 token
    sel = torch.where(ctx_t[owner_t] >= t_pre,
                      torch.full_like(owner_t, t_tail - 1),
                      torch.full_like(owner_t, t_tail))
    logits = torch.empty(r, model.head.out_features, device=device)
    for r0 in range(0, r, row_chunk):
        r1 = min(r0 + row_chunk, r)
        hh = emb[r0:r1]
        ob = owner_t[r0:r1]
        for li, layer in enumerate(model.backbone.layers):
            pk, pv = past[li]
            hh, _ = _cached_layer_forward(layer, hh, (pk[ob], pv[ob]),
                                          bias[r0:r1])
        idx = torch.arange(r1 - r0, device=device)
        logits[r0:r1] = model.head(hh[idx, sel[r0:r1]]).float()
    return torch.log_softmax(logits, dim=-1).cpu().numpy()


@torch.no_grad()
def beam_search_sid(model, tok, contexts, beam=50, batch_users=32, device="cpu"):
    """（D1）带 context 前缀缓存的受限解码，输出与 beam_search_sid_naive 一致。

    思路：BOS+上下文的 K/V 每用户只算一次；前向各步仅处理"已生成码字"
    （≤ 层数 - 1 个 token），注意力横跨 [缓存前缀 + 尾部]。
    仅支持 transformer 骨干，其他骨干自动回退参考实现。
    """
    if getattr(model, "backbone_name", "transformer") != "transformer":
        return beam_search_sid_naive(model, tok, contexts, beam, batch_users,
                                     device)
    model.eval()
    out_lists = []
    for s in range(0, len(contexts), batch_users):
        chunk = contexts[s:s + batch_users]
        b = len(chunk)
        x = _pad_rows(chunk, device)                     # (b, Tpre)，尾部 padding
        t_pre = x.shape[1]
        pad = (x == PAD)
        pos0 = torch.arange(t_pre, device=device)[None, :].expand(b, -1)
        h = model.emb(x) + model.pos(pos0)
        bias0 = _prefix_bias(pad)
        past = []
        for layer in model.backbone.layers:
            h, kv = _cached_layer_forward(layer, h, None, bias0)
            past.append(kv)
        ctx_len = [len(c) for c in chunk]
        ctx_t = torch.tensor(ctx_len, device=device)
        # 与原实现完全一致：step0 取 padding 批次的"末位槽"隐状态
        # （短于批次最大长度的行，该槽为 padding 的"幽灵查询"；保持既有结果不变）
        logits0 = model.head(h[:, -1, :]).float()

        states = [[{"prefix": (), "cum": 0.0, "seq": list(c)}]
                  for c in chunk]
        for step in range(tok.n_levels):
            rows, owner = [], []
            for ui, beams in enumerate(states):
                for st in beams:
                    rows.append(st)
                    owner.append(ui)
            if not rows:
                break
            if step == 0:
                logp = torch.log_softmax(logits0, dim=-1).cpu().numpy()
            else:
                logp = _tail_logprobs(model, rows, owner, ctx_t, t_pre,
                                      pad, past, device)
            for ui in range(b):
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
        for ui in range(b):
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
def beam_search_sid_naive(model, tok, contexts, beam=50, batch_users=32,
                          device="cpu"):
    """对每个 context（token 列表）做 L 步受限解码（D1 参考实现，勿改语义）。"""
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
