"""NC-Decoder-768: parameter-efficient non-commutative decoder.
d=768, depth=12, fits single T4 16GB. No PE, no QKV bloat.

Core idea (from v2-final):
- order via cumulative Lie-algebra phases p_t = cumsum(delta_t), delta from x
  (first-order BCH of unitary product, parallel, no expm, no wrap)
- attention bias = antisym(u_i - u_j) + content QK + ALiBi, causal
- standard SDPA for content path (fast on T4), SwiGLU MLP 2x (not 4x)
- tied embeddings + low-rank corrector (49k) instead of full head
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.w


class NCBias(nn.Module):
    """Non-commutative order bias. Parallel, O(B*T*K).

    delta_t = eps0 * tanh(W_theta x_t)  in R^K  (Lie-algebra increment)
    p_t = cumsum(delta)                  (log of prefix product, 1st order)
    u_t = p_t @ a                        (scalar direction score)
    bias[i,j] = (u_i - u_j) - m*(i-j), j<=i else -inf
    Antisymmetric by construction, no positional tables.
    """
    def __init__(self, d_model, k_phase=32, eps0=0.15):
        super().__init__()
        self.W_theta = nn.Linear(d_model, k_phase, bias=False)
        self.a = nn.Parameter(torch.randn(k_phase) * 0.1)
        self.eps0 = eps0
        self.k = k_phase
        # ALiBi slopes per head init later; single scalar here, per-head in block
    def forward(self, x, n_heads):
        # x: (B,T,D) -> delta (B,T,K)
        delta = self.eps0 * torch.tanh(self.W_theta(x))  # bounded, wrap-safe
        p = torch.cumsum(delta, dim=1)  # (B,T,K)
        u = torch.einsum('btk,k->bt', p, self.a).unsqueeze(-1)  # (B,T,1)
        # bias (B,1,T,T): u_i - u_j
        b = u.transpose(1, 2) - u  # b[:,i,j] = u_j - u_i? fix sign below
        # we want bias[i,j] = u_i - u_j for query i attending key j
        b = -b.transpose(1, 2) if False else (u - u.transpose(1, 2))
        # u shape (B,T,1): u - u^T gives (B,T,T) with [i,j]=u_i-u_j. correct.
        return b.unsqueeze(1)  # (B,1,T,T), broadcast over heads


class NCBlock(nn.Module):
    def __init__(self, d=768, n_heads=12, ffn_mult=2, k_phase=32, dropout=0.0):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.h = n_heads
        self.hd = d // n_heads
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.nc = NCBias(d, k_phase=k_phase)
        # ALiBi slopes: 2^(-8*i/h)
        slopes = torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])
        self.register_buffer('slopes', slopes.view(1, n_heads, 1, 1))
        # SwiGLU FFN, hidden = d*ffn_mult (1536 for d=768, vs 3072 in GPT2)
        hff = d * ffn_mult
        self.gate = nn.Linear(d, hff, bias=False)
        self.up = nn.Linear(d, hff, bias=False)
        self.down = nn.Linear(hff, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias_nc=None):
        B, T, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).view(B, T, 3, self.h, self.hd)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        # (B,H,T,Hd)
        if attn_bias_nc is None:
            attn_bias_nc = self.nc(h, self.h)  # (B,1,T,T)
        # ALiBi distance penalty
        pos = torch.arange(T, device=x.device)
        dist = (pos.view(1, 1, T, 1) - pos.view(1, 1, 1, T)).clamp(min=0).float()
        # causal mask
        causal = torch.full((T, T), float('-inf'), device=x.device)
        causal = torch.triu(causal, diagonal=1).view(1, 1, T, T)
        bias = attn_bias_nc + (-self.slopes * dist) + causal
        # SDPA with additive bias: use manual softmax for T4 compat (flash via math)
        # torch SDPA supports attn_mask as bool or additive? use manual for clarity + fp16 safe
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd) + bias
        a = torch.softmax(s.float(), dim=-1).to(x.dtype)
        a = self.drop(a)
        o = (a @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.o(o))
        # FFN
        h2 = self.n2(x)
        x = x + self.drop(self.down(F.silu(self.gate(h2)) * self.up(h2)))
        return x


class NCDecoderLM(nn.Module):
    def __init__(self, vocab=50257, d=768, layers=12, heads=12, ffn_mult=2,
                 k_phase=32, max_ctx=2048, lora_r=32, dropout=0.0):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(vocab, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            NCBlock(d, heads, ffn_mult, k_phase, dropout) for _ in range(layers)
        ])
        self.nf = RMSNorm(d)
        # low-rank corrector: h' = h + B(A h), 2*d*r params (~49k for 768/32)
        self.lora_A = nn.Linear(d, lora_r, bias=False)
        self.lora_B = nn.Linear(lora_r, d, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.vocab = vocab
        self.layers = layers
        self.apply(self._init)
        nn.init.zeros_(self.lora_B.weight)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        # idx: (B,T)
        x = self.tok(idx)
        x = self.drop(x)
        for blk in self.blocks:
            # gradient checkpointing handled outside; plain forward here
            x = blk(x)
        x = self.nf(x)
        xc = x + self.lora_B(self.lora_A(x))
        # tied head: logits = E @ h
        return xc @ self.tok.weight.T

    def count_params(self):
        tot = sum(p.numel() for p in self.parameters())
        emb = self.tok.weight.numel()
        return {'total': tot, 'emb': emb, 'backbone': tot - emb}


def build_nc768(vocab=50257):
    return NCDecoderLM(vocab=vocab, d=768, layers=12, heads=12,
                       ffn_mult=2, k_phase=32, lora_r=32)
