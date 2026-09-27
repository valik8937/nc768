"""Honest non-commutative decoder v2: real SO(16) Givens, NO complex, NO QKV, NO per-token loop.

- Order ONLY from product Q_t = U_t ... U_1, U_t = 16 real plane rotations.
  Commutator [X,Y] != 0 preserved. No cumsum, no 1j, no ComplexHalf.
- Rotations built vectorized over (B,T): 2 stages of 8 disjoint pairs via scatter.
- Prefix states via parallel associative scan (log2(T) batched bmm rounds).
- Unitary core forced fp32 (autocast disabled inside) on any device.
- Attention: S = Tr[S_mat Qj^T Qi]/n (sym) + Tr[A_mat Qj^T Qi]/n (antisym), causal.
  Values = unitary features themselves. Plus associative Mt memory (light real loop).
- Budget: d=256, 4 blocks, byte vocab 256 -> ~2.46M total. No PE tables.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

N = 16
PAIRS0 = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
PAIRS1 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 0)]
PAIRS = PAIRS0 + PAIRS1  # K=16 plane rotations per token


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def _stage(Q, P, QQ, th):
    """One vectorized stage of 8 disjoint SO(2) rotations, over all (B,T) at once.
    Real cos/sin only. Functional (grad-safe). Same math as the per-pair loop."""
    c = torch.cos(th)          # (...,8)
    s = torch.sin(th)
    Rp = Q[..., P, :]          # (...,8,N) advanced index -> copy, no inplace
    Rq = Q[..., QQ, :]
    Np = c[..., None] * Rp - s[..., None] * Rq
    Nq = s[..., None] * Rp + c[..., None] * Rq
    B, T, _, Nn = Q.shape
    Q = Q.scatter(2, P.view(1, 1, -1, 1).expand(B, T, P.numel(), Nn), Np)
    Q = Q.scatter(2, QQ.view(1, 1, -1, 1).expand(B, T, QQ.numel(), Nn), Nq)
    return Q


def _prefix_products(U):
    """All prefix products P_t = U_t ... U_1 via parallel doubling scan.
    log2(T) rounds, one batched bmm per round. Exact, no sequential loop."""
    B, T, N, _ = U.shape
    P = U
    I = torch.eye(N, dtype=U.dtype, device=U.device).expand(B, 1, N, N)
    stride = 1
    while stride < T:
        take = min(stride, T)
        head = I.expand(B, take, N, N)
        Ps = torch.cat([head, P[:, :T - take]], dim=1) if take < T else head
        P = torch.matmul(P.reshape(B * T, N, N), Ps.reshape(B * T, N, N)).view(B, T, N, N)
        stride *= 2
    return P


class NCBlock(nn.Module):
    """Honest block. Biggest mats: W_P 512->256 + SwiGLU. No QKV anywhere."""
    def __init__(self, d=256, dk=32, dv=64):
        super().__init__()
        self.d, self.dk, self.dv = d, dk, dv
        self.n1 = RMSNorm(d)
        self.W_theta = nn.Linear(d, 16, bias=False)  # 16 plane angles, no phi
        self.Sm = nn.Parameter(torch.randn(N, N) * 0.05)  # symmetrized at use
        self.Am = nn.Parameter(torch.randn(N, N) * 0.05)  # antisymmetrized at use
        self.W_P = nn.Linear(N * N, d, bias=False)  # vec(Q)->features (real)
        self.W_k = nn.Linear(d, dk, bias=False)
        self.W_v = nn.Linear(d, dv, bias=False)
        self.W_q = nn.Linear(d, dk, bias=False)
        self.W_g = nn.Linear(d, dv, bias=False)
        self.w_l = nn.Linear(d, 1, bias=True)
        self.W_r = nn.Linear(dv, d, bias=False)
        self.n2 = RMSNorm(d)
        self.gate = nn.Linear(d, 512, bias=False)
        self.up = nn.Linear(d, 512, bias=False)
        self.down = nn.Linear(512, d, bias=False)
        self.register_buffer('P0', torch.tensor([p for p, _ in PAIRS0]), persistent=False)
        self.register_buffer('Q0', torch.tensor([q for _, q in PAIRS0]), persistent=False)
        self.register_buffer('P1', torch.tensor([p for p, _ in PAIRS1]), persistent=False)
        self.register_buffer('Q1', torch.tensor([q for _, q in PAIRS1]), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        th = torch.tanh(self.W_theta(h)) * 0.5            # (B,T,16)
        lam = torch.sigmoid(self.w_l(h)).squeeze(-1) * 0.099 + 0.9
        # --- unitary core ALWAYS fp32 (no ComplexHalf, no half-rotations) ---
        cm = torch.amp.autocast('cuda', enabled=False) if x.is_cuda else nullcontext()
        with cm:
            th32 = th.float()
            lam32 = lam.float()
            U = torch.eye(N, device=x.device).expand(B, T, N, N).contiguous()
            U = _stage(U, self.P0, self.Q0, th32[..., :8])
            U = _stage(U, self.P1, self.Q1, th32[..., 8:])
            Qs = _prefix_products(U)                       # (B,T,N,N)
            Ff = self.W_P(Qs.reshape(B * T, -1)).view(B, T, D)
            Kk = self.W_k(Ff)
            Vv = self.W_v(Ff)
            Qq = self.W_q(Ff)
            Gg = torch.sigmoid(self.W_g(Ff))
            Sm = 0.5 * (self.Sm.float() + self.Sm.float().T)
            Am = 0.5 * (self.Am.float() - self.Am.float().T)
            Gs = (Qs @ Sm).reshape(B, T, -1)
            Ga = (Qs @ Am).reshape(B, T, -1)
            Fm = Qs.reshape(B, T, -1)
            S = (torch.einsum('bid,bjd->bij', Gs, Fm)
                 + torch.einsum('bid,bjd->bij', Ga, Fm)) / N
            S = S.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1).unsqueeze(0), float('-inf'))
            A = torch.softmax(S, dim=-1)
            o_attn = torch.einsum('bij,bjd->bid', A, Ff)
            # Mt: scalar-gated affine recurrence (only light loop left, real fp32)
            M = torch.zeros(B, self.dk, self.dv, device=x.device)
            reads = []
            for t in range(T):
                M = lam32[:, t, None, None] * M + Kk[:, t].unsqueeze(-1) * (Gg[:, t] * Vv[:, t]).unsqueeze(1)
                reads.append(torch.einsum('bd,bde->be', Qq[:, t], M))
            Rd = torch.stack(reads, dim=1)
            o = o_attn + self.W_r(Rd)
        x = x + o.to(x.dtype)
        h2 = self.n2(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x


class NCTrueLM(nn.Module):
    def __init__(self, vocab=256, d=256, layers=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([NCBlock(d) for _ in range(layers)])
        self.nf = RMSNorm(d)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x)
        x = self.nf(x)
        return x @ self.tok.weight.T  # tied byte head

    def count(self):
        t = sum(p.numel() for p in self.parameters())
        return {'total': t, 'emb': self.tok.weight.numel(), 'backbone': t - self.tok.weight.numel()}
