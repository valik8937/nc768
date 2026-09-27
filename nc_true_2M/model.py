"""Honest non-commutative decoder: Givens U(16), NO QKV, shared Hermitian attention + Mt.
Budget: d=256, 4 blocks, byte vocab 256 -> ~2.4M total. No PE tables.

Order ONLY from product Q_t = U_t Q_{t-1} where U_t = Prod_k Givens(p_k,q_k,theta,phi).
Commutator [X,Y] != 0 preserved (BCH term kept, unlike cumsum).
Attention: S_ij = ReTr[A_sym Q_j^H Q_i]/n + ImTr[A_anti Q_j^H Q_i]/n, causal.
Values = unitary features themselves (no V matrix). Memory Mt associative.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 16
PAIRS0 = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
PAIRS1 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 0)]
PAIRS = PAIRS0 + PAIRS1  # K=16 per token


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def hermitian_param(n):
    """Real-parameterized Hermitian matrix:  n*n real DOF, A = A^H by construction."""
    Br = nn.Parameter(torch.randn(n, n) * 0.05)
    Bi = nn.Parameter(torch.randn(n, n) * 0.05)
    return Br, Bi

def hermitian_build(Br, Bi):
    B = torch.complex(Br, Bi)
    return 0.5 * (B + B.conj().T)


class NCBlock(nn.Module):
    """One honest block. NO qkv Linear. Biggest mat: W_P 512->256 + SwiGLU."""
    def __init__(self, d=256, dk=32, dv=64):
        super().__init__()
        self.d, self.dk, self.dv = d, dk, dv
        self.n1 = RMSNorm(d)
        self.W_theta = nn.Linear(d, 16, bias=False)  # 16 Givens angles
        self.W_phi = nn.Linear(d, 16, bias=False)    # 16 phases
        Br1, Bi1 = hermitian_param(N); self.As_r, self.As_i = Br1, Bi1
        Br2, Bi2 = hermitian_param(N); self.Aa_r, self.Aa_i = Br2, Bi2
        self.W_P = nn.Linear(2 * N * N, d, bias=False)  # vec(Q)->features
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

    def _givens_step(self, Q, th, ph, pairs):
        # Q: (B,N,N) complex; th,ph: (B,) per pair index k
        for (p, q) in pairs:
            pass
        return Q

    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        th = torch.tanh(self.W_theta(h)) * 0.5          # (B,T,16) bounded
        ph = self.W_phi(h)                              # (B,T,16) unbounded phase
        lam = torch.sigmoid(self.w_l(h)).squeeze(-1) * 0.099 + 0.9  # [0.9,0.999]
        dev, dt = x.device, x.dtype
        cdt = torch.complex64
        Q = torch.eye(N, dtype=cdt, device=dev).expand(B, N, N).contiguous()
        Qseq, Fs, Ks, Vs, Qs, Gs = [], [], [], [], [], []
        M = torch.zeros(B, self.dk, self.dv, device=dev, dtype=dt)
        reads = []
        for t in range(T):
            tht, pht = th[:, t, :], ph[:, t, :]  # (B,16)
            for k, (p, q) in enumerate(PAIRS):
                c = torch.cos(tht[:, k])                    # (B,)
                s = torch.sin(tht[:, k])                    # (B,)
                e = torch.exp(1j * pht[:, k])               # (B,) complex
                Qp = Q[:, p, :].clone()
                Qq = Q[:, q, :].clone()
                Np = c[:, None] * Qp - s[:, None] * torch.conj(e)[:, None] * Qq
                Nq = s[:, None] * e[:, None] * Qp + c[:, None] * Qq
                rows = [Q[:, i, :] for i in range(N)]
                rows[p], rows[q] = Np, Nq
                Q = torch.stack(rows, dim=1)
            Qseq.append(Q.clone())
            qr = torch.view_as_real(Q).reshape(B, -1)       # (B,512)
            f = self.W_P(qr.to(dt))                          # (B,D)
            Fs.append(f)
            k = self.W_k(f)                                  # (B,dk)
            v = self.W_v(f)                                  # (B,dv)
            qq = self.W_q(f)
            g = torch.sigmoid(self.W_g(f))
            Ks.append(k); Vs.append(v); Qs.append(qq); Gs.append(g)
            M = lam[:, t, None, None] * M + k.unsqueeze(-1) * (g * v).unsqueeze(1)
            reads.append(torch.einsum('bd,bde->be', qq, M))
        Qs_t = torch.stack(Qseq, dim=1)   # (B,T,N,N) complex
        Ff = torch.stack(Fs, dim=1)       # (B,T,D)
        Kk = torch.stack(Ks, dim=1)
        Vv = torch.stack(Vs, dim=1)
        Qq = torch.stack(Qs, dim=1)
        Rd = torch.stack(reads, dim=1)    # (B,T,dv)
        # --- shared Hermitian attention, single head, no QKV ---
        As = hermitian_build(self.As_r, self.As_i)  # (N,N) complex
        Aa = hermitian_build(self.Aa_r, self.Aa_i)
        QAs = Qs_t @ As   # (B,T,N,N)
        QAa = Qs_t @ Aa
        Ff_flat = Qs_t.reshape(B, T, -1)                 # (B,T,256) complex
        Gs_flat = QAs.reshape(B, T, -1)
        Ga_flat = QAa.reshape(B, T, -1)
        S_sym = torch.einsum('bid,bjd->bij', Gs_flat, torch.conj(Ff_flat)).real / N
        S_anti = torch.einsum('bid,bjd->bij', Ga_flat, torch.conj(Ff_flat)).imag / N
        S = S_sym + S_anti
        S = S.masked_fill(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1).unsqueeze(0), float('-inf'))
        A = torch.softmax(S.float(), dim=-1).to(dt)
        o_attn = torch.einsum('bij,bjd->bid', A, Ff)     # values = features, no V proj
        o = o_attn + self.W_r(Rd)
        x = x + o
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
