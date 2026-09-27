"""Smoke: params<=3M, no QKV, commutator!=0, causal, short train run. Usage: python smoke.py [--seq 256]"""
import argparse, os, re, time
import torch, torch.nn.functional as F
from model import NCTrueLM, N, _stage

p = argparse.ArgumentParser()
p.add_argument('--seq', type=int, default=256)
a = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))  # never grep the repo-root clone again
torch.manual_seed(0)
m = NCTrueLM(vocab=256, d=256, layers=4)
c = m.count()
print(c)
assert c['total'] <= 3_000_000, f"TOO BIG: {c['total']}"

# 1. no QKV in OUR file (not the old clone at repo root)
src = open(os.path.join(HERE, 'model.py')).read()
bad = re.findall(r'Linear\s*\(\s*d\s*,\s*3\s*\*\s*d', src)
assert not bad, f"QKV FOUND: {bad}"
assert 'torch.exp(1j' not in src and 'complex64' not in src and 'view_as_real' not in src, "complex leftovers found"
print('no-QKV, no-complex OK')

# 2. commutator: U_a U_b != U_b U_a (real SO)
m.eval()
with torch.no_grad():
    b0 = m.blocks[0]
    P0 = torch.tensor([p for p, _ in [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]])
    Q0 = torch.tensor([q for _, q in [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]])
    P1 = torch.tensor([p for p, _ in [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,0)]])
    Q1 = torch.tensor([q for _, q in [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,0)]])
    def unit_of(tok):
        x1 = m.tok.weight[tok].unsqueeze(0).unsqueeze(0)
        th = (torch.tanh(b0.W_theta(b0.n1(x1))) * 0.5)[0, 0]  # (16,)
        U = torch.eye(N).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        U = _stage(U, P0, Q0, th[:8].unsqueeze(0).unsqueeze(0))
        U = _stage(U, P1, Q1, th[8:].unsqueeze(0).unsqueeze(0))
        return U[0, 0]
    Ua, Ub = unit_of(10), unit_of(200)
    comm = (Ua @ Ub - Ub @ Ua).abs().max().item()
    print('commutator max:', comm)
    assert comm > 1e-4, "COMMUTES - fake!"
    x1 = torch.tensor([[10, 200]])
    x2 = torch.tensor([[200, 10]])
    d = (m(x1)[0, -1] - m(x2)[0, -1]).abs().max().item()
    print('swap diff:', d)
    assert d > 1e-4, "ORDER INVARIANT!"
print('non-commutative OK')

# 3. causal
with torch.no_grad():
    full = torch.randint(0, 256, (1, 16))
    assert (m(full)[0, 5] - m(full[:, :6])[0, 5]).abs().max().item() < 1e-4
print('causal OK')

# 4. train few steps at --seq + tok/s
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
m.train()
t0 = time.time()
for s in range(3):
    x = torch.randint(0, 256, (2, a.seq))
    loss = F.cross_entropy(m(x).view(-1, 256), torch.roll(x, -1, 1).reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    print(f'step {s} loss {loss.item():.3f}')
print(f'tok/s (CPU): {3 * 2 * a.seq / (time.time() - t0):.0f}')

# 5. loop == parallel dual form (exact same math)
B, T, dk, dv = 2, 17, 8, 12
lam = 0.9 + 0.099 * torch.rand(B, T)
K = torch.randn(B, T, dk); V = torch.randn(B, T, dv)
Q = torch.randn(B, T, dk); G = torch.rand(B, T, dv)
M = torch.zeros(B, dk, dv); ref = []
for t in range(T):
    M = lam[:, t, None, None] * M + K[:, t].unsqueeze(-1) * (G[:, t] * V[:, t]).unsqueeze(1)
    ref.append(torch.einsum('bd,bde->be', Q[:, t], M))
ref = torch.stack(ref, 1)
cum = torch.cumsum(torch.log(lam.clamp(min=1e-5)), 1)
dec = torch.exp(cum.unsqueeze(2) - cum.unsqueeze(1))
dec = dec.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0), 0.0)
par = torch.matmul(torch.matmul(Q, K.transpose(-2, -1)) * dec, G * V)
dd = (ref - par).abs().max().item()
print('dual-form diff:', dd)
assert dd < 1e-4, "dual form mismatch!"
print('SMOKE OK')
