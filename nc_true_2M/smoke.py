"""Smoke: params<=3M, no QKV, commutator!=0, causal, 3 train steps."""
import torch, torch.nn.functional as F, re
from model import NCTrueLM, PAIRS

torch.manual_seed(0)
m = NCTrueLM(vocab=256, d=256, layers=4)
c = m.count()
print(c)
assert c['total'] <= 3_000_000, f"TOO BIG: {c['total']}"

# 1. no QKV: forbidden pattern Linear(d,3*d)
src = open('model.py').read()
bad = re.findall(r'Linear\s*\(\s*d\s*,\s*3\s*\*\s*d', src)
assert not bad, f"QKV FOUND: {bad}"
code = [l for l in src.splitlines() if not l.strip().startswith(('"""', '#', '*'))]
code_s = '\n'.join(code).lower()
assert 'self.qkv' not in code_s and 'self.qk' not in code_s and 'self.q_' not in code_s, "qkv attr found"
print('no-QKV OK')

# 2. commutator: U_a U_b != U_b U_a for two different tokens
m.eval()
with torch.no_grad():
    e = m.tok.weight
    # build two unitaries from two embeddings via block0 thetas
    b0 = m.blocks[0]
    xa = e[10].unsqueeze(0).unsqueeze(0).expand(1, 1, -1)
    xb = e[200].unsqueeze(0).unsqueeze(0).expand(1, 1, -1)
    def unit_of(x1):
        h = b0.n1(x1)
        th = (torch.tanh(b0.W_theta(h)) * 0.5)[0, 0]
        ph = b0.W_phi(h)[0, 0]
        Q = torch.eye(16, dtype=torch.complex64)
        for k, (p, q) in enumerate(PAIRS):
            c_, s_ = torch.cos(th[k]), torch.sin(th[k])
            ee = torch.exp(1j * ph[k])
            Rp, Rq = Q[p, :].clone(), Q[q, :].clone()
            Q[p, :] = c_ * Rp - s_ * torch.conj(ee) * Rq
            Q[q, :] = s_ * ee * Rp + c_ * Rq
        return Q
    Ua, Ub = unit_of(xa), unit_of(xb)
    comm = (Ua @ Ub - Ub @ Ua).abs().max().item()
    print('commutator max:', comm)
    assert comm > 1e-4, "COMMUTES - fake non-commutative!"
    # order swap changes doc state
    x1 = torch.tensor([[10, 200]])
    x2 = torch.tensor([[200, 10]])
    with torch.no_grad():
        l1 = m(x1)[0, -1]
        l2 = m(x2)[0, -1]
    d = (l1 - l2).abs().max().item()
    print('swap diff:', d)
    assert d > 1e-4, "ORDER INVARIANT - PE needed, fake!"
print('non-commutative OK')

# 3. causal: prefix vs full
with torch.no_grad():
    full = torch.randint(0, 256, (1, 16))
    assert (m(full)[0, 5] - m(full[:, :6])[0, 5]).abs().max().item() < 1e-4
print('causal OK')

# 4. train 3 steps, loss ~ ln256=5.54 start
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
m.train()
for s in range(3):
    x = torch.randint(0, 256, (2, 32))
    loss = F.cross_entropy(m(x).view(-1, 256), torch.roll(x, -1, 1).reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    print(f'step {s} loss {loss.item():.3f}')
print('SMOKE OK')
