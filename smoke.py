"""Local CPU smoke-test: tiny NC model, random data, 5 train steps."""
import torch, torch.nn.functional as F
from model import NCDecoderLM

torch.manual_seed(0)
m = NCDecoderLM(vocab=512, d=64, layers=2, heads=4, ffn_mult=2, k_phase=8, lora_r=8)
print(m.count_params())
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
m.train()
for s in range(5):
    x = torch.randint(0, 512, (2, 32))
    y = torch.roll(x, -1, dims=1)
    logits = m(x)
    loss = F.cross_entropy(logits.view(-1, 512), y.view(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    print(f'step {s} loss {loss.item():.3f}')
# causal check: prefix vs full must match at position t
m.eval()
with torch.no_grad():
    full = torch.randint(0, 512, (1, 16))
    l_full = m(full)[0, 5]
    l_pref = m(full[:, :6])[0, 5]
    print('leakage diff:', (l_full - l_pref).abs().max().item())
print('SMOKE OK')
