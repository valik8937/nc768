"""Train honest NC decoder on FineWeb-Edu BYTES (vocab 256), 1x T4.
  python train.py --steps 20000 --seq 512 --batch 8 --accum 8 --out /kaggle/working/nc_true
Model ~2.47M, fp16, fits T4 easily (~2GB active).
"""
import argparse, math, os, random, time
import torch, torch.nn.functional as F
from model import NCTrueLM

def args_():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--seq', type=int, default=256)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--out', type=str, default='/kaggle/working/nc_true')
    p.add_argument('--log', type=int, default=50)
    p.add_argument('--save', type=int, default=2000)
    p.add_argument('--sample_every', type=int, default=500)
    p.add_argument('--sample_len', type=int, default=120)
    p.add_argument('--sample_temp', type=float, default=0.8)
    p.add_argument('--sample_topk', type=int, default=40)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--vocab', type=int, default=256)
    return p.parse_args()

def main():
    a = args_()
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)
    m = NCTrueLM(vocab=a.vocab, d=a.d, layers=a.layers, dropout=a.dropout).to(dev)
    print(m.count())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    sc = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=5000)
    it, stream = iter(ds), []
    def batch():
        nonlocal it, stream
        while len(stream) < a.batch * (a.seq + 1):
            try: ex = next(it)
            except StopIteration: it = iter(ds); ex = next(it)
            t = ex.get('text', '')
            if not t: continue
            stream.extend(list(t.encode('utf-8', errors='ignore')) + [0])
        blk = torch.tensor(stream[:a.batch * (a.seq + 1)]).view(a.batch, a.seq + 1)
        stream = stream[a.batch * (a.seq + 1):]
        return blk[:, :-1].to(dev), blk[:, 1:].to(dev)
    m.train(); run, t0, step = 0.0, time.time(), 0
    wall0 = time.time()
    while step < a.steps:
        for _ in range(a.accum):
            xb, yb = batch()
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda'), dtype=torch.float16):
                loss = F.cross_entropy(m(xb).view(-1, a.vocab), yb.reshape(-1)) / a.accum
            sc.scale(loss).backward(); run += loss.item() * a.accum
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.5 + 0.5 * math.cos(math.pi * step / a.steps))
        for pg in opt.param_groups: pg['lr'] = lr
        sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(opt); sc.update(); opt.zero_grad(); step += 1
        if step % a.log == 0:
            dt = time.time() - t0
            avg = run / (a.log * a.accum)
            tps = a.log * a.accum * a.batch * a.seq / max(dt, 1e-6)
            vram = torch.cuda.memory_allocated() / 1e9 if dev == 'cuda' else 0.0
            eta = (time.time() - wall0) / max(step, 1) * (a.steps - step)
            eh, er = divmod(int(eta), 3600); em, es = divmod(er, 60)
            print(f"[ {step}/{a.steps} | {100 * step / a.steps:.1f}%] loss: {avg:.3f} (bpb: {avg / math.log(2):.2f}) | lr: {lr:.1e} | {tps / 1000:.1f}k tok/s | VRAM: {vram:.1f}G | ETA: {eh:02d}:{em:02d}:{es:02d}", flush=True)
            run, t0 = 0.0, time.time()
            if step % a.sample_every == 0:
                m.eval()
                with torch.no_grad():
                    prompt = torch.tensor([list(b"The science of ")], device=dev)
                    for _ in range(a.sample_len):
                        lg = m(prompt[:, -256:])[:, -1] / max(a.sample_temp, 1e-3)
                        if a.sample_topk > 0:
                            v, _ = torch.topk(lg, min(a.sample_topk, lg.size(-1)))
                            lg = lg.masked_fill(lg < v[:, -1:], float('-inf'))
                        prompt = torch.cat([prompt, torch.multinomial(torch.softmax(lg, -1), 1)], dim=1)
                    txt = bytes(prompt[0].tolist()).decode('utf-8', errors='ignore')
                    print(f"--- [SAMPLE @ step {step} | T={a.sample_temp} k={a.sample_topk}] ---\n{txt[:300]}\n---------------------------", flush=True)
                m.train()
        if step % a.save == 0:
            torch.save({'step': step, 'model': m.state_dict()}, f"{a.out}/ckpt_{step}.pt")
    torch.save({'step': step, 'model': m.state_dict()}, f"{a.out}/final.pt")
    print('done')

if __name__ == '__main__':
    main()
