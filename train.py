"""Train NC-Decoder-768 on FineWeb-Edu, single Nvidia T4 (Kaggle).
Usage (Kaggle):
  pip install -r requirements.txt
  python train.py --steps 20000 --seq 1024 --batch 4 --accum 8 --lr 3e-4 --out /kaggle/working/nc768
Memory: ~110M params (38M emb tied + ~72M backbone).
  fp16 + SDPA-manual + grad-accum -> fits 16GB T4 at batch=4, seq=1024.
Dataset: HuggingFaceFW/fineweb-edu sample-10BT, streaming, GPT2 BPE.
"""
import argparse, math, os, random, time
import torch
import torch.nn.functional as F
from model import NCDecoderLM

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--seq', type=int, default=1024)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--out', type=str, default='/kaggle/working/nc768')
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()

def lr_schedule(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))

def main():
    a = get_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    V = tok.vocab_size  # 50257

    model = NCDecoderLM(vocab=V, d=768, layers=12, heads=12,
                        ffn_mult=2, k_phase=32, lora_r=32).to(dev)
    info = model.count_params()
    print(f"params total={info['total']/1e6:.1f}M emb={info['emb']/1e6:.1f}M backbone={info['backbone']/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))

    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                      split='train', streaming=True)
    ds = ds.shuffle(seed=a.seed, buffer_size=10000)

    buf = []
    it = iter(ds)
    step = 0
    model.train()
    t0 = time.time()
    running = 0.0
    opt.zero_grad()

    def next_batch():
        nonlocal it
        batch_ids = []
        while len(batch_ids) < a.batch:
            try:
                ex = next(it)
            except StopIteration:
                it = iter(ds); ex = next(it)
            text = ex.get('text', '')
            if not text or len(text) < 50:
                continue
            ids = tok(text, truncation=False, add_special_tokens=False)['input_ids']
            batch_ids.extend(ids + [tok.eos_token_id])
        # pack stream into batch x seq
        need = a.batch * (a.seq + 1)
        while len(batch_ids) < need:
            try: ex = next(it)
            except StopIteration:
                it = iter(ds); ex = next(it)
            text = ex.get('text', '')
            ids = tok(text, truncation=False, add_special_tokens=False)['input_ids']
            batch_ids.extend(ids + [tok.eos_token_id])
        arr = torch.tensor(batch_ids[:need], dtype=torch.long).view(a.batch, a.seq + 1)
        return arr[:, :-1], arr[:, 1:]

    while step < a.steps:
        for m in range(a.warmup if False else 0):
            pass
        for g in range(a.accum):
            xb, yb = next_batch()
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda'), dtype=torch.float16):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1)) / a.accum
            scaler.scale(loss).backward()
            running += loss.item() * a.accum
        # lr
        lr = lr_schedule(step, a.steps, a.warmup, a.lr)
        for pg in opt.param_groups: pg['lr'] = lr
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()
        step += 1
        if step % a.log_every == 0:
            dt = time.time() - t0
            avg = running / (a.log_every * a.accum)
            # tokens/sec
            toks = a.log_every * a.accum * a.batch * a.seq / max(dt, 1e-6)
            ppl = math.exp(min(avg, 10))
            print(f"step {step}/{a.steps} loss {avg:.3f} ppl {ppl:.1f} lr {lr:.2e} tok/s {toks:.0f}", flush=True)
            running = 0.0; t0 = time.time()
            # tiny sample
            if step % (a.log_every * 4) == 0:
                model.eval()
                with torch.no_grad():
                    prompt = torch.tensor([[tok.eos_token_id]], device=dev)
                    for _ in range(60):
                        lg = model(prompt[:, -256:])
                        nx = torch.argmax(lg[:, -1], dim=-1, keepdim=True)
                        prompt = torch.cat([prompt, nx], dim=1)
                    print('SAMPLE:', tok.decode(prompt[0].tolist()[:80])[:400].replace('\n', ' '))
                model.train()
        if step % a.save_every == 0:
            torch.save({'step': step, 'model': model.state_dict(),
                        'opt': opt.state_dict()},
                       os.path.join(a.out, f'ckpt_{step}.pt'))
    torch.save({'step': step, 'model': model.state_dict()},
               os.path.join(a.out, 'final.pt'))
    print('done, saved to', a.out)

if __name__ == '__main__':
    main()
