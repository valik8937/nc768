# NC-768 — Kaggle T4 Quickstart

Папка `nc768/`:
- `model.py` — NC-Decoder d=768, 12 шарів, без PE, tied head + LoRA
- `train.py` — стрім FineWeb-Edu sample-10BT, fp16, grad-accum
- `smoke.py` — локальний CPU тест

## Kaggle (1x T4 16GB)
1. New Notebook -> GPU T4, internet ON.
2. Upload `nc768/` або `!git clone <repo>`.
3. Run:
```bash
pip install -q -r nc768/requirements.txt
python nc768/smoke.py   # опційно, швидка перевірка
python nc768/train.py --steps 20000 --seq 1024 --batch 4 --accum 8 --lr 3e-4 --out /kaggle/working/nc768
```
4. Чекпоінти: `/kaggle/working/nc768/ckpt_*.pt`.

## Чому влізе в T4
- ~110M total (38.6M emb tied + ~71M backbone), fp16 ~220MB ваг
- batch=4, seq=1024, accum=8 → eff 32 seq = 32k токенів/крок
- ~6-8k ток/с на T4 → 20k степів ≈ 18-24 год (в межах тижневого ліміту Kaggle, можна бити на чанки по 5k)
- Якщо OOM: `--batch 2 --accum 16` або `--seq 512`.

## Що міряти
- loss/ppl в лозі, семпли кожні 200 степів
- leakage diff ≈ 0 (каузальність), PPL має бити біграму вже на 1-2k степах
