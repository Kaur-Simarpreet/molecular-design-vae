# Large Mode Setup (CPU)

**~100,000 ZINC molecules · 400 epochs · CPU · ~4–6 hours**

Large mode trains on the full 100,000-molecule drug-like ZINC subset using a wider model (latent 256, hidden 512). Generated molecules are more chemically diverse, more novel, and better at exploring unexplored regions of drug-like space.

> **Important — Large CPU mode does NOT work well on Google Colab free tier.**  
> Free Colab times out before training finishes. Either run this **locally overnight**, or use **Large GPU mode** instead which finishes in 20–40 minutes.

---

## What you need

- Python 3.8 or newer
- 16 GB RAM (8 GB minimum)
- Internet on first run only (~8 min ZINC download)
- No GPU required (but a GPU would be much faster — see [`SETUP_LARGE_GPU.md`](SETUP_LARGE_GPU.md))

---


---

## Optional: enable real docking

Large mode supports **real AutoDock Vina docking** against EGFR 1IEP. Without setup, you get RDKit-estimated scores (same as basic/moderate). With setup, you get real molecular docking scores in kcal/mol.

```bash
bash setup_docking.sh --vina
```

Takes ~2 minutes. Then run training as normal — Vina is used automatically.

Adds ~25 seconds per molecule to generation time. Full guide: [`DOCKING_SETUP.md`](DOCKING_SETUP.md).

## Step 1 — Clone and install

```bash
git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
cd molecular-design-vae
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

## Step 2 — Train

```bash
python train_vae_extended.py --mode large
```

Compared to moderate:

| Setting | Moderate | Large |
|---------|:--------:|:-----:|
| Training molecules | ~50k | ~100k |
| Model parameters | 1.2M | 4.9M |
| Epochs | 250 | 400 |
| Batch size | 128 | 256 |

Roughly 1 epoch per minute on a modern CPU — 4–6 hours total.

---

## Tips for long training runs

### Run in background (Linux/Mac)

```bash
nohup python train_vae_extended.py --mode large > training.log 2>&1 &
tail -f training.log
```

### Run in background (Windows PowerShell)

```powershell
Start-Process python -ArgumentList "train_vae_extended.py --mode large" `
  -RedirectStandardOutput training.log
Get-Content training.log -Wait
```

### Best checkpoint is always saved

If training is interrupted, `saved_model/vae_best.pt` preserves the best validity reached so far. To use that partial checkpoint:

```bash
cp saved_model/vae_best.pt saved_model/vae.pt
python serve.py
```

---

## Step 3 — Run the server

```bash
python serve.py
```

Open `index.html`. The UI is identical to other modes.

---

## What to expect vs moderate mode

| Metric | Moderate | Large |
|--------|:--------:|:-----:|
| Training molecules | ~50k | ~100k |
| Model parameters | 1.2M | 4.9M |
| Validity rate | 65–78% | 72–82% |
| Scaffold diversity | Good | Excellent |
| Chemical novelty | Good | Excellent |
| Training time (CPU) | 60–90 min | 4–6 hrs |

---

## Why no Colab URL for Large CPU?

Free Google Colab times out after 90 minutes of inactivity, and has a 12-hour total session limit even with activity. Large CPU mode trains for 4–6 hours of pure compute — almost guaranteed to time out before completion.

You have two options:

1. **Run locally overnight** — train in a terminal, use the model the next morning
2. **Use Large GPU mode** — same model, same data, same output quality, but finishes in 20–40 minutes

The Colab notebook for large CPU mode shows results in notebook cells (validity curves, sample molecules, score table, CSV export) rather than launching a live UI server, because the server would die before you could interact with it.

---

## Troubleshooting

**Out of memory**
Reduce `batch_size` in the `large` config inside `train_vae_extended.py` from 256 down to 128.

**Training crashed mid-way**
Use `vae_best.pt`:
```bash
cp saved_model/vae_best.pt saved_model/vae.pt
python serve.py
```
