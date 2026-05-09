# Moderate Mode Setup

**~25,000 ZINC molecules · 250 epochs · CPU · ~60–90 minutes**

Moderate mode trains on a 25,000-molecule subset of ZINC-250K alongside the 295 curated seeds. Generated molecules are more diverse and explore further from kinase scaffolds.

---

## What you need

- Python 3.8 or newer
- 8 GB RAM
- Internet on first run only (~2 min download, then cached)
- No GPU required

---

## Step 1 — Clone and install

```bash
git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
cd molecular-design-vae
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

## Step 2 — Train

```bash
python train_vae_extended.py --mode moderate
```

This will:

1. Download ZINC-250K (~2 min, cached for next time)
2. Filter to 25,000 drug-like molecules
3. Augment seeds 4× and ZINC molecules 2×
4. Train the VAE for 250 epochs (~60–90 min)

## Step 3 — Run the server

```bash
python serve.py
```

Open `index.html` in your browser. The UI is identical to basic mode — same 8 tabs, same controls, same behaviour. The model behind the scenes is just trained on 50× more data.

---

## Run on Google Colab (free, no laptop tied up)

```python
# Paste into a fresh Colab notebook (no GPU needed)
!git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
%cd molecular-design-vae
!pip install torch selfies flask flask-cors scipy pyngrok requests
!pip install rdkit
!python train_vae_extended.py --mode moderate
```

For the URL setup with ngrok, see [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md).

---

## What to expect vs basic mode

| Metric | Basic | Moderate |
|--------|:-----:|:--------:|
| Training molecules | ~1k | ~50k |
| Validity rate | 55–70% | 65–78% |
| Mean QED | 0.55–0.75 | 0.58–0.78 |
| Scaffold diversity | Moderate | Good |
| Training time | 5–8 min | 60–90 min |

---

## Second run is offline

ZINC data is cached after the first download. Subsequent runs work without internet:

```
INFO  Loading cached ZINC data from saved_model/zinc_cache_25000.txt
```

---

## Troubleshooting

**Download failed on first run**
Check your internet, retry. The cache means you only need to download once.

**Out of memory during training**
Edit `train_vae_extended.py` and reduce `batch_size` in the `moderate` config from 128 to 64.

**Training is slow**
On older hardware moderate mode can take >90 min. Leave it running.
