# Basic Mode Setup

**295 curated seed molecules · 150 epochs · CPU · ~5–8 minutes**

The fastest way to get started. No internet needed after cloning, no GPU, no waiting around. Good for trying things out, demos, and supervisor presentations.

---

## What you need

- Python 3.8 or newer
- 4 GB RAM
- No GPU

---

## Step 1 — Clone

```bash
git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
cd molecular-design-vae
```

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> RDKit is most reliable via conda:
> ```bash
> conda install -c conda-forge rdkit
> pip install -r requirements.txt
> ```

## Step 3 — Train

```bash
python train_vae.py
```

This trains the basic VAE in ~5–8 minutes. It produces:

```
saved_model/vae.pt           Trained model weights
saved_model/tokenizer.pkl    Fitted SELFIES tokenizer
saved_model/latents.pt       Pre-computed seed latents
saved_model/config.json      Hyperparameters and final validity rate
```

## Step 4 — Start the server

```bash
python serve.py
```

The Flask API runs on `http://localhost:5000`.

## Step 5 — Open the UI

Open `index.html` directly in your browser, or visit `http://localhost:5000`.

Enter a target like `EGFR kinase`, click **Generate Molecules**, and the pipeline runs.

---

## What you can do once running

All 8 tabs are functional:

- **Generate** — produce novel molecules for a target
- **Optimize Molecule** — paste a SMILES, get improved analogues
- **Scaffold Hop** — keep the core, change the substituents
- **Chemical Space** — 2D PCA of generated molecules
- **SAR Heatmap** — functional groups vs scores
- **Activity Cliffs** — similar molecules with very different rewards
- **MMP Analysis** — matched molecular pairs with single-bond changes
- **Retrosynthesis** — suggested synthetic routes

---

## Honest notes

**Docking scores in basic mode are estimates.** They are computed by RDKit from molecular properties (MW, LogP, ring count, pharmacophore hits) — not from real protein docking. They are useful for ranking molecules within a batch but are not wet-lab-grade affinities. For real Vina-based docking, use moderate or large modes with the optional docking add-on.

**Reward-guided RL is hill-climbing.** The "reward-guided optimisation" component does a directed random walk in the VAE latent space — it adds noise, decodes, scores, and moves toward better scores. It is not the full PPO algorithm with policy and value networks. This is honest and sufficient for the task; renaming makes the documentation match what the code actually does.

---

## What basic mode is good for

- First impression / demo
- Supervisor presentation
- Understanding the pipeline architecture
- Hosted version (the [live app](https://kaur-simarpreet-molecular-design-vae.hf.space) runs basic mode)

## What basic mode is NOT good for

- Exploring broad chemical space — only 295 seed molecules
- Generating molecules far from kinase scaffolds (most seeds are kinase-related)
- Final research results — use moderate or large mode

For broader diversity see [`SETUP_MODERATE.md`](SETUP_MODERATE.md) or [`SETUP_LARGE.md`](SETUP_LARGE.md).

---

## Troubleshooting

**"FileNotFoundError: saved_model/vae.pt"**
You haven't run `train_vae.py` yet. Run it once before starting the server.

**"RDKit module not found"**
Use conda: `conda install -c conda-forge rdkit`

**"Cannot reach backend" in browser**
Make sure `python serve.py` is still running. The UI needs both: `index.html` open in browser AND `serve.py` running on `localhost:5000`.
