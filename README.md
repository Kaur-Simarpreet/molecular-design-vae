# Molecular Design VAE

A multi-stage generative framework for de novo drug discovery. Combines a SELFIES-based Beta-VAE with reward-guided latent space optimisation, NSGA-II Pareto ranking, and an interactive web interface for molecular analysis.

> Simarpreet Kaur (SR No: 26727) · M. Bhagya Sri (SR No: 27003)  
> Generative AI for Research — coursework project

---

## Try it now

The basic version is deployed and works without any setup — just click the link:

**[Try Basic Version Now →](https://kaur-simarpreet-molecular-design-vae.hf.space)**

The advanced versions (moderate, large, large-gpu) work locally only — see [Advanced versions](#advanced-versions) below.

---

## Overview

The pipeline learns a continuous latent representation of drug-like chemical space from a curated seed library, then uses reward-guided optimisation to navigate this space toward molecules with desirable properties: drug-likeness (QED), synthetic accessibility (SA), predicted binding affinity, and ADMET profile. SELFIES encoding guarantees that every generated structure is chemically valid by construction.

The system provides a complete browser-based analysis platform — chemical space PCA, SAR heatmaps, activity cliff detection, matched molecular pair analysis, scaffold hopping, retrosynthetic route suggestion, and 3D structure visualisation.

---

## The four versions

The architecture is identical across all four modes — only the training data, model width, and epochs change.

| Mode | Dataset | Model | Epochs | Hardware | Time | Where it runs |
|------|---------|-------|--------|----------|------|---------------|
| **Basic** | 295 curated seeds | 1.2M params | 150 | CPU | 5–8 min | Hosted app + local |
| **Moderate** | ~25k ZINC molecules | 1.2M params | 250 | CPU | 60–90 min | Local or Colab |
| **Large** | ~100k ZINC molecules | 4.9M params | 400 | CPU | 4–6 hours | **Local only** (see note) |
| **Large GPU** | ~100k ZINC molecules | 4.9M params | 400 | GPU | 20–40 min | Local or Colab GPU |

**Important note on Large CPU mode:** This mode does not provide a live URL when run on Google Colab. The free Colab tier times out before training finishes. To use Large mode you should either (a) run it on Colab and view results in the notebook output cells, or (b) train it locally overnight and serve from your own machine. If you have a GPU, use Large GPU mode instead.

Detailed setup for each version:
- [`docs/SETUP_BASIC.md`](docs/SETUP_BASIC.md)
- [`docs/SETUP_MODERATE.md`](docs/SETUP_MODERATE.md)
- [`docs/SETUP_LARGE.md`](docs/SETUP_LARGE.md)
- [`docs/SETUP_LARGE_GPU.md`](docs/SETUP_LARGE_GPU.md)
- [`docs/COLAB_INSTRUCTIONS.md`](docs/COLAB_INSTRUCTIONS.md) — running any mode on Google Colab

---

## Advanced versions

Moderate, Large, and Large-GPU modes give you broader chemical diversity but require local setup. The frontend dropdown for these modes redirects you here — that's intentional. The hosted web app runs basic mode only because the others need either substantial CPU time, a GPU, or both.

### Run on Google Colab (free, recommended for first-time users)

```python
# Paste this into a fresh Colab notebook cell and run
!git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
%cd molecular-design-vae
!pip install torch selfies flask flask-cors scipy pyngrok requests
!pip install rdkit
!python train_vae_extended.py --mode moderate    # or large, or large-gpu
# Then start serve.py and expose it via ngrok — see docs/COLAB_INSTRUCTIONS.md
```

For the full Colab walkthrough including ngrok URL setup, see [`docs/COLAB_INSTRUCTIONS.md`](docs/COLAB_INSTRUCTIONS.md).

### Run locally

```bash
git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
cd molecular-design-vae
pip install -r requirements.txt

# Pick a mode:
python train_vae.py                                # basic — 5-8 min
python train_vae_extended.py --mode moderate       # ~60-90 min
python train_vae_extended.py --mode large          # ~4-6 hours
python train_vae_extended.py --mode large-gpu      # ~20-40 min on GPU

python serve.py
# Open index.html in your browser
```

---

## Dataset

### Curated seed library (used by all modes)

295 hand-picked drug-like molecules from ChEMBL 34 and ZINC-250K, spanning 12 scaffold families: kinase inhibitors, GPCR ligands, protease inhibitors, amino acid derivatives, heterocyclics, urea/hydroxamic acids, sulfonamides, fluorinated compounds, morpholine/piperazine, pyrimidines, fused rings, and indole/benzimidazole. Includes 16 named reference drugs (Aspirin, Ciprofloxacin, Celecoxib, Adenosine, Salbutamol, Nicotine, Paracetamol, etc.). Every molecule is RDKit-validated and passes Lipinski's Rule of Five.

### External dataset (moderate, large, large-gpu)

Extended modes additionally use **ZINC-250K** — the standard benchmark dataset in molecular generation research:

- **View the dataset directly:** [github.com/aspuru-guzik-group/.../250k_rndm_zinc_drugs_clean_3.csv](https://github.com/aspuru-guzik-group/chemical_vae/blob/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv)
- **Source:** Gómez-Bombarelli et al. 2018, *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules*
- **Size:** 250,000 SMILES strings (~13 MB CSV)
- **Filter applied:** MW 150–500 Da, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10

Moderate mode uses the first 25,000 entries; large modes use the first 100,000. The download happens automatically on first run and is cached locally — subsequent runs work offline.

---

## Docking — what's real, what's an estimate

Different modes use different docking approaches:

| Mode | Docking | Hardware | Time per molecule |
|------|---------|----------|-------------------|
| Basic | Mock (RDKit estimate) | CPU | instant |
| Moderate | Mock (RDKit estimate) | CPU | instant |
| **Large** | **AutoDock Vina** (real) | CPU | ~25 sec |
| **Large GPU** | **DiffDock** (real, blind docking) | GPU | ~30 sec |

**Basic and Moderate** use RDKit-based property estimates — fast, useful for ranking molecules within a batch, but not real molecular docking. Acceptable for demonstration and exploration.

**Large and Large GPU** use real molecular docking against EGFR 1IEP (Imatinib binding pocket). Vina is the standard force-field method; DiffDock is a deep-learning blind docking model. Real docking requires one-time setup:

```bash
bash setup_docking.sh --vina       # for Large mode
bash setup_docking.sh --diffdock   # for Large GPU mode
```

Full setup guide: [`docs/DOCKING_SETUP.md`](docs/DOCKING_SETUP.md).

If real docking isn't set up, large modes fall back to mock scores automatically with a warning — the pipeline never breaks.

**Important:** Even real docking scores are computational predictions, not wet-lab affinities. Treat them as strong ranking signals, not absolute binding constants.

---

## Architecture

```
Seed library (ChEMBL 34) [+ ZINC-250K for extended modes]
    │
    ▼  SELFIES tokenisation + RDKit validation + atom-reorder augmentation
    │
    Beta-VAE
    ├── Encoder: Bidirectional LSTM → μ, log σ²
    ├── Latent:  z = μ + σ·ε  (reparameterisation)
    └── Decoder: Autoregressive LSTM → SELFIES tokens
    │
    ▼  Cyclic cosine KL annealing
    │
    Reward-guided latent space exploration
    │  Reward = w_qed·QED + w_dock·dock + w_sa·(1-sa/10) + w_nov·novelty + w_admet·admet
    │  (Hill-climbing in latent space — not full PPO. See docs/SETUP_BASIC.md for details.)
    │
    ▼
    8-stage quality filter
    │  MW · Tanimoto diversity · Lipinski · QED min · PAINS · ADMET · Novelty · Wet-lab
    │
    ▼
    NSGA-II Pareto ranking (QED↑, Docking↑, SA↓, Novelty↑)
    │
    ▼
    Wet-lab ready shortlist → Browser UI
```

### Hyperparameters by mode

| Parameter | Basic | Moderate | Large / Large-GPU |
|-----------|-------|----------|-------------------|
| Latent dim | 128 | 128 | 256 |
| LSTM hidden | 256 | 256 | 512 |
| Embedding dim | 64 | 64 | 128 |
| Max SELFIES length | 80 | 80 | 100 |
| Total epochs | 150 | 250 | 400 |
| Batch size | 64 | 128 | 256 / 512 |
| Learning rate | 5e-4 | 3e-4 | 2e-4 |
| β_max (KL weight) | 0.20 | 0.18 | 0.15 |

---

## Project structure

```
molecular-design-vae/
│
├── train_vae.py              Basic mode trainer (295 seeds, CPU)
├── train_vae_extended.py     Extended modes (moderate/large/large-gpu)
├── serve.py                  Flask API server
├── index.html                Web UI (basic mode in dropdown is active; others redirect here)
│
├── ablation.py               Run quantitative experiments, produce plots
├── docking.py                Vina + DiffDock + mock fallback (used in large modes)
├── setup_docking.sh          One-command real-docking setup
├── tests/test_pipeline.py    Unit tests
│
├── requirements.txt          Python dependencies
│
├── README.md                 This file
├── COMPARISON.md             Comparison with existing tools
├── COMPARISON.tex            LaTeX version
│
├── saved_model/              Created by training scripts
│
└── docs/
    ├── SETUP_BASIC.md
    ├── SETUP_MODERATE.md
    ├── SETUP_LARGE.md
    ├── SETUP_LARGE_GPU.md
    ├── DOCKING_SETUP.md
    └── COLAB_INSTRUCTIONS.md
```

---

## Testing

```bash
pip install pytest requests
pytest tests/
```

Tests cover SELFIES round-trip, seed library validity, augmentation correctness, and the `/health` endpoint (skipped automatically if the server is not running).

---

## Limitations — honest list

- **Mock docking by default.** All modes use RDKit-based property estimates instead of real protein docking. Real Vina/Gnina docking against EGFR 1IEP is documented as an optional add-on.
- **Reward-guided RL is not full PPO.** It is hill-climbing in the latent space with a reward signal — effective in practice but simpler than a published PPO implementation.
- **Basic seed library is small** (295 molecules, kinase-heavy). Use moderate or large for broader chemical diversity.
- **No wet-lab validation.** All outputs are computational predictions.
- **Target-agnostic generation.** The model does not condition directly on protein pocket structure.
- **~50% quality filter pass rate.** SELFIES guarantees chemical validity; only about half of generated molecules also pass all 8 drug-like filters.

---

## License

MIT

---

## Acknowledgements

- [SELFIES](https://github.com/aspuru-guzik-group/selfies) — Krenn et al. 2020
- [RDKit](https://www.rdkit.org/) — open-source cheminformatics
- ZINC-250K — Gómez-Bombarelli et al. 2018
- ChEMBL 34 — Mendez et al.

For comparison with related tools, see [`COMPARISON.md`](COMPARISON.md).
