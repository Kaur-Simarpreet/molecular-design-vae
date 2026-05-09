# Real Docking Setup

This guide is for enabling **real molecular docking** in Large and Large GPU modes. Basic and Moderate modes always use mock RDKit-based estimates and need no setup.

---

## What you get

| Mode | Engine | Hardware | Time per molecule | Setup time |
|------|--------|----------|-------------------|-----------|
| Basic | Mock (RDKit) | CPU | instant | none |
| Moderate | Mock (RDKit) | CPU | instant | none |
| **Large** | **AutoDock Vina** | CPU | ~25 sec | 2 min |
| **Large GPU** | **DiffDock** | GPU | ~30 sec | 10 min (downloads ~3 GB) |

**Target:** EGFR Kinase, PDB ID `1IEP` — Imatinib-bound. Open-source PDB structure, no account needed.

---

## Quick start

```bash
# For Large mode (Vina, CPU)
bash setup_docking.sh --vina

# For Large GPU mode (DiffDock, GPU)
bash setup_docking.sh --diffdock

# Check what's installed
bash setup_docking.sh --status
```

After setup, real docking happens automatically when you run the matching training mode.

---

## Vina vs DiffDock — what's the difference?

**AutoDock Vina** (used in Large mode)
Classical force-field-based docking. Fast, decades of literature, used in thousands of published papers. Uses a predefined grid box around the active site — for EGFR 1IEP we use the Imatinib binding pocket coordinates.

- Setup: 2 minutes, ~50 MB
- Speed: ~25 seconds per molecule on CPU
- Accuracy: well-validated for kinase pockets like EGFR

**DiffDock** (used in Large GPU mode)
Deep-learning blind docking — uses a diffusion generative model. Doesn't need a predefined grid box; predicts where binding happens. State-of-the-art accuracy for novel chemotypes — exactly what this project generates.

- Setup: 10 minutes (downloads ~3 GB)
- Speed: ~30 seconds per molecule on GPU
- Accuracy: best available for novel molecules; benchmark-leading on PDBBind

For your project, both produce real binding scores against a real PDB structure. Vina is good enough for most use cases; DiffDock is meaningfully better if you have a GPU.

---

## What real docking actually means

These are **real molecular docking scores** — the molecule's 3D conformation is computed, placed into the protein's active site, and the binding energy is calculated using either a force field (Vina) or a deep neural network (DiffDock).

This is **not** wet-lab validation. Real docking against a real PDB structure is the strongest computational evidence you can produce, but binding affinity in vitro can still differ. Treat scores as strong ranking signals, not as absolute affinities.

---

## Setup details

### Vina (Large mode)

```bash
bash setup_docking.sh --vina
```

What it does:
1. Downloads `1IEP.pdb` from RCSB
2. Strips waters and ligands → keeps protein only
3. Converts to PDBQT format (Vina's input format)
4. Installs `vina` and `meeko` Python packages
5. Verifies everything works

After setup:
```bash
python train_vae_extended.py --mode large
python serve.py
# Open index.html — real Vina docking is now active
```

### DiffDock (Large GPU mode)

```bash
bash setup_docking.sh --diffdock
```

What it does:
1. Downloads `1IEP.pdb` from RCSB
2. Clones the official DiffDock repository
3. Installs PyTorch Geometric and DiffDock dependencies
4. Downloads DiffDock model weights (~1.5 GB) and ESM-2 protein language model weights (~1.5 GB)
5. Verifies the installation

After setup:
```bash
python train_vae_extended.py --mode large-gpu
python serve.py
# Real DiffDock blind docking is now active
```

---

## Troubleshooting

### `bash: setup_docking.sh: Permission denied`

```bash
chmod +x setup_docking.sh
bash setup_docking.sh --vina
```

### Vina install fails on Linux

```bash
sudo apt-get install autodock-vina
pip install meeko
```

### Vina install fails on macOS

```bash
brew install autodock-vina
pip install vina meeko
```

### DiffDock fails: "torch_geometric not found"

DiffDock needs specific torch versions. Try:
```bash
pip install torch_geometric torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)' | cut -d+ -f1).html
```

### "Real docking unavailable, using mock"

The pipeline never breaks — if real docking fails it falls back to mock scores with a warning. Run `bash setup_docking.sh --status` to see what's missing.

### Docking is too slow during generation

Vina at 25 sec/molecule × 6 molecules = ~2.5 min per generation request. This is expected for real docking. The UI shows a progress indicator. If you want faster results during exploration, switch to Moderate mode (mock docking, instant) and only use Large mode for final candidates.

---

## Active site — EGFR 1IEP

The Vina grid box is centred on the Imatinib binding pocket:

```
center:    (16.5, 50.5, 35.0)    # Angstroms
box size:  (22, 22, 22)          # Angstroms
```

These coordinates are hardcoded in `docking.py`. To use a different target, edit the `EGFR_TARGET` dict in that file.

DiffDock doesn't use a grid box — it predicts the binding site itself.

---

## Verifying real docking is active

After setup, check the server output when generating molecules. With real docking active you'll see log lines like:

```
INFO  Vina dock CC(=O)Oc1ccccc1C(=O)O = -7.42 kcal/mol
```

With mock docking:

```
WARNING  Vina or receptor not available — falling back to mock score.
```

Or run the diagnostic directly:

```bash
python docking.py
```

This prints a table of which engines are available and shows a test docking score for Aspirin.
