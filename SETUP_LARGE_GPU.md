# Large GPU Mode Setup

**~100,000 ZINC molecules · 400 epochs · GPU · ~20–40 minutes**

Identical to large mode in every way — same dataset, same model architecture, same output quality — but uses your GPU. Training that takes 4–6 hours on CPU completes in 20–40 minutes on a modern GPU.

This is the recommended way to get the best model.

---

## GPU requirements

| GPU | Approx training time | Works? |
|-----|:-------------------:|:------:|
| RTX 3060 / 3070 | ~40 min | ✓ |
| RTX 3080 / 3090 | ~25 min | ✓ |
| RTX 4070 / 4080 | ~20 min | ✓ |
| RTX 4090 | ~15 min | ✓ |
| Google Colab T4 (free) | ~35 min | ✓ |
| Google Colab A100 (Pro) | ~12 min | ✓ |
| Apple M1/M2/M3 (MPS) | ~50 min | ✓ |

Minimum VRAM: 4 GB. The model is ~5M parameters — light by GPU standards.

This is your own consumer GPU, **not** a data centre. NVIDIA MolMIM requires A100/H100 — this project runs on an RTX 3060 gaming card.

---


---

## Optional: enable real DiffDock docking

Large GPU mode supports **real DiffDock blind docking** against EGFR 1IEP — state-of-the-art deep-learning docking. Without setup, you get RDKit-estimated scores. With setup, you get real binding affinities from DiffDock's neural network.

```bash
bash setup_docking.sh --diffdock
```

Takes ~10 minutes (downloads ~3 GB of model weights). Then run training as normal — DiffDock is used automatically.

Adds ~30 seconds per molecule to generation time on GPU. Full guide: [`DOCKING_SETUP.md`](DOCKING_SETUP.md).

## Local setup with GPU

### Step 1 — Install CUDA PyTorch

Check your CUDA version:
```bash
nvidia-smi
```

Install PyTorch with matching CUDA support:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Step 2 — Install the rest

```bash
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

Verify GPU is detected:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

### Step 3 — Train

```bash
python train_vae_extended.py --mode large-gpu
```

The script auto-detects CUDA. If your GPU is unavailable it warns and falls back to CPU automatically.

### Step 4 — Run the server

```bash
python serve.py
```

Open `index.html`. The UI is identical to other modes.

---

## Google Colab (free GPU — recommended path)

If you don't have a local GPU, use Colab's free T4:

1. Open https://colab.research.google.com
2. Click **Runtime → Change runtime type → T4 GPU**
3. Paste this in a cell and run:

```python
!git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
%cd molecular-design-vae
!pip install torch selfies flask flask-cors scipy pyngrok requests
!pip install rdkit
!python train_vae_extended.py --mode large-gpu
```

Training takes ~35 min on free T4. After it finishes, see [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md) for the ngrok URL setup that exposes the UI.

---

## Apple Silicon (M1/M2/M3)

PyTorch supports Apple's MPS backend natively:

```bash
pip install torch
python -c "import torch; print(torch.backends.mps.is_available())"  # should print True
```

The training script auto-detects MPS when you select `large-gpu` mode. Training is slightly slower than a mid-range NVIDIA GPU but much faster than CPU.

---

## What to expect

| Metric | Large CPU | Large GPU |
|--------|:---------:|:---------:|
| Training time | 4–6 hrs | 20–40 min |
| Final validity | 72–82% | 75–85% |
| Output quality | Same | Same |
| Model size | 4.9M params | 4.9M params |

The output files in `saved_model/` are identical regardless of which hardware trained them.

---

## Troubleshooting

**"GPU not found" warning**
The script falls back to CPU automatically. Either install CUDA PyTorch (see Step 1) or just use Large CPU mode if you're patient.

**CUDA out of memory**
Reduce `batch_size` in the `large-gpu` config from 512 to 256 or 128.

**Apple MPS slower than expected**
MPS is ~2× slower than equivalent NVIDIA GPU. This is normal — it's still much faster than CPU.
