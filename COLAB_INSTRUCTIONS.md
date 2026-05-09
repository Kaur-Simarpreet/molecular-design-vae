# Running on Google Colab

This guide shows how to train and serve any of the four modes on Google Colab — including how to expose the Flask server via a public URL using ngrok.

---

## Quick reference

| Mode | Colab works? | Live URL? | What you get |
|------|:------------:|:---------:|--------------|
| basic | Yes | Yes | Full UI via ngrok |
| moderate | Yes | Yes | Full UI via ngrok |
| **large** | **Will time out** | **No** | Results in notebook cells |
| large-gpu | Yes (T4 GPU) | Yes | Full UI via ngrok |

**Important:** Large CPU mode trains for 4–6 hours of pure compute. Free Colab times out before this finishes. For Large mode use **Large GPU** instead, or run **Large CPU locally overnight**.

---

## Setting up ngrok (one-time, ~2 min)

Ngrok exposes your Colab Flask server to the public internet via a temporary URL.

1. Go to https://dashboard.ngrok.com/signup and create a free account (no credit card)
2. Visit https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your auth token (looks like `2abc...XYZ`)
4. Use this token in the cell below

---

## Basic mode notebook

Paste these cells into a fresh Colab notebook:

### Cell 1 — Setup

```python
!git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
%cd molecular-design-vae
!pip install -q torch selfies flask flask-cors scipy pyngrok requests
!pip install -q rdkit
print("Setup complete")
```

### Cell 2 — Train (5–8 min)

```python
!python train_vae.py
```

### Cell 3 — Serve via ngrok (live URL)

```python
import os
from pyngrok import ngrok, conf
import threading
import subprocess

# Replace with your ngrok auth token
NGROK_TOKEN = "PASTE_YOUR_NGROK_AUTH_TOKEN_HERE"
conf.get_default().auth_token = NGROK_TOKEN

# Start serve.py in the background
def run_server():
    subprocess.run(["python", "serve.py"])

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait a moment for the server to come up
import time
time.sleep(8)

# Open ngrok tunnel to port 5000
public_url = ngrok.connect(5000)
print(f"\n=== UI available at: {public_url} ===\n")
print("Open this URL in any browser. The full UI works identically to localhost.")
```

### Cell 4 — Use it

Open the URL printed by Cell 3 in any browser. All 8 tabs work — Generate, Optimize Molecule, Scaffold Hop, Chemical Space, SAR Heatmap, Activity Cliffs, MMP Analysis, Retrosynthesis.

When you're done, you can also download the trained model files:

```python
from google.colab import files
import shutil
shutil.make_archive('saved_model', 'zip', 'saved_model')
files.download('saved_model.zip')
```

---

## Moderate mode notebook

Same as basic but Cell 2 is:

```python
!python train_vae_extended.py --mode moderate
```

Training takes 60–90 min. Cells 3 and 4 are identical to basic mode.

---

## Large GPU mode notebook (recommended for best results)

### First — switch to GPU runtime

**Runtime → Change runtime type → T4 GPU → Save**

### Cell 1 — Setup (same as before, but verify GPU)

```python
!git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
%cd molecular-design-vae
!pip install -q torch selfies flask flask-cors scipy pyngrok requests
!pip install -q rdkit

import torch
assert torch.cuda.is_available(), "GPU not detected — change runtime type to T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2 — Train (20–40 min on T4)

```python
!python train_vae_extended.py --mode large-gpu
```

### Cells 3 and 4 — same as basic mode

The serve and use cells are identical. The trained model is bigger (4.9M params vs 1.2M) but the API and UI are the same.

---

## Large CPU mode notebook (no live URL)

Free Colab will time out before Large CPU finishes training. The notebook for this mode therefore does not start a server. Instead, it shows the trained model's results directly in the notebook.

### Cell 1 — Setup (same as before)

```python
!git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
%cd molecular-design-vae
!pip install -q torch selfies flask flask-cors scipy requests
!pip install -q rdkit
```

### Cell 2 — Train (will probably time out, but you'll get a partial best checkpoint)

```python
!python train_vae_extended.py --mode large
```

### Cell 3 — Use the partial checkpoint and show results in notebook

```python
# Load the best checkpoint (saved during training even if interrupted)
import shutil, os
if os.path.exists("saved_model/vae_best.pt") and not os.path.exists("saved_model/vae.pt"):
    shutil.copy("saved_model/vae_best.pt", "saved_model/vae.pt")

# Generate sample molecules using the API directly (no server, no UI)
import sys
sys.path.insert(0, '.')
from serve import vae, tokenizer, score_mol
import torch
import random

generated = []
for _ in range(20):
    z = torch.randn(1, vae.encoder.mu.out_features)
    smi = vae.decode_z(z, tokenizer, temperature=0.9)
    if smi:
        scored = score_mol(smi, "EGFR kinase", weights={"qed":0.30, "dock":0.40, "sa":0.15, "nov":0.05, "admet":0.10})
        if scored:
            generated.append(scored)

# Show as a table
import pandas as pd
df = pd.DataFrame(generated)
print(df[['smiles', 'qed', 'sa_score', 'docking_score', 'rl_reward']].to_string())

# Export CSV
df.to_csv('large_mode_results.csv', index=False)
print("\nResults saved to large_mode_results.csv")
```

### Cell 4 — Download results

```python
from google.colab import files
files.download('large_mode_results.csv')
files.download('saved_model.zip')   # if you also want the trained model
```

---

## Practical tips

**Keep the Colab tab open** — closing it stops the runtime. Use a second tab for the ngrok URL.

**Free tier limits** — sessions auto-disconnect after 90 min of inactivity. To keep alive during long training, click in the notebook periodically or use a keep-alive script.

**Download results before disconnecting** — Colab's filesystem is wiped on disconnect. Always download `saved_model.zip` and any results before you close the tab.

**ngrok URL changes** — every time you restart Cell 3 you get a new URL. Save it somewhere if you'll use it for a while.
