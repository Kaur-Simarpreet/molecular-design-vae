"""
train_vae_extended.py
=====================
Extended training modes for the Molecular Design VAE.

Usage
-----
    python train_vae_extended.py --mode moderate    # ~25k mols, CPU, 60-90 min
    python train_vae_extended.py --mode large       # ~100k mols, CPU, 4-6 hrs
    python train_vae_extended.py --mode large-gpu   # ~100k mols, GPU, 20-40 min

After training, start the server with:
    python serve.py

Dataset
-------
ZINC-250K — open source, no account needed.
View:   https://github.com/aspuru-guzik-group/chemical_vae/blob/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
Source: Gomez-Bombarelli et al. 2018, Chemical VAE

Filter applied: MW 150-500 Da, LogP <= 5, HBD <= 5, HBA <= 10
Cached locally after first download — subsequent runs work offline.

Output (same files as basic mode — serve.py works for all modes)
------
    saved_model/vae.pt
    saved_model/vae_best.pt
    saved_model/tokenizer.pkl
    saved_model/latents.pt
    saved_model/config.json
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import urllib.request
import warnings
from typing import List

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

# Reuse seed library and model classes from the basic trainer
from train_vae import (
    SEED_SMILES, SEED_NAMES,
    SELFIESTokenizer, BetaVAE,
    augment_smiles, check_validity,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ZINC-250K — open source permanent URL
ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae"
    "/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

MODES = {
    "moderate": {
        "description":     "~25k ZINC molecules · 250 epochs · CPU · ~60-90 min",
        "n_external":      25_000,
        "total_epochs":    250,
        "warmup_epochs":   30,
        "cycle_len":       40,
        "beta_max":        0.18,
        "batch_size":      128,
        "lr":              3e-4,
        "embed_dim":       64,
        "hidden":          256,
        "latent":          128,
        "max_len":         80,
        "n_aug_external":  2,
        "validity_check":  50,
        "device_pref":     "cpu",
    },
    "large": {
        "description":     "~100k ZINC molecules · 400 epochs · CPU · ~4-6 hours",
        "n_external":      100_000,
        "total_epochs":    400,
        "warmup_epochs":   40,
        "cycle_len":       50,
        "beta_max":        0.15,
        "batch_size":      256,
        "lr":              2e-4,
        "embed_dim":       128,
        "hidden":          512,
        "latent":          256,
        "max_len":         100,
        "n_aug_external":  1,
        "validity_check":  50,
        "device_pref":     "cpu",
    },
    "large-gpu": {
        "description":     "~100k ZINC molecules · 400 epochs · GPU · ~20-40 min",
        "n_external":      100_000,
        "total_epochs":    400,
        "warmup_epochs":   40,
        "cycle_len":       50,
        "beta_max":        0.15,
        "batch_size":      512,
        "lr":              2e-4,
        "embed_dim":       128,
        "hidden":          512,
        "latent":          256,
        "max_len":         100,
        "n_aug_external":  1,
        "validity_check":  50,
        "device_pref":     "cuda",
    },
}


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda":
        if torch.cuda.is_available():
            log.info("Using GPU: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Using Apple MPS")
            return torch.device("mps")
        log.warning("GPU requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


def is_druglike(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return (150 <= Descriptors.MolWt(mol) <= 500
                and Descriptors.MolLogP(mol) <= 5
                and rdMolDescriptors.CalcNumHBD(mol) <= 5
                and rdMolDescriptors.CalcNumHBA(mol) <= 10)
    except Exception:
        return False


def load_zinc_subset(n_mols: int) -> List[str]:
    """Download and cache ZINC-250K drug-like subset."""
    cache_path = os.path.join(SAVE_DIR, f"zinc_cache_{n_mols}.txt")
    if os.path.exists(cache_path):
        log.info("Loading cached ZINC data from %s", cache_path)
        with open(cache_path) as f:
            return [line.strip() for line in f if line.strip()]

    log.info("Downloading ZINC-250K subset (%d molecules)...", n_mols)
    log.info("  Source: %s", ZINC_URL)
    try:
        req = urllib.request.Request(ZINC_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; molecular-design-vae)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            text = r.read().decode("utf-8")
    except Exception as e:
        log.error("Download failed: %s", e)
        log.error("Cannot proceed without internet on first run.")
        raise SystemExit(1)

    smiles_list: List[str] = []
    for line in text.strip().splitlines():
        smi = line.split(",")[0].split()[0].strip()
        if not smi or smi.lower() == "smiles":
            continue
        if is_druglike(smi):
            smiles_list.append(smi)
        if len(smiles_list) >= n_mols:
            break

    log.info("  Downloaded %d drug-like molecules", len(smiles_list))
    with open(cache_path, "w") as f:
        f.write("\n".join(smiles_list))
    log.info("  Cached to %s", cache_path)
    return smiles_list[:n_mols]


def kl_weight_for_epoch(epoch: int, cfg: dict) -> float:
    warmup, cycle, beta_max = cfg["warmup_epochs"], cfg["cycle_len"], cfg["beta_max"]
    if epoch < warmup:
        return beta_max * (epoch / warmup)
    t = (epoch - warmup) % cycle
    return beta_max * 0.5 * (1 - math.cos(math.pi * t / cycle))


def scheduled_tf_ratio(epoch: int, total: int) -> float:
    return max(0.5, 1.0 - 0.5 * (epoch / total))


def train(mode: str):
    cfg = MODES[mode]
    device = resolve_device(cfg["device_pref"])

    if cfg["device_pref"] == "cuda" and device.type == "cpu":
        cfg = {**cfg, "batch_size": min(cfg["batch_size"], 256)}
        log.info("Adjusted batch_size to %d for CPU fallback", cfg["batch_size"])

    log.info("=" * 60)
    log.info("Mode        : %s", mode)
    log.info("Description : %s", cfg["description"])
    log.info("Device      : %s", device)
    log.info("=" * 60)

    log.info("Step 1/7  Loading dataset...")
    external = load_zinc_subset(cfg["n_external"])
    seed_aug     = augment_smiles(SEED_SMILES, n_augment=4)
    external_aug = augment_smiles(external,    n_augment=cfg["n_aug_external"])
    all_smiles = seed_aug + external_aug
    random.shuffle(all_smiles)
    log.info("  Seeds (4x augmented) : %d", len(seed_aug))
    log.info("  External (%dx aug)   : %d", cfg["n_aug_external"], len(external_aug))
    log.info("  Total training set   : %d", len(all_smiles))

    log.info("Step 2/7  Building SELFIES tokenizer...")
    tokenizer = SELFIESTokenizer().fit(all_smiles)
    log.info("  Vocab size: %d", tokenizer.vocab_size)

    log.info("Step 3/7  Encoding to tensors...")
    encoded = []
    for smi in all_smiles:
        t = tokenizer.encode(smi, max_len=cfg["max_len"])
        if t is not None:
            encoded.append(t)
    log.info("  Encoded %d / %d successfully", len(encoded), len(all_smiles))
    if len(encoded) < 100:
        log.error("Too few encoded molecules")
        return

    log.info("Step 4/7  Building model...")
    vae = BetaVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden=cfg["hidden"],
        latent=cfg["latent"],
        max_len=cfg["max_len"],
    ).to(device)
    n_params = sum(p.numel() for p in vae.parameters())
    log.info("  Parameters: %d (%.1fM)", n_params, n_params / 1e6)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(100, cfg["total_epochs"] // 4), T_mult=1, eta_min=1e-6
    )

    log.info("Step 5/7  Training for %d epochs...", cfg["total_epochs"])
    best_validity = 0.0
    vae.train()

    for epoch in range(cfg["total_epochs"]):
        beta = kl_weight_for_epoch(epoch, cfg)
        tf   = scheduled_tf_ratio(epoch, cfg["total_epochs"])
        random.shuffle(encoded)
        epoch_loss = epoch_recon = 0.0
        n_batches = 0

        for i in range(0, len(encoded), cfg["batch_size"]):
            batch = encoded[i: i + cfg["batch_size"]]
            if not batch:
                continue
            x = torch.stack(batch).to(device)
            recon, mu, logvar, _ = vae(x, tf_ratio=tf)
            loss, r_loss, _ = vae.elbo_loss(recon, x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            epoch_loss  += loss.item()
            epoch_recon += r_loss
            n_batches   += 1

        scheduler.step()

        if (epoch + 1) % cfg["validity_check"] == 0:
            validity = check_validity(vae, tokenizer, n_samples=200)
            log.info("Epoch %4d | Loss=%.4f | Recon=%.4f | beta=%.4f | TF=%.2f | Validity=%.1f%%",
                     epoch + 1, epoch_loss/max(n_batches,1), epoch_recon/max(n_batches,1),
                     beta, tf, validity * 100)
            if validity > best_validity:
                best_validity = validity
                torch.save(vae.state_dict(), os.path.join(SAVE_DIR, "vae_best.pt"))
                log.info("  --> New best: %.1f%%", validity * 100)
        elif (epoch + 1) % 25 == 0:
            log.info("Epoch %4d | Loss=%.4f | beta=%.4f",
                     epoch + 1, epoch_loss/max(n_batches,1), beta)

    log.info("Step 6/7  Final evaluation...")
    vae.eval()
    final_validity = check_validity(vae, tokenizer, n_samples=500)
    log.info("Final validity : %.1f%%", final_validity * 100)
    log.info("Best validity  : %.1f%%", best_validity * 100)

    best_path = os.path.join(SAVE_DIR, "vae_best.pt")
    if os.path.exists(best_path) and best_validity > final_validity:
        vae.load_state_dict(torch.load(best_path, map_location=device))
        log.info("Loaded best weights")
        final_validity = best_validity

    log.info("Step 7/7  Saving model artifacts...")
    vae_cpu = vae.cpu()
    torch.save(vae_cpu.state_dict(), os.path.join(SAVE_DIR, "vae.pt"))
    with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    vae_cpu.eval()
    latents = {}
    with torch.no_grad():
        for smi in SEED_SMILES:
            ids = tokenizer.encode(smi, max_len=cfg["max_len"])
            if ids is None:
                continue
            mu, _ = vae_cpu.encoder(ids.unsqueeze(0))
            latents[smi] = mu.squeeze(0)
    torch.save(latents, os.path.join(SAVE_DIR, "latents.pt"))

    config_to_save = dict(cfg)
    config_to_save["mode"]            = mode
    config_to_save["vocab_size"]      = tokenizer.vocab_size
    config_to_save["n_seeds"]         = len(SEED_SMILES)
    config_to_save["n_external"]      = len(external)
    config_to_save["n_total_train"]   = len(encoded)
    config_to_save["final_validity"]  = float(final_validity)
    config_to_save["best_validity"]   = float(best_validity)
    config_to_save["seed_names"]      = SEED_NAMES
    config_to_save["all_seed_smiles"] = SEED_SMILES
    config_to_save["device_used"]     = str(device)

    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("Done. Files saved to %s/", SAVE_DIR)
    log.info("Final validity: %.1f%%", final_validity * 100)
    log.info("Now run:  python serve.py")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extended training modes for Molecular Design VAE"
    )
    parser.add_argument("--mode", required=True, choices=list(MODES.keys()),
                        help="Training mode: moderate, large, or large-gpu")
    args = parser.parse_args()
    train(args.mode)
