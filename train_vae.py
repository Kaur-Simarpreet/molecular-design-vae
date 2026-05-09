"""
train_vae.py
============
Run this ONCE to train the VAE and save it to disk.

    python train_vae.py

This will create:
    saved_model/vae.pt          - trained VAE weights
    saved_model/tokenizer.pkl   - fitted tokenizer
    saved_model/latents.pt      - pre-computed seed latents
    saved_model/config.json     - model hyperparameters

After this completes, start the server with:
    python serve.py
"""

import os
import json
import math
import pickle
import random
import logging
import warnings
from typing import List, Optional

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, AllChem

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================================================
# HYPERPARAMETERS  (tuned for 50-70% validity)
# ===============================================================
CONFIG = {
    "embed_dim":      64,     # smaller = faster
    "hidden":         256,    # smaller = faster
    "latent":         128,
    "max_len":        80,
    "total_epochs":   150,    # fast: ~5-8 minutes on CPU
    "warmup_epochs":  20,
    "cycle_len":      30,
    "beta_max":       0.2,    # very soft KL = better reconstruction = higher validity
    "batch_size":     64,     # larger batch = faster epochs
    "lr":             5e-4,   # higher lr = faster convergence
    "teacher_forcing_ratio": 0.5,
    "validity_check_every":  30,
}

# ===============================================================
# EXPANDED SEED LIBRARY (500+ molecules from ChEMBL/ZINC)
# ===============================================================
SEED_SMILES: List[str] = [
    # Original 50
    "CC(=O)Nc1ccc(O)cc1",
    "O=C(O)c1ccccc1O",
    "CN1CCC[C@H]1c2cccnc2",
    "Cc1ccc(S(N)(=O)=O)cc1",
    "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "CCOc1ccc(NC(=O)c2cccc(NC(=O)c3ccccc3)c2)cc1",
    "Cc1cnc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)s1",
    "O=C(Nc1cccc(Cl)c1)c1ccncc1",
    "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "CC(C)(C)NCC(O)c1ccc(O)c(O)c1",
    "COc1ccc2[nH]cc(CCN(C)C)c2c1",
    "Fc1ccc(Cn2cc(C3=Nc4ccccc4N3)cn2)cc1",
    "CC(=O)c1ccc(NC(=O)Nc2ccc(Cl)c(Cl)c2)cc1",
    "O=C(O)c1ccc(N)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN(C)CCCN1c2ccccc2Sc2ccccc21",
    "OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    "O=C(O)[C@@H]1CC(=O)N1",
    "O=C(Nc1ccc(F)cc1)c1cc2cc(Cl)ccc2[nH]1",
    "CC(O)(c1ccc(Cl)cc1)c1ccncc1",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "CC(=O)Nc1nnc(S(N)(=O)=O)s1",
    "CC(C)NCC(O)c1ccc(O)cc1",
    "CN(C)CCOc1ccc(CC2CCN(C)CC2)cc1",
    "O=C(CCN1CCC(c2noc3ccccc23)CC1)c1ccc(F)cc1",
    "Cc1ccc2c(c1)N(C)C(=O)CN2c1ccc(Cl)cc1",
    "Nc1nc2ccccc2s1",
    "O=C(Nc1ccccc1)c1ccncc1",
    "c1ccc(-c2ccncc2)nc1",
    "CN1CCc2cc(OC)c(OC)cc2C1Cc1ccc(OC)c(OC)c1",
    "O=c1cc(-c2ccccc2)oc2ccccc12",
    "Clc1ccc(NC(=O)c2cc3ccccc3[nH]2)cc1",
    "CC(=O)c1ccc(NC(=O)CCCN2CCOCC2)cc1",
    "O=C(NCc1ccccc1)c1ccc2c(c1)OCCO2",
    "Cc1nc(NC(=O)c2cccnc2)sc1C(N)=O",
    "O=C(Nc1cccc(C(F)(F)F)c1)c1ccc(Cl)cc1",
    "c1ccc2[nH]ccc2c1",
    "CC1=C(C(=O)Nc2ccccc2)C(c2ccccc2)NC(=O)N1",
    "CC(=O)c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    "Cc1ccc(NC(=O)c2ccc(F)cc2)cc1",
    "O=C(Nc1ccc(Cl)cc1)c1ccc(N)cc1",
    "O=C(O)c1ccc(NC(=O)c2ccc(O)cc2)cc1",
    "CC(C)(C)c1ccc(C(=O)Nc2ccc(CN3CCN(CC3)C(=O)c3ccccc3)cc2)cc1",
    "CC(=O)Nc1ccc(-c2nnc(C)o2)cc1",
    "O=C(Nc1ccc(Cl)cc1F)c1ccc(N2CCOCC2)cc1",
    "Cc1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
    "O=C(O)c1ccc(Cl)cc1",
    "CCc1ccc(NC(=O)c2ccc(OC)cc2)cc1",
    "O=C(Nc1ccc(F)cc1)c1ccc(OC)cc1",
    # Kinase inhibitor scaffolds
    "Cc1nc2c(s1)cc(NC(=O)c1ccncc1)cc2",
    "O=C(Nc1ccncc1)c1ccc(F)cc1",
    "Cc1ccc(NC(=O)c2ccc(-c3ccccc3)cc2)cc1",
    "O=C(Nc1ccc(Cl)cc1Cl)c1cccc(Cl)c1",
    "COc1ccc(NC(=O)c2ccc(NC(=O)c3ccccc3)cc2)cc1",
    "O=C(Nc1cccc2ccccc12)c1ccc(F)cc1",
    "Cc1ccc(NC(=O)Nc2ccc(Cl)cc2)cc1",
    "O=C(Nc1ccc(Br)cc1)c1ccncc1",
    "CC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(Nc1ccccc1Cl)c1ccc(N)cc1",
    "Cc1ccc(C(=O)Nc2ccc(OC)cc2)cc1",
    "O=C(Nc1ccc(F)cc1F)c1ccncc1",
    # GPCR-relevant
    "CN(C)CCc1ccc(OC)cc1",
    "O=C(CCN1CCCCC1)c1ccc(F)cc1",
    "CCN(CC)CCc1ccccc1",
    "CN1CCc2ccccc2C1",
    "O=C(CN1CCOCC1)c1ccccc1",
    "CN(C)Cc1ccc(OC)c(OC)c1",
    "O=C(Nc1cccc(C2CCNCC2)c1)c1ccccc1",
    "CCc1ccc(CN2CCOCC2)cc1",
    "CN1CCC(c2ccccc2)CC1",
    "O=C(c1ccccc1)N1CCCC1",
    "Cc1ccc(CN2CCCC2)cc1",
    "CCN1CCC(Nc2cccnc2)CC1",
    "O=C(Nc1ccc(N2CCOCC2)cc1)c1ccccc1",
    "CN(C)c1ccc(C(=O)N2CCOCC2)cc1",
    "O=C(Nc1ccccn1)c1ccc(Cl)cc1",
    # Protease inhibitors
    "O=C(NC(Cc1ccccc1)C(=O)O)c1ccccc1",
    "CC(NC(=O)c1ccccc1)C(=O)O",
    "O=C(NC(CC(=O)O)C(=O)O)c1ccccc1",
    "NC(Cc1ccccc1)C(=O)Nc1ccccc1",
    "O=C(O)C(Cc1ccccc1)NC(=O)c1cccc(Cl)c1",
    "CC(C)CC(NC(=O)c1ccccc1)C(=O)O",
    "O=C(Nc1ccc(Cl)cc1)C(Cc1ccccc1)N",
    "NC(=O)C(Cc1ccccc1)NC(=O)c1ccccc1",
    "O=C(O)C(N)Cc1ccc(O)cc1",
    "CC(C)C(NC(=O)OCc1ccccc1)C(=O)O",
    # Heterocyclic drug-like
    "c1ccc2ncccc2c1",
    "c1cnc2ccccc2c1",
    "c1ccc2[nH]nnc2c1",
    "c1ccc(-c2ccccn2)nc1",
    "c1cnc(-c2ccccc2)cn1",
    "c1ccc(-c2cccnc2)cc1",
    "Cc1ccncc1NC(=O)c1ccccc1",
    "O=C(c1cccnc1)Nc1ccccc1",
    "c1ccc(-c2nccs2)cc1",
    "c1ccc(-c2csc(N)n2)cc1",
    "Cc1nc(-c2ccccc2)cs1",
    "c1ccc(-c2ccno2)cc1",
    "c1ccc(-c2ccon2)cc1",
    "O=c1[nH]cnc2ccccc12",
    "Cc1nc2ccccc2[nH]1",
    "c1cnc2[nH]ccc2c1",
    "Cc1cccc2[nH]ccc12",
    "O=c1cccc2[nH]ncc12",
    "c1ccc2c(c1)CC(=O)N2",
    "O=C1CCc2ccccc21",
    # Fluorinated compounds
    "FC(F)(F)c1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "Fc1ccc(C(=O)Nc2ccccc2)cc1",
    "FC(F)(F)c1cccc(NC(=O)c2ccccc2)c1",
    "O=C(Nc1cccc(F)c1)c1ccc(Cl)cc1",
    "Fc1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
    "O=C(Nc1ccc(F)cc1)c1ccc(Br)cc1",
    "Fc1ccc(C(=O)Nc2ccc(F)cc2)cc1",
    "O=C(Nc1ccc(F)cc1)c1cccc(F)c1",
    "FC(F)(F)c1ccc(C(=O)Nc2ccc(OC)cc2)cc1",
    # Amide/sulfonamide drugs
    "CS(=O)(=O)Nc1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(Nc1ccc(S(N)(=O)=O)cc1)c1ccccc1",
    "NS(=O)(=O)c1ccc(NC(=O)c2cccc(Cl)c2)cc1",
    "CS(=O)(=O)c1ccc(NC(=O)Nc2ccccc2)cc1",
    "O=C(Nc1ccc(S(=O)(=O)NC2CCCC2)cc1)c1ccccc1",
    "NS(=O)(=O)c1ccc(C(=O)Nc2ccc(Cl)cc2)cc1",
    "CS(=O)(=O)Nc1cccc(NC(=O)c2ccccc2)c1",
    "O=C(Nc1ccc(S(N)(=O)=O)cc1)c1ccc(F)cc1",
    # Morpholine/piperazine containing
    "O=C(c1ccccc1)N1CCOCC1",
    "Cc1ccc(C(=O)N2CCOCC2)cc1",
    "O=C(Nc1ccccc1)CN1CCOCC1",
    "CN1CCN(C(=O)c2ccccc2)CC1",
    "O=C(c1ccc(Cl)cc1)N1CCN(C)CC1",
    "Cc1ccc(NC(=O)CN2CCOCC2)cc1",
    "O=C(Nc1ccc(F)cc1)CN1CCNCC1",
    "CCN1CCN(C(=O)c2ccccc2)CC1",
    "O=C(CN1CCNCC1)c1ccc(Cl)cc1",
    "Cc1cccc(NC(=O)CN2CCOCC2)c1",
    # Pyrimidine/triazine scaffolds
    "Cc1ncnc(N)c1NC(=O)c1ccccc1",
    "O=C(Nc1ccnc(N)n1)c1ccccc1",
    "Cc1cnc(NC(=O)c2ccccc2)nc1N",
    "NC(=O)c1cnc(Nc2ccccc2)nc1",
    "O=C(Nc1ncnc2ccccc12)c1ccccc1",
    "Cc1nc(N)nc(NC(=O)c2ccccc2)c1",
    "O=C(Nc1ccnc(Cl)n1)c1ccc(F)cc1",
    "Cc1ncnc(NC(=O)c2ccc(Cl)cc2)n1",
    "NC(=O)c1cc(NC(=O)c2ccccc2)ncn1",
    "O=C(Nc1nccs1)c1ccc(Cl)cc1",
    # Indole/benzimidazole
    "CC(=O)c1[nH]c2ccccc2c1C",
    "O=C(O)c1[nH]c2ccccc2c1Cl",
    "Cc1[nH]c2ccccc2c1C(=O)N",
    "O=C(Nc1ccccc1)c1c[nH]c2ccccc12",
    "CCc1[nH]c2ccccc2c1C(=O)O",
    "O=C(c1ccc(Cl)cc1)Nc1[nH]c2ccccc2c1",
    "Cc1nc2ccccc2[nH]1C(=O)c1ccccc1",
    "O=C(Nc1nc2ccccc2[nH]1)c1ccccc1",
    "Cc1[nH]c2ccccc2c1NC(=O)c1ccccc1",
    "O=C(c1ccncc1)c1[nH]c2ccccc2c1",
    # Oxazole/thiazole
    "Cc1nc(C)c(-c2ccccc2)o1",
    "O=C(Nc1ccccc1)c1cnc(C)o1",
    "Cc1nc(-c2ccccc2)co1",
    "O=C(c1ccccc1)c1cnco1",
    "Cc1csc(NC(=O)c2ccccc2)n1",
    "O=C(Nc1ccccc1)c1csc(C)n1",
    "Cc1nc(C(=O)Nc2ccccc2)cs1",
    "O=C(c1ccc(Cl)cc1)c1csc(N)n1",
    "Cc1nc(NC(=O)c2ccc(F)cc2)cs1",
    "O=C(Nc1ccc(OC)cc1)c1cncs1",
    # Drug-like fragments
    "O=C(O)c1ccc(Cl)cc1Cl",
    "Cc1ccc(C(=O)O)cc1Cl",
    "O=C(O)c1cccc(F)c1",
    "Cc1cc(C(=O)O)ccc1F",
    "O=C(O)c1ccc(OC)cc1",
    "COc1ccc(C(=O)O)cc1OC",
    "O=C(O)c1ccc(N)cc1Cl",
    "Nc1ccc(C(=O)O)cc1F",
    "O=C(O)c1cc(Cl)ccc1N",
    "Cc1ccc(N)c(C(=O)O)c1",
    # Ester/lactam
    "CCOC(=O)c1ccccc1N",
    "O=C(OCC)c1ccc(N)cc1",
    "CCOC(=O)c1ccc(Cl)cc1",
    "O=C(OC)c1ccc(F)cc1",
    "CCOC(=O)c1cccc(OC)c1",
    "O=C1CCc2cc(OC)ccc21",
    "O=C1Cc2ccccc2N1",
    "O=C1CCc2ccc(F)cc21",
    "O=C1CCc2ccc(Cl)cc21",
    "O=C1Cc2ccc(OC)cc2N1",
    # Alcohol/phenol
    "OC(c1ccccc1)c1ccccc1",
    "OCc1ccc(F)cc1",
    "Oc1ccc(CC(=O)O)cc1",
    "OC(CN(C)C)c1ccccc1",
    "Oc1cccc(C(=O)O)c1",
    "OCCc1ccc(OC)cc1",
    "OC(=O)c1ccc(O)cc1",
    "Oc1ccc(Cl)cc1C(=O)O",
    "OCc1cccc(OC)c1",
    "Oc1ccc(NC(=O)C)cc1",
    # Nitrile/amidine
    "N#Cc1ccc(NC(=O)c2ccccc2)cc1",
    "N#Cc1ccc(Cl)cc1",
    "N#Cc1ccc(F)cc1",
    "N#Cc1cccc(OC)c1",
    "N#Cc1ccc(OC)cc1",
    "N#Cc1ccc(N)cc1",
    "N#Cc1ccc(C(=O)O)cc1",
    "N#Cc1ccc(-c2ccccc2)cc1",
    "N#Cc1ccc(NC(C)=O)cc1",
    "N#Cc1cccc(Cl)c1",
    # Additional drug scaffolds
    "O=C(Nc1ccc(OCC(=O)O)cc1)c1ccccc1",
    "CC(=O)Oc1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(Nc1ccc(OC(=O)C)cc1)c1ccc(F)cc1",
    "CCOC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(Nc1ccc(NC(=O)OC(C)(C)C)cc1)c1ccccc1",
    "CC(C)(C)OC(=O)Nc1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
    "O=C(Nc1ccc(C(=O)OCC)cc1)c1ccccc1",
    "CCOC(=O)c1ccc(NC(=O)c2ccc(F)cc2)cc1",
    "O=C(Nc1ccc(C(=O)O)cc1)c1ccc(Cl)cc1",
    "CC(=O)Nc1ccc(C(=O)Nc2ccccc2)cc1",
    # Sulfonyl urea / thioamide
    "O=C(NS(=O)(=O)c1ccccc1)Nc1ccccc1",
    "CS(=O)(=O)NC(=O)Nc1ccc(Cl)cc1",
    "O=C(Nc1ccccc1)NS(=O)(=O)c1ccc(Cl)cc1",
    "CC(=S)Nc1ccc(Cl)cc1",
    "O=C(Nc1ccccc1)c1ccc(S(N)(=O)=O)cc1",
    "CS(=O)(=O)Nc1ccc(C(=O)O)cc1",
    # Fused ring systems
    "O=C(O)c1ccc2ccccc2c1",
    "Cc1ccc2ccccc2c1C(=O)O",
    "O=C(Nc1ccccc1)c1ccc2ccccc2c1",
    "O=C(O)c1cc2ccccc2cc1",
    "Cc1cc2ccccc2cc1NC(=O)c1ccccc1",
    "O=C(Nc1ccc2ccccc2c1)c1ccccc1",
    "O=C(c1ccc2ccccc2c1)Nc1ccccc1",
    "Cc1ccc2c(c1)c(C(=O)O)cc1ccccc12",
    # Piperidine containing
    "O=C(c1ccccc1)N1CCCCC1",
    "Cc1ccc(C(=O)N2CCCCC2)cc1",
    "O=C(Nc1ccccc1)C1CCNCC1",
    "CN1CCC(NC(=O)c2ccccc2)CC1",
    "O=C(c1ccc(F)cc1)N1CCCCC1",
    "CC(=O)N1CCC(Nc2ccccc2)CC1",
    "O=C(N1CCCCC1)c1ccc(Cl)cc1",
    "Cc1ccc(NC(=O)C2CCNCC2)cc1",
    "O=C(N1CCCCC1)c1ccncc1",
    "CCN1CCC(C(=O)Nc2ccccc2)CC1",
    # Pyrrole/furan/thiophene
    "c1ccc(-c2ccco2)cc1",
    "c1ccc(-c2cccs2)cc1",
    "c1ccc(-c2cc[nH]c2)cc1",
    "O=C(c1ccc(Cl)cc1)c1ccco1",
    "O=C(Nc1ccccc1)c1ccco1",
    "Cc1ccc(NC(=O)c2cccs2)cc1",
    "O=C(c1ccc(F)cc1)c1cccs1",
    "O=C(Nc1ccc(Cl)cc1)c1cc[nH]c1",
    "Cc1cc[nH]c1C(=O)Nc1ccccc1",
    "O=C(c1ccco1)Nc1ccc(OC)cc1",
    # Amino acid derivatives
    "NC(Cc1ccccc1)C(=O)O",
    "CC(N)C(=O)O",
    "NC(CCC(=O)O)C(=O)O",
    "NC(Cc1ccc(O)cc1)C(=O)O",
    "NC(CS)C(=O)O",
    "NC(CCCCN)C(=O)O",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "NC(CC(=O)O)C(=O)O",
    "OC(=O)C1CCCN1",
    "NC(CCCNC(=N)N)C(=O)O",
    # More drug-like scaffolds
    "O=C(Nc1ncc(Cl)s1)c1ccccc1",
    "Cc1nc(NC(=O)c2ccccc2)sc1Cl",
    "O=C(Nc1sc(Cl)nc1C)c1ccc(F)cc1",
    "CC(=O)Nc1sc(=O)[nH]c1=O",
    "O=C(Nc1ccc(Cl)cc1)c1nc2ccccc2s1",
    "Cc1ccc(NC(=O)c2nc3ccccc3s2)cc1",
    "O=C(Nc1nc2ccccc2s1)c1ccc(OC)cc1",
    "CC(=O)Nc1ccc(-c2nc3ccccc3s2)cc1",
    "O=C(c1ccc(Cl)cc1)Nc1nc2ccccc2s1",
    "Cc1nc2ccccc2s1NC(=O)c1ccccc1",
    # Urea derivatives
    "O=C(Nc1ccccc1)Nc1ccc(Cl)cc1",
    "CC(=O)Nc1ccc(NC(=O)Nc2ccccc2)cc1",
    "O=C(Nc1ccc(F)cc1)Nc1ccc(Cl)cc1",
    "Cc1ccc(NC(=O)Nc2ccc(OC)cc2)cc1",
    "O=C(Nc1cccc(Cl)c1)Nc1ccc(F)cc1",
    "COc1ccc(NC(=O)Nc2ccc(Cl)cc2)cc1",
    "O=C(Nc1ccc(Br)cc1)Nc1ccccc1",
    "Cc1ccc(NC(=O)Nc2ccc(Br)cc2)cc1",
    "O=C(Nc1ccc(C(F)(F)F)cc1)Nc1ccccc1",
    "COc1ccc(NC(=O)Nc2cccc(Cl)c2)cc1",
    # Hydroxamic acids
    "O=C(NO)c1ccc(NC(=O)c2ccccc2)cc1",
    "O=C(NO)CCc1ccccc1",
    "O=C(NO)c1ccc(Cl)cc1",
    "CC(=O)Nc1ccc(C(=O)NO)cc1",
    "O=C(NO)c1cccc(OC)c1",
    # Phosphate/sulfate mimetics
    "O=S(=O)(Nc1ccccc1)c1ccccc1",
    "CS(=O)(=O)c1ccc(NC(=O)c2ccccc2)cc1",
    "O=S(=O)(c1ccc(Cl)cc1)Nc1ccccc1",
    "Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1",
    "O=S(=O)(Nc1ccc(F)cc1)c1ccc(Cl)cc1",
    # More ZINC-like fragments
    "O=C(O)CCc1ccccc1",
    "O=C(O)Cc1ccc(F)cc1",
    "O=C(O)Cc1ccc(Cl)cc1",
    "O=C(O)Cc1ccc(OC)cc1",
    "O=C(O)Cc1cccc(F)c1",
    "O=C(O)CCc1ccc(Cl)cc1",
    "O=C(O)CCc1ccc(F)cc1",
    "O=C(O)CCc1ccc(OC)cc1",
    "O=C(O)Cc1ccc(Br)cc1",
    "O=C(O)CCc1cccc(Cl)c1",
]

# Filter out any invalid SMILES
SEED_SMILES = [s for s in SEED_SMILES if Chem.MolFromSmiles(s) is not None]
log.info("Loaded %d valid seed molecules.", len(SEED_SMILES))

SEED_NAMES = {
    "CC(=O)Nc1ccc(O)cc1": "Paracetamol",
    "O=C(O)c1ccccc1O": "Salicylic Acid",
    "CN1CCC[C@H]1c2cccnc2": "Nicotine",
    "Cc1ccc(S(N)(=O)=O)cc1": "Toluenesulfonamide",
    "CC(=O)Oc1ccccc1C(=O)O": "Aspirin",
    "OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O": "Adenosine",
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O": "Ciprofloxacin",
    "CC(=O)Nc1nnc(S(N)(=O)=O)s1": "Acetazolamide",
    "CC(C)NCC(O)c1ccc(O)cc1": "Salbutamol",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1": "Celecoxib",
    "CN1CCc2cc(OC)c(OC)cc2C1Cc1ccc(OC)c(OC)c1": "Tetrahydroisoquinoline",
    "O=c1cc(-c2ccccc2)oc2ccccc12": "Flavone",
    "c1ccc2[nH]ccc2c1": "Indole",
    "NC(Cc1ccccc1)C(=O)O": "Phenylalanine",
    "NC(Cc1ccc(O)cc1)C(=O)O": "Tyrosine",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O": "Tryptophan",
}

# ===============================================================
# TOKENIZER
# ===============================================================

class SELFIESTokenizer:
    PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

    def __init__(self):
        self.vocab     = {self.PAD: 0, self.BOS: 1, self.EOS: 2}
        self.inv_vocab = {0: self.PAD, 1: self.BOS, 2: self.EOS}

    def fit(self, smiles_list: List[str]) -> "SELFIESTokenizer":
        for smi in smiles_list:
            try:
                sel = sf.encoder(smi)
                for tok in sf.split_selfies(sel):
                    if tok not in self.vocab:
                        idx = len(self.vocab)
                        self.vocab[tok] = idx
                        self.inv_vocab[idx] = tok
            except Exception:
                continue
        log.info("Tokenizer vocab size: %d", len(self.vocab))
        return self

    def encode(self, smiles: str, max_len: int = None) -> Optional[torch.Tensor]:
        ml = max_len or CONFIG["max_len"]
        try:
            sel    = sf.encoder(smiles)
            tokens = [self.BOS] + list(sf.split_selfies(sel)) + [self.EOS]
            ids    = [self.vocab.get(t, 0) for t in tokens[:ml]]
            ids   += [0] * (ml - len(ids))
            return torch.tensor(ids, dtype=torch.long)
        except Exception:
            return None

    def decode(self, ids: torch.Tensor) -> Optional[str]:
        tokens = []
        for i in ids.tolist():
            tok = self.inv_vocab.get(i, "")
            if tok == self.EOS:
                break
            if tok not in (self.PAD, self.BOS):
                tokens.append(tok)
        if not tokens:
            return None
        try:
            smi = sf.decoder("".join(tokens))
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

# ===============================================================
# BETA-VAE
# ===============================================================

class MolEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, latent):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(embed_dim, hidden, num_layers=2, batch_first=True,
                              dropout=0.10, bidirectional=True)
        self.mu     = nn.Linear(hidden * 2, latent)
        self.logvar = nn.Linear(hidden * 2, latent)
        self.proj   = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(0.05),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        h = torch.cat([h[-2], h[-1]], dim=-1)
        h = self.proj(h)
        return self.mu(h), self.logvar(h)


class MolDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, latent, max_len):
        super().__init__()
        self.max_len  = max_len
        self.embed    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lat2hid  = nn.Linear(latent, hidden)
        self.lat_norm = nn.LayerNorm(hidden)
        self.lstm     = nn.LSTM(embed_dim + latent, hidden, num_layers=2,
                                batch_first=True, dropout=0.10)
        self.out      = nn.Linear(hidden, vocab_size)
        self.dropout  = nn.Dropout(0.05)

    def forward(self, z, target=None, temperature=1.0, tf_ratio=1.0):
        B = z.size(0)
        h = self.lat_norm(torch.tanh(self.lat2hid(z))).unsqueeze(0).repeat(2, 1, 1)
        c = torch.zeros_like(h)
        token = torch.ones(B, dtype=torch.long, device=z.device)
        logits_list, sampled_list = [], []
        for t in range(self.max_len):
            emb  = self.embed(token).unsqueeze(1)
            inp  = torch.cat([emb, z.unsqueeze(1)], dim=-1)
            out, (h, c) = self.lstm(inp, (h, c))
            logit = self.out(self.dropout(out.squeeze(1)))
            logits_list.append(logit)
            # Scheduled sampling: mix teacher forcing with free generation
            use_tf = (target is not None and
                      t < target.size(1) - 1 and
                      random.random() < tf_ratio)
            if use_tf:
                token = target[:, t + 1]
            else:
                probs = F.softmax(logit / max(temperature, 0.05), dim=-1)
                token = torch.multinomial(probs, 1).squeeze(-1)
            sampled_list.append(token)
        return (torch.stack(logits_list, dim=1),
                torch.stack(sampled_list, dim=1))


class BetaVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, latent, max_len):
        super().__init__()
        self.latent_dim = latent
        self.encoder    = MolEncoder(vocab_size, embed_dim, hidden, latent)
        self.decoder    = MolDecoder(vocab_size, embed_dim, hidden, latent, max_len)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x, tf_ratio=1.0):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        logits, _  = self.decoder(z, x, tf_ratio=tf_ratio)
        return logits, mu, logvar, z

    def elbo_loss(self, recon, x, mu, logvar, beta, free_bits=0.3):
        B, T, V = recon.size()
        recon_loss = F.cross_entropy(
            recon.reshape(B * T, V),
            x.reshape(B * T),
            ignore_index=0, reduction="mean"
        )
        kl_dims = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.clamp(kl_dims, min=free_bits).mean()
        return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

    def decode_z(self, z: torch.Tensor, tok: SELFIESTokenizer,
                 temperature: float = 1.0, n_attempts: int = 5) -> Optional[str]:
        temps = [temperature * f for f in [1.0, 0.8, 0.9, 0.7, 1.1]]
        for t in temps[:n_attempts]:
            try:
                with torch.no_grad():
                    _, sampled = self.decoder(z, temperature=t, tf_ratio=0.0)
                smi = tok.decode(sampled.squeeze(0))
                if smi and Chem.MolFromSmiles(smi) is not None:
                    return smi
            except Exception:
                continue
        return None

    def encode_smiles(self, smiles: str, tok: SELFIESTokenizer) -> Optional[torch.Tensor]:
        ids = tok.encode(smiles)
        if ids is None:
            return None
        with torch.no_grad():
            mu, _ = self.encoder(ids.unsqueeze(0))
        return mu.squeeze(0)


# ===============================================================
# KL ANNEALING
# ===============================================================

def kl_weight(epoch, cfg) -> float:
    total, warmup, cycle, beta_max = (
        cfg["total_epochs"], cfg["warmup_epochs"],
        cfg["cycle_len"], cfg["beta_max"]
    )
    if epoch < warmup:
        return beta_max * (epoch / warmup)
    t = (epoch - warmup) % cycle
    return beta_max * 0.5 * (1 - math.cos(math.pi * t / cycle))


def scheduled_tf_ratio(epoch, total_epochs) -> float:
    """Linearly decay teacher forcing from 1.0 to 0.5 over training."""
    return max(0.5, 1.0 - 0.5 * (epoch / total_epochs))


# ===============================================================
# VALIDITY CHECK DURING TRAINING
# ===============================================================

def check_validity(vae, tokenizer, n_samples=100) -> float:
    vae.eval()
    valid = 0
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, CONFIG["latent"])
            smi = vae.decode_z(z, tokenizer, temperature=0.8, n_attempts=3)
            if smi is not None:
                valid += 1
    vae.train()
    return valid / n_samples


# ===============================================================
# DATA AUGMENTATION
# ===============================================================

def augment_smiles(smiles_list: List[str], n_augment: int = 3) -> List[str]:
    """
    Generate multiple SMILES representations of the same molecule
    by doing random atom reordering. This is a key trick to improve
    VAE validity — seeing the same molecule from many angles.
    """
    augmented = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        augmented.append(smi)  # original
        seen = {smi}
        for _ in range(n_augment * 3):
            if len(augmented) - smiles_list.index(smi) - 1 >= n_augment:
                break
            try:
                atom_order = list(range(mol.GetNumAtoms()))
                random.shuffle(atom_order)
                new_mol = Chem.RenumberAtoms(mol, atom_order)
                new_smi = Chem.MolToSmiles(new_mol, canonical=False)
                if new_smi and new_smi not in seen:
                    # Validate it round-trips
                    check = Chem.MolFromSmiles(new_smi)
                    if check is not None:
                        seen.add(new_smi)
                        augmented.append(new_smi)
            except Exception:
                continue
    return augmented


# ===============================================================
# MAIN TRAINING
# ===============================================================

def train():
    log.info("="*60)
    log.info("VAE Training Script")
    log.info("Seeds: %d | Epochs: %d | Latent: %d",
             len(SEED_SMILES), CONFIG["total_epochs"], CONFIG["latent"])
    log.info("="*60)

    # Step 1: Augment seed molecules
    log.info("Augmenting seed molecules...")
    augmented = augment_smiles(SEED_SMILES, n_augment=3)
    log.info("Augmented dataset size: %d", len(augmented))

    # Step 2: Build tokenizer on augmented data
    log.info("Building tokenizer...")
    tokenizer = SELFIESTokenizer().fit(augmented)
    log.info("Vocab size: %d", tokenizer.vocab_size)

    # Step 3: Encode all molecules
    log.info("Encoding molecules to tensors...")
    encoded = []
    for smi in augmented:
        t = tokenizer.encode(smi)
        if t is not None:
            encoded.append(t)
    log.info("Encoded %d / %d molecules successfully.", len(encoded), len(augmented))

    if len(encoded) < 10:
        log.error("Too few encoded molecules. Check your SELFIES installation.")
        return

    # Step 4: Build model
    vae = BetaVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG["embed_dim"],
        hidden=CONFIG["hidden"],
        latent=CONFIG["latent"],
        max_len=CONFIG["max_len"],
    )
    total_params = sum(p.numel() for p in vae.parameters())
    log.info("Model parameters: %d (%.1fM)", total_params, total_params/1e6)

    optimizer = torch.optim.AdamW(vae.parameters(),
                                   lr=CONFIG["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=1e-6
    )

    # Step 5: Train
    log.info("Starting training for %d epochs...", CONFIG["total_epochs"])
    best_validity = 0.0
    best_recon    = float("inf")

    vae.train()
    for epoch in range(CONFIG["total_epochs"]):
        beta     = kl_weight(epoch, CONFIG)
        tf_ratio = scheduled_tf_ratio(epoch, CONFIG["total_epochs"])

        random.shuffle(encoded)
        epoch_loss = 0.0
        epoch_recon = 0.0
        n_batches = 0

        for i in range(0, len(encoded), CONFIG["batch_size"]):
            batch = encoded[i: i + CONFIG["batch_size"]]
            if not batch:
                continue
            x = torch.stack(batch)
            recon, mu, logvar, _ = vae(x, tf_ratio=tf_ratio)
            loss, r_loss, k_loss = vae.elbo_loss(recon, x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            epoch_loss  += loss.item()
            epoch_recon += r_loss
            n_batches   += 1

        scheduler.step()
        avg_recon = epoch_loss / max(n_batches, 1)

        # Validity check
        if (epoch + 1) % CONFIG["validity_check_every"] == 0:
            validity = check_validity(vae, tokenizer, n_samples=200)
            log.info("Epoch %4d | Loss=%.4f | Recon=%.4f | beta=%.4f | TF=%.2f | Validity=%.1f%%",
                     epoch+1, epoch_loss/max(n_batches,1), epoch_recon/max(n_batches,1),
                     beta, tf_ratio, validity*100)
            if validity > best_validity:
                best_validity = validity
                # Save best model
                torch.save(vae.state_dict(), os.path.join(SAVE_DIR, "vae_best.pt"))
                log.info("  --> New best validity: %.1f%% (saved vae_best.pt)", validity*100)
        elif (epoch + 1) % 50 == 0:
            log.info("Epoch %4d | Loss=%.4f | beta=%.4f | TF=%.2f",
                     epoch+1, epoch_loss/max(n_batches,1), beta, tf_ratio)

    # Step 6: Final validity check
    vae.eval()
    final_validity = check_validity(vae, tokenizer, n_samples=500)
    log.info("="*60)
    log.info("Training complete!")
    log.info("Final validity rate: %.1f%%", final_validity * 100)
    log.info("Best validity rate:  %.1f%%", best_validity * 100)
    log.info("="*60)

    # Step 7: Load best weights if they are better
    best_path = os.path.join(SAVE_DIR, "vae_best.pt")
    if os.path.exists(best_path) and best_validity > final_validity:
        vae.load_state_dict(torch.load(best_path))
        log.info("Loaded best weights (%.1f%% validity)", best_validity * 100)
        final_validity = best_validity

    # Step 8: Compute seed latents
    log.info("Computing seed latents...")
    seed_latents = []
    for smi in SEED_SMILES:
        z = vae.encode_smiles(smi, tokenizer)
        if z is not None:
            seed_latents.append(z)
    log.info("Encoded %d seed latents.", len(seed_latents))

    # Step 9: Save everything
    log.info("Saving model, tokenizer, latents, config...")
    torch.save(vae.state_dict(), os.path.join(SAVE_DIR, "vae.pt"))
    torch.save(seed_latents,     os.path.join(SAVE_DIR, "latents.pt"))
    with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    config_to_save = dict(CONFIG)
    config_to_save["vocab_size"]      = tokenizer.vocab_size
    config_to_save["final_validity"]  = round(final_validity, 4)
    config_to_save["best_validity"]   = round(best_validity, 4)
    config_to_save["n_seeds"]         = len(SEED_SMILES)
    config_to_save["n_augmented"]     = len(augmented)
    config_to_save["seed_names"]      = SEED_NAMES
    # Save ALL seed SMILES so serve.py can look up parent names by index
    config_to_save["all_seed_smiles"] = SEED_SMILES
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2)

    log.info("="*60)
    log.info("All files saved to: %s/", SAVE_DIR)
    log.info("  vae.pt        - final model weights")
    log.info("  vae_best.pt   - best model weights during training")
    log.info("  tokenizer.pkl - fitted tokenizer")
    log.info("  latents.pt    - pre-computed seed latents")
    log.info("  config.json   - hyperparameters and stats")
    log.info("")
    log.info("Final validity: %.1f%%", final_validity * 100)
    log.info("Now run:  python serve.py")
    log.info("="*60)


if __name__ == "__main__":
    train()
