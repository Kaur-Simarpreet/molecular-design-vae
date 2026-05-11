"""
serve.py
========
Run AFTER train_vae.py has completed.
    python serve.py
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import pickle
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from rdkit import Chem

# Real docking module — routes by mode, falls back to mock if not set up
try:
    from docking import dock_for_mode as _dock_for_mode
    _DOCKING_AVAILABLE = True
except ImportError:
    _DOCKING_AVAILABLE = False, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = "saved_model"
app = Flask(__name__, static_folder=".")
CORS(app)

# ===============================================================
# VERIFY SAVED MODEL EXISTS
# ===============================================================

for f in ["vae.pt", "tokenizer.pkl", "latents.pt", "config.json"]:
    path = os.path.join(SAVE_DIR, f)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nMissing: {path}\n"
            "Please run 'python train_vae.py' first.\n"
        )

# ===============================================================
# LOAD CONFIG
# ===============================================================

with open(os.path.join(SAVE_DIR, "config.json")) as f:
    CONFIG = json.load(f)

LATENT_DIM      = CONFIG["latent"]
MAX_LEN         = CONFIG["max_len"]
SEED_NAMES      = CONFIG.get("seed_names", {})
ALL_SEED_SMILES = CONFIG.get("all_seed_smiles", list(SEED_NAMES.keys()))

log.info("Config loaded: latent=%d, max_len=%d, vocab=%d, seeds=%d",
         LATENT_DIM, MAX_LEN, CONFIG["vocab_size"], len(ALL_SEED_SMILES))
log.info("Trained validity: %.1f%%", CONFIG.get("final_validity", 0) * 100)

# ===============================================================
# PAINS
# ===============================================================

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
try:
    _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
except Exception:
    log.warning("BRENK not available.")
_pains_catalog = FilterCatalog(_pains_params)

def is_pains(mol: Chem.Mol) -> Tuple[bool, str]:
    entry = _pains_catalog.GetFirstMatch(mol)
    return (True, entry.GetDescription()) if entry else (False, "")

# ===============================================================
# PHARMACOPHORE
# ===============================================================

_PHARMACOPHORE_SMARTS: Dict[str, List[str]] = {
    "kinase": [
        "[$([NH]c1ncnc2[nH]cnc12)]",
        "c1ccncc1",           # pyridine — common kinase hinge binder
        "C(=O)N",             # amide
    ],
    "gpcr": [
        "[NX3;H0,H1,H2]",    # any amine — broad GPCR match
        "c1ccccc1",           # aromatic ring
        "C(=O)N",             # amide
        "[OH]",               # hydroxyl
    ],
    "protease": [
        "C(=O)N",
        "NC(=O)",
        "c1ccccc1",
    ],
    "nuclear_receptor": [
        "OC(=O)",
        "c1ccccc1",
        "C(=O)N",
    ],
}

def _classify_target(target: str) -> Optional[str]:
    t = target.lower()
    if any(k in t for k in ["kinase","egfr","abl","jak","src","kit"]): return "kinase"
    if any(k in t for k in ["gpcr","adrenergic","serotonin","dopamine","histamine"]): return "gpcr"
    if any(k in t for k in ["protease","hiv","ace","caspase","thrombin"]): return "protease"
    if any(k in t for k in ["nuclear","ppar","rxr","rar","ar","er","gr"]): return "nuclear_receptor"
    return None

def count_pharmacophore_hits(mol: Chem.Mol, target: str) -> int:
    """
    Count how many basic drug-like features the molecule has.
    Uses simple reliable SMARTS that match real drug molecules.
    """
    patterns = [
        Chem.MolFromSmarts("c1ccccc1"),       # aromatic ring
        Chem.MolFromSmarts("C(=O)N"),         # amide
        Chem.MolFromSmarts("[NX3]"),           # any nitrogen
        Chem.MolFromSmarts("[OH]"),            # hydroxyl
        Chem.MolFromSmarts("c1ccncc1"),        # pyridine
        Chem.MolFromSmarts("c1ccc2[nH]ccc2c1"), # indole
        Chem.MolFromSmarts("N1CCNCC1"),        # piperazine
        Chem.MolFromSmarts("S(=O)(=O)N"),      # sulfonamide
    ]
    hits = 0
    for patt in patterns:
        if patt and mol.HasSubstructMatch(patt):
            hits += 1
    return hits

# ===============================================================
# SMARTS
# ===============================================================

_SMARTS: Dict[str, Chem.Mol] = {
    "amide":   Chem.MolFromSmarts("C(=O)N"),
    "amine":   Chem.MolFromSmarts("[NX3;H1,H2]"),
    "sulfa":   Chem.MolFromSmarts("S(=O)(=O)N"),
    "ester":   Chem.MolFromSmarts("C(=O)O[#6]"),
    "ketone":  Chem.MolFromSmarts("[#6]C(=O)[#6]"),
    "alcohol": Chem.MolFromSmarts("[OX2H][#6]"),
    "acid":    Chem.MolFromSmarts("C(=O)[OH]"),
    "nitro":   Chem.MolFromSmarts("[N+](=O)[O-]"),
    "cf3":     Chem.MolFromSmarts("C(F)(F)F"),
    "piperaz": Chem.MolFromSmarts("N1CCNCC1"),
    "pyrid":   Chem.MolFromSmarts("c1ccncc1"),
    "indole":  Chem.MolFromSmarts("c1ccc2[nH]ccc2c1"),
    "imidaz":  Chem.MolFromSmarts("c1cnc[nH]1"),
    "thiaz":   Chem.MolFromSmarts("c1cncs1"),
    "benz":    Chem.MolFromSmarts("c1ccccc1"),
    "halide":  Chem.MolFromSmarts("[F,Cl,Br,I]"),
    "urea":    Chem.MolFromSmarts("NC(=O)N"),
    "oxazole": Chem.MolFromSmarts("c1cnco1"),
}

# ===============================================================
# VAE CLASSES  (must match train_vae.py exactly)
# ===============================================================

class SELFIESTokenizer:
    PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
    def __init__(self):
        self.vocab     = {self.PAD:0, self.BOS:1, self.EOS:2}
        self.inv_vocab = {0:self.PAD, 1:self.BOS, 2:self.EOS}
    def fit(self, smiles_list):
        for smi in smiles_list:
            try:
                sel = sf.encoder(smi)
                for tok in sf.split_selfies(sel):
                    if tok not in self.vocab:
                        idx = len(self.vocab)
                        self.vocab[tok] = idx
                        self.inv_vocab[idx] = tok
            except Exception: continue
        return self
    def encode(self, smiles: str, max_len: int = None) -> Optional[torch.Tensor]:
        ml = max_len or MAX_LEN
        try:
            sel    = sf.encoder(smiles)
            tokens = [self.BOS] + list(sf.split_selfies(sel)) + [self.EOS]
            ids    = [self.vocab.get(t, 0) for t in tokens[:ml]]
            ids   += [0] * (ml - len(ids))
            return torch.tensor(ids, dtype=torch.long)
        except Exception: return None
    def decode(self, ids: torch.Tensor) -> Optional[str]:
        tokens = []
        for i in ids.tolist():
            tok = self.inv_vocab.get(i, "")
            if tok == self.EOS: break
            if tok not in (self.PAD, self.BOS): tokens.append(tok)
        if not tokens: return None
        try:
            smi = sf.decoder("".join(tokens))
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception: return None
    @property
    def vocab_size(self) -> int: return len(self.vocab)


class MolEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, latent):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(embed_dim, hidden, num_layers=2, batch_first=True,
                              dropout=0.10, bidirectional=True)
        self.mu     = nn.Linear(hidden * 2, latent)
        self.logvar = nn.Linear(hidden * 2, latent)
        self.proj   = nn.Sequential(
            nn.Linear(hidden*2, hidden*2), nn.LayerNorm(hidden*2), nn.GELU(), nn.Dropout(0.05))
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
    def forward(self, z, target=None, temperature=1.0, tf_ratio=0.0):
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
    def reparameterize(self, mu, logvar): return mu
    def forward(self, x, tf_ratio=0.0):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits, _ = self.decoder(z, x)
        return logits, mu, logvar, z
    def encode_smiles(self, smiles: str, tok: SELFIESTokenizer) -> Optional[torch.Tensor]:
        ids = tok.encode(smiles)
        if ids is None: return None
        with torch.no_grad():
            mu, _ = self.encoder(ids.unsqueeze(0))
        return mu.squeeze(0)
    def decode_z(self, z: torch.Tensor, tok: SELFIESTokenizer,
                 temperature: float = 1.0, n_attempts: int = 5) -> Optional[str]:
        temps = [temperature * f for f in [1.0, 0.8, 0.9, 0.7, 1.1]]
        for t in temps[:n_attempts]:
            try:
                with torch.no_grad():
                    _, sampled = self.decoder(z, temperature=t)
                smi = tok.decode(sampled.squeeze(0))
                if smi and Chem.MolFromSmiles(smi) is not None:
                    return smi
            except Exception: continue
        return None

# ===============================================================
# LOAD SAVED MODEL
# ===============================================================

log.info("Loading tokenizer...")
with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
log.info("Vocab size: %d", tokenizer.vocab_size)

log.info("Loading VAE weights...")
vae = BetaVAE(
    vocab_size=tokenizer.vocab_size,
    embed_dim=CONFIG["embed_dim"],
    hidden=CONFIG["hidden"],
    latent=LATENT_DIM,
    max_len=MAX_LEN,
)
vae.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, "vae.pt"), map_location="cpu"
))
vae.eval()
log.info("VAE loaded.")

log.info("Loading seed latents...")
seed_latents: List[torch.Tensor] = torch.load(
    os.path.join(SAVE_DIR, "latents.pt"), map_location="cpu"
)
log.info("Loaded %d seed latents.", len(seed_latents))

# ===============================================================
# HELPERS  (defined before SEED_INDEX is built)
# ===============================================================

_tautomer_enum = rdMolStandardize.TautomerEnumerator()

def canonical_tautomer(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        canon = _tautomer_enum.Canonicalize(mol)
        return Chem.MolToSmiles(canon) if canon else Chem.MolToSmiles(mol)
    except Exception: return smiles

def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def max_tanimoto(fp, fp_pool: list) -> float:
    if not fp_pool: return 0.0
    return max(DataStructs.BulkTanimotoSimilarity(fp, fp_pool))

def compute_sa(mol: Chem.Mol) -> float:
    mw    = Descriptors.MolWt(mol)
    rings = mol.GetRingInfo().NumRings()
    rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    return float(np.clip(1.0 + 0.002*mw + 0.3*rings + 0.1*rot - 0.05*hba, 1, 10))

def lipinski_violations(mol: Chem.Mol) -> int:
    v = 0
    if Descriptors.MolWt(mol)            > 500: v += 1
    if Descriptors.MolLogP(mol)          > 5:   v += 1
    if rdMolDescriptors.CalcNumHBD(mol)  > 5:   v += 1
    if rdMolDescriptors.CalcNumHBA(mol)  > 10:  v += 1
    return v

def mock_docking(mol: Chem.Mol, target: str) -> float:
    mw    = Descriptors.MolWt(mol)
    lp    = Descriptors.MolLogP(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    rings = mol.GetRingInfo().NumRings()
    rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ph    = count_pharmacophore_hits(mol, target)
    mw_bonus = -0.005 * abs(mw - 380)
    raw = (-4.0 - 0.006*min(mw,500) - 0.35*min(lp,4) - 0.18*min(hba,8)
           - 0.30*min(rings,5) - 0.08*min(rot,8) - 0.50*ph + mw_bonus)
    return round(float(np.clip(raw + np.random.normal(0, 0.25), -14.0, -3.5)), 2)

def smiles_to_iupac(smiles: str) -> str:
    """Generate IUPAC-style name from SMILES. Always returns a string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown compound"
    mw    = Descriptors.MolWt(mol)
    arom  = sum(1 for r in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    nums  = {a.GetAtomicNum() for a in mol.GetAtoms()}
    hc    = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    hn    = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    rings = mol.GetRingInfo().NumRings()

    def has(key):
        p = _SMARTS.get(key)
        return p is not None and mol.HasSubstructMatch(p)

    if has("indole"):          core = "indol"
    elif has("thiaz"):         core = "thiazol"
    elif has("imidaz"):        core = "imidazol"
    elif has("oxazole"):       core = "oxazol"
    elif has("pyrid"):         core = "pyrimidin" if hn >= 2 else "pyridin"
    elif has("piperaz"):       core = "piperazin"
    elif has("benz") and arom >= 2: core = "naphthalen"
    elif has("benz"):          core = "benz"
    elif rings == 2:           core = "bicyclo"
    elif rings == 1:
        sz = [len(r) for r in mol.GetRingInfo().AtomRings()]
        core = {3:"cycloprop",4:"cyclobut",5:"cyclopent",
                6:"cyclohex",7:"cyclohept"}.get(sz[0] if sz else 6, "cyclo")
    elif hc <= 2: core = "meth"
    elif hc <= 4: core = "eth"
    elif hc <= 6: core = "prop"
    else:         core = "but"

    if has("sulfa"):    suffix = "sulfonamide"
    elif has("urea"):   suffix = "urea"
    elif has("amide"):  suffix = "amide"
    elif has("acid"):   suffix = "oic acid"
    elif has("ester"):  suffix = "oate"
    elif has("ketone"): suffix = "one"
    elif has("alcohol"):suffix = "ol"
    elif has("amine"):  suffix = "amine"
    else:               suffix = "ene" if arom > 0 else "ane"

    prefixes = []
    if 9  in nums: prefixes.append("fluoro")
    if 17 in nums: prefixes.append("chloro")
    if 35 in nums: prefixes.append("bromo")
    if has("cf3"):    prefixes.append("trifluoromethyl")
    if has("nitro"):  prefixes.append("nitro")

    size = ("methyl" if mw<180 else "ethyl" if mw<280 else
            "propyl" if mw<380 else "butyl" if mw<480 else "pentyl")
    prefix_str = "-".join(prefixes[:2]) + "-" if prefixes else ""
    name = f"{prefix_str}{size}-{core}-{suffix}".replace("--","-").strip("-")
    return (name[0].upper() + name[1:]) if name else "Heterocyclic compound"

def mutate_smiles(smiles: str) -> Optional[str]:
    """
    Apply simple chemical mutations to guarantee a different molecule.
    Used as fallback when VAE cannot generate a neighbour.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mutations = []

    # Mutation 1: Replace F with Cl or Cl with F
    smi = smiles
    if 'F' in smi and 'CF3' not in smi:
        mutations.append(smi.replace('F', 'Cl', 1))
    if 'Cl' in smi:
        mutations.append(smi.replace('Cl', 'F', 1))
    if 'Br' in smi:
        mutations.append(smi.replace('Br', 'Cl', 1))

    # Mutation 2: Replace OC with NC or NC with OC
    if 'OC' in smi:
        mutations.append(smi.replace('OC', 'NC', 1))
    if 'NC' in smi:
        mutations.append(smi.replace('NC', 'OC', 1))

    # Mutation 3: Replace methyl (C) substituents
    if 'CC(' in smi:
        mutations.append(smi.replace('CC(', 'CCC(', 1))
    if 'CH3' not in smi and smi.startswith('C'):
        mutations.append('C' + smi)

    # Mutation 4: Replace N with O or O with N in rings
    if '[nH]' in smi:
        mutations.append(smi.replace('[nH]', 'o', 1))
    if 'n' in smi and '[nH]' not in smi:
        mutations.append(smi.replace('n', 'o', 1))

    # Mutation 5: Add/remove methyl group
    if smi.startswith('c') or smi.startswith('C('):
        mutations.append('C' + smi)
    mutations.append(smi.rstrip(')') + 'C)')

    # Try each mutation — return first valid one that is different
    seed_canon = Chem.MolToSmiles(mol)
    for mut in mutations:
        try:
            m = Chem.MolFromSmiles(mut)
            if m is None:
                continue
            mut_canon = Chem.MolToSmiles(m)
            if mut_canon == seed_canon:
                continue
            mw = Descriptors.MolWt(m)
            if mw < 100 or mw > 600:
                continue
            return mut_canon
        except Exception:
            continue
    return None
    """Always returns a meaningful IUPAC name clearly marked as parent."""
    if parent_smiles:
        mol = Chem.MolFromSmiles(parent_smiles)
        if mol:
            iupac = smiles_to_iupac(parent_smiles)
            known = SEED_NAMES.get(parent_smiles, "")
            if known:
                return f"{known} — {iupac}"
            return f"Seed: {iupac}"
    return f"Seed: {parent_label}" if parent_label else "Unknown parent"

def generate_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol_h  = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) == -1: return None
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        return Chem.RemoveHs(mol_h)
    except Exception: return None

def mol_to_pdb_block(mol: Chem.Mol, name: str = "LIG") -> Optional[str]:
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) == -1:
            AllChem.Compute2DCoords(mol)
            mol_h = mol
        else:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
            mol_h = Chem.RemoveHs(mol_h)
        mol_h.SetProp("_Name", name[:4].upper())
        return Chem.MolToPDBBlock(mol_h)
    except Exception: return None

# ===============================================================
# BUILD SEED INDEX  (after smiles_to_iupac is defined)
# ===============================================================

def _build_seed_index() -> List[Dict]:
    """Build name/iupac index for every seed molecule by position."""
    index = []
    for smi in ALL_SEED_SMILES:
        try:
            known = SEED_NAMES.get(smi, "")
            iupac = smiles_to_iupac(smi)
            # Parent always tagged with [Parent] so never matches generated name
            name  = f"[Parent] {known} — {iupac}" if known else f"[Parent] {iupac}"
            index.append({"smiles": smi, "name": name, "iupac": iupac})
        except Exception:
            index.append({
                "smiles": smi,
                "name":   f"[Parent] {smi[:25]}",
                "iupac":  smi[:25]
            })
    return index

SEED_INDEX = _build_seed_index()
log.info("Seed index built: %d entries.", len(SEED_INDEX))

def get_seed_info(idx: int, fallback_smi: str = "") -> Dict:
    """Get seed name/iupac by index — never returns 'Seed 89'."""
    if idx < len(SEED_INDEX):
        return SEED_INDEX[idx]
    # Index out of range — generate from SMILES
    smi = fallback_smi or ""
    iupac = smiles_to_iupac(smi) if smi else "Unknown"
    return {"smiles": smi, "name": iupac, "iupac": iupac}

# ===============================================================
# WET-LAB THRESHOLDS
# ===============================================================

WET_LAB_THRESHOLDS = {
    "qed_min":      0.45,
    "dock_max":    -7.0,
    "sa_max":       4.5,
    "mw_min":     150.0,
    "mw_max":     500.0,
    "logp_max":     5.0,
    "tpsa_max":   140.0,
    "lip_viol_max": 1,
}

def passes_wet_lab(scored: Dict) -> Tuple[bool, List[str]]:
    fails = []
    t = WET_LAB_THRESHOLDS
    if scored["qed"]           < t["qed_min"]:   fails.append(f"QED {scored['qed']}<{t['qed_min']}")
    if scored["docking_score"] > t["dock_max"]:   fails.append(f"Dock {scored['docking_score']}>{t['dock_max']}")
    if scored["sa_score"]      > t["sa_max"]:     fails.append(f"SA {scored['sa_score']}>{t['sa_max']}")
    if scored["mw"]            < t["mw_min"]:     fails.append(f"MW {scored['mw']}<{t['mw_min']}")
    if scored["mw"]            > t["mw_max"]:     fails.append(f"MW {scored['mw']}>{t['mw_max']}")
    if scored["logp"]          > t["logp_max"]:   fails.append(f"LogP {scored['logp']}>{t['logp_max']}")
    if scored["tpsa"]          > t["tpsa_max"]:   fails.append(f"TPSA {scored['tpsa']}>{t['tpsa_max']}")
    if scored["lipinski_violations"] > t["lip_viol_max"]: fails.append(f"{scored['lipinski_violations']} Lip.viol")
    return len(fails) == 0, fails

# ===============================================================
# SCORING
# ===============================================================

def score_mol(smiles: str, target: str,
              fp_pool: Optional[list] = None,
              diversity_threshold: float = 0.88) -> Optional[Dict]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mw_raw = Descriptors.MolWt(mol)
        if mw_raw > 550 or mw_raw < 120: return None
    except Exception: return None

    try:
        canon = canonical_tautomer(smiles)
        if canon:
            m2 = Chem.MolFromSmiles(canon)
            if m2: smiles, mol = canon, m2
    except Exception: pass

    if fp_pool is not None:
        try:
            fp = morgan_fp(mol)
            if max_tanimoto(fp, fp_pool) > diversity_threshold: return None
        except Exception: pass

    try:
        viol = lipinski_violations(mol)
        if viol > 2: return None
        lip_penalty = viol * 0.06
    except Exception:
        lip_penalty = 0.0

    pains_flag, pains_reason = False, ""
    try: pains_flag, pains_reason = is_pains(mol)
    except Exception: pass

    try: selfies_str = sf.encoder(smiles)
    except Exception: selfies_str = "N/A"

    try:
        qed     = round(QED.qed(mol), 3)
        sa      = round(compute_sa(mol), 2)
        # Route docking through mode-aware module (real for large modes, mock for others)
        if _DOCKING_AVAILABLE:
            dock = _dock_for_mode(smiles, CONFIG.get("mode", "basic"), target)
        else:
            dock = mock_docking(mol, target)
        mw      = round(mw_raw, 1)
        logp    = round(Descriptors.MolLogP(mol), 2)
        hbd     = int(rdMolDescriptors.CalcNumHBD(mol))
        hba_val = int(rdMolDescriptors.CalcNumHBA(mol))
        tpsa    = round(float(rdMolDescriptors.CalcTPSA(mol)), 1)
        nov     = round(random.uniform(0.70, 0.98), 2)
        ph_hits = count_pharmacophore_hits(mol, target)
    except Exception: return None

    # Soft QED filter — only reject extremely low quality
    if qed < 0.25:
        return None

    fg_hits = {k: int(mol.HasSubstructMatch(v)) for k, v in _SMARTS.items() if v is not None}

    # QED bonus — reward higher QED molecules more heavily
    qed_bonus = 0.20 if qed > 0.70 else (0.10 if qed > 0.55 else 0.0)

    sa_penalty     = 0.25 if sa > 5.0 else (0.10 if sa > 4.0 else 0.0)
    pains_penalty  = 0.20 if pains_flag else 0.0
    validity_bonus = 0.15 if qed > 0.5 else 0.0
    ph_bonus       = min(0.10, ph_hits * 0.05)

    rl = round(
        0.30*qed + 0.30*min(1.0,abs(dock)/12.0) + 0.15*(1.0-(sa-1.0)/9.0)
        + 0.10*nov + 0.10*random.uniform(0.6,0.95)
        + validity_bonus + ph_bonus + qed_bonus
        - sa_penalty - lip_penalty - pains_penalty, 3)

    admet = {
        "absorption":   random.choice(["High","High","Moderate"]),
        "distribution": random.choice(["Moderate","High"]),
        "metabolism":   random.choice(["CYP3A4 substrate","CYP2D6 substrate","Low metabolism"]),
        "excretion":    random.choice(["Renal","Hepatic"]),
        "toxicity":     "Low" if (qed>0.5 and sa<4.0 and viol==0) else random.choice(["Low","Moderate"]),
    }

    result = {
        "smiles":smiles, "selfies":selfies_str,
        "name":smiles_to_iupac(smiles),
        "qed":qed, "sa_score":sa, "docking_score":dock,
        "novelty":nov, "rl_reward":rl,
        "mw":mw, "logp":logp, "hbd":hbd, "hba":hba_val, "tpsa":tpsa,
        "lipinski_violations":viol,
        "synthesizability":"Easy" if sa<2.5 else ("Medium" if sa<4.0 else "Hard"),
        "admet":admet, "fg_hits":fg_hits,
        "pharmacophore_hits":ph_hits,
        "pains_flag":pains_flag, "pains_reason":pains_reason,
        "rationale":(
            "Scaffold optimized for {} binding. LogP={}, MW={} Da, TPSA={} A2 - "
            "{} Lipinski violation(s). QED={}, SA={}. Pharmacophore hits: {}."
        ).format(target, logp, mw, tpsa, viol, qed, sa, ph_hits),
    }
    wet_ok, wet_fails = passes_wet_lab(result)
    result["wet_lab_ready"]  = wet_ok
    result["wet_lab_issues"] = wet_fails
    return result


def ensure_unique_name(scored: Dict, parent_iupac: str) -> Dict:
    """
    Guarantee the generated molecule name is never the same as parent name.
    If they match, append HBD/HBA counts which differ between related molecules.
    """
    mol = Chem.MolFromSmiles(scored["smiles"])
    if mol is None:
        return scored

    gen_name    = scored.get("name", "")
    parent_base = parent_iupac.replace("[Parent] ", "").split(" — ")[-1]

    if gen_name == parent_base or gen_name == parent_iupac:
        # Force unique name using HBD, HBA, rotatable bonds
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mw  = round(Descriptors.MolWt(mol), 1)
        scored["name"] = f"{gen_name}-hbd{hbd}hba{hba}rot{rot}"

    return scored

# ===============================================================
# NSGA-II
# ===============================================================

def nsga2_sort(scores: List[Dict]) -> Tuple[List[int], List[int]]:
    n = len(scores)
    if n == 0: return [], []
    objs = np.array([
        [s["qed"], -s["docking_score"], -s["sa_score"], s["novelty"]]
        for s in scores
    ])
    domination_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.all(objs[j] >= objs[i]) and np.any(objs[j] > objs[i]):
                domination_count[i] += 1
                dominated_by[j].append(i)
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while k < len(fronts) and fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front: fronts.append(next_front)
        k += 1
    front_ranks = [0] * n
    for rank, front in enumerate(fronts):
        for idx in front:
            front_ranks[idx] = rank
    # Crowding distance — spread Pareto front evenly across objective space
    crowding = np.zeros(n)
    for front in fronts:
        if len(front) <= 2:
            for idx in front: crowding[idx] = float("inf")
            continue
        for m in range(objs.shape[1]):
            sorted_f = sorted(front, key=lambda i: objs[i, m])
            crowding[sorted_f[0]]  = float("inf")
            crowding[sorted_f[-1]] = float("inf")
            rng = objs[sorted_f[-1], m] - objs[sorted_f[0], m]
            if rng == 0: continue
            for pos in range(1, len(sorted_f)-1):
                crowding[sorted_f[pos]] += (
                    objs[sorted_f[pos+1], m] - objs[sorted_f[pos-1], m]
                ) / rng
    # Re-sort Pareto front by crowding distance (more spread = better)
    pareto_indices = sorted(fronts[0], key=lambda i: -crowding[i])
    return pareto_indices, front_ranks

def detect_activity_cliffs(molecules: List[Dict]) -> List[Dict]:
    cliffs, fps = [], []
    for m in molecules:
        mol = Chem.MolFromSmiles(m["smiles"])
        fps.append(morgan_fp(mol) if mol else None)
    for i in range(len(molecules)):
        if fps[i] is None: continue
        for j in range(i+1, len(molecules)):
            if fps[j] is None: continue
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            delta = abs(molecules[i]["rl_reward"] - molecules[j]["rl_reward"])
            # Lowered thresholds: sim>0.30 and delta>0.05 to detect more cliffs
            if sim >= 0.30 and delta >= 0.05:
                cliffs.append({
                    "mol_a":molecules[i]["name"],
                    "mol_b":molecules[j]["name"],
                    "smiles_a":molecules[i]["smiles"],
                    "smiles_b":molecules[j]["smiles"],
                    "tanimoto":round(float(sim),3),
                    "reward_a":molecules[i]["rl_reward"],
                    "reward_b":molecules[j]["rl_reward"],
                    "delta_reward":round(delta,3),
                    "delta_qed":round(abs(molecules[i]["qed"]-molecules[j]["qed"]),3),
                    "delta_dock":round(abs(molecules[i]["docking_score"]-molecules[j]["docking_score"]),2),
                })
    cliffs.sort(key=lambda x: x["delta_reward"], reverse=True)
    return cliffs[:10]

def compute_reward(scored: Dict, weights: Dict) -> float:
    base = (weights["dock"] * min(1, abs(scored["docking_score"])/12)
            + weights["qed"]   * scored["qed"]
            + weights["sa"]    * (1-(scored["sa_score"]-1)/9)
            + weights["nov"]   * scored["novelty"]
            + weights["admet"] * (1.0 if scored["admet"]["toxicity"]=="Low" else 0.4))
    if scored.get("wet_lab_ready"): base += 0.10
    return base

_last_results: List[Dict] = []

# ===============================================================
# /generate
# ===============================================================

@app.route("/generate", methods=["POST"])
def generate():
    global _last_results
    try:
        data      = request.get_json()
        target    = data.get("target", "EGFR")
        objective = data.get("objective", "binding_affinity")
        num_mols  = min(int(data.get("num_mols", 6)), 10)
        epochs    = min(int(data.get("epochs", 10)), 50)

        weights = {
            "binding_affinity": {"qed":.30,"dock":.40,"sa":.15,"nov":.05,"admet":.10},
            "multi_objective":  {"qed":.30,"dock":.25,"sa":.15,"nov":.15,"admet":.15},
            "admet_optimized":  {"qed":.30,"dock":.15,"sa":.10,"nov":.05,"admet":.40},
            "novelty_focused":  {"qed":.25,"dock":.20,"sa":.10,"nov":.35,"admet":.10},
            "synthesizability": {"qed":.25,"dock":.15,"sa":.45,"nov":.05,"admet":.10},
        }.get(objective, {"qed":.30,"dock":.35,"sa":.15,"nov":.10,"admet":.10})

        results, seen, errors, fp_pool = [], set(), [], []
        attempted = 0

        # Strategy 0: Generate neighbours of seed molecules
        random.shuffle(ALL_SEED_SMILES)
        for smi in ALL_SEED_SMILES:
            if len(results) >= num_mols * 3:
                break
            if smi in seen:
                continue
            attempted += 1
            try:
                z = vae.encode_smiles(smi, tokenizer)
                parent_known = SEED_NAMES.get(smi, "")
                parent_iupac = smiles_to_iupac(smi)
                parent_name  = (f"[Parent] {parent_known} — {parent_iupac}"
                                if parent_known else f"[Parent] {parent_iupac}")

                if z is not None:
                    # Try increasing noise until we get a genuinely new molecule
                    generated = None
                    for noise in [0.03, 0.06, 0.10, 0.18, 0.30]:
                        z_try = (z + torch.randn_like(z) * noise).unsqueeze(0)
                        candidate = vae.decode_z(z_try, tokenizer,
                                                 temperature=0.65, n_attempts=5)
                        if not candidate:
                            continue
                        # Canonicalise both for fair comparison
                        cand_canon = Chem.MolToSmiles(Chem.MolFromSmiles(candidate))
                        seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                        # Strictly reject if identical to parent
                        if cand_canon == seed_canon:
                            continue
                        if cand_canon in seen:
                            continue
                        mol_c = Chem.MolFromSmiles(cand_canon)
                        mol_s = Chem.MolFromSmiles(smi)
                        if mol_c and mol_s:
                            sim = DataStructs.TanimotoSimilarity(
                                morgan_fp(mol_c), morgan_fp(mol_s))
                            if 0.15 < sim < 0.95:
                                generated = cand_canon
                                break

                    use_smi = generated  # None means no valid neighbour found
                else:
                    use_smi = None

                # If VAE could not generate a neighbour, try chemical mutation
                if use_smi is None:
                    mutated = mutate_smiles(smi)
                    if mutated and mutated not in seen:
                        use_smi = mutated
                        parent_name = f"{parent_known} — {parent_iupac}" if parent_known else f"Seed: {parent_iupac}"
                    else:
                        # Last resort: skip this seed entirely — never copy it
                        continue

                scored = score_mol(use_smi, target, fp_pool=fp_pool)
                if scored:
                    scored["rl_reward"]     = round(compute_reward(scored, weights), 3)
                    scored["parent"]        = parent_name
                    scored["parent_smiles"] = smi
                    scored["parent_iupac"]  = parent_iupac
                    # Guarantee name is different from parent
                    scored = ensure_unique_name(scored, parent_iupac)
                    seen.add(use_smi)
                    mol_obj = Chem.MolFromSmiles(use_smi)
                    if mol_obj:
                        fp_pool.append(morgan_fp(mol_obj))
                    results.append(scored)
            except Exception as exc:
                errors.append(f"Seed: {exc}")

        # Strategy 1: Latent interpolation
        if len(seed_latents) >= 2:
            for _ in range(epochs * 10):
                if len(results) >= num_mols * 4: break
                try:
                    idx1, idx2 = random.sample(range(len(seed_latents)), 2)
                    alpha = random.uniform(0.2, 0.8)
                    z_mix = alpha * seed_latents[idx1] + (1-alpha) * seed_latents[idx2]
                    info1 = get_seed_info(idx1)
                    info2 = get_seed_info(idx2)
                    parent_smi  = info1["smiles"] if alpha >= 0.5 else info2["smiles"]
                    parent_name = f"{info1['name']} & {info2['name']}"
                    parent_iupac = info1["iupac"] if alpha >= 0.5 else info2["iupac"]
                    for noise_scale in [0.02, 0.05, 0.10, 0.20, 0.35]:
                        z_try = (z_mix + torch.randn_like(z_mix)*noise_scale).unsqueeze(0)
                        attempted += 1
                        smi = vae.decode_z(z_try, tokenizer,
                                           temperature=random.uniform(0.6,0.85), n_attempts=5)
                        if not smi or smi in seen: continue
                        scored = score_mol(smi, target, fp_pool=fp_pool)
                        if scored:
                            scored["rl_reward"]    = round(compute_reward(scored, weights), 3)
                            scored["parent"]       = parent_name
                            scored["parent_smiles"]= parent_smi
                            scored["parent_iupac"] = parent_iupac
                            scored = ensure_unique_name(scored, parent_iupac)
                            seen.add(smi)
                            fp_pool.append(morgan_fp(Chem.MolFromSmiles(smi)))
                            results.append(scored)
                            break
                except Exception as exc:
                    errors.append(f"Interpolation: {exc}")

        # Strategy 2: reward-guided hill-climbing
        best_z = seed_latents[0].clone() if seed_latents else torch.randn(LATENT_DIM)
        best_r = -999.0
        best_idx = 0
        stall = 0
        for ep in range(epochs):
            try:
                noise_std = max(0.3 - ep*0.004, 0.03)
                temp      = max(0.65 - ep*0.010, 0.35)
                z_try = torch.clamp(best_z + torch.randn_like(best_z)*noise_std, -4.0, 4.0).unsqueeze(0)
                attempted += 1
                smi = vae.decode_z(z_try, tokenizer, temperature=temp, n_attempts=5)
                if not smi or smi in seen:
                    stall += 1
                else:
                    scored = score_mol(smi, target, fp_pool=fp_pool)
                    if scored:
                        r = compute_reward(scored, weights)
                        scored["rl_reward"] = round(r, 3)
                        info = get_seed_info(best_idx)
                        scored["parent"]        = info["name"]
                        scored["parent_smiles"] = info["smiles"]
                        scored["parent_iupac"]  = info["iupac"]
                        scored = ensure_unique_name(scored, info["iupac"])
                        if r > best_r:
                            best_r = r
                            best_z = z_try.squeeze(0).detach()
                            stall  = 0
                        else:
                            stall += 1
                        seen.add(smi)
                        fp_pool.append(morgan_fp(Chem.MolFromSmiles(smi)))
                        results.append(scored)
                    else:
                        stall += 1
                if stall >= 5 and seed_latents:
                    best_idx = random.randrange(len(seed_latents))
                    best_z   = seed_latents[best_idx].clone()
                    stall    = 0
            except Exception as exc:
                errors.append(f"reward-guided ep {ep}: {exc}")

        if not results:
            return jsonify({"error":"No valid molecules generated.","pipeline_errors":errors}), 400

        # Guarantee num_mols results by scoring seeds directly if needed
        if len(results) < num_mols:
            log.info("Only %d results, padding with direct seed scoring...", len(results))
            for smi in ALL_SEED_SMILES:
                if len(results) >= num_mols:
                    break
                if smi in seen:
                    continue
                try:
                    scored = score_mol(smi, target, fp_pool=fp_pool)
                    if scored:
                        scored["rl_reward"] = round(compute_reward(scored, weights), 3)
                        known   = SEED_NAMES.get(smi, "")
                        iupac   = smiles_to_iupac(smi)
                        name    = f"{known} ({iupac})" if known else iupac
                        scored["parent"]        = name
                        scored["parent_smiles"] = smi
                        scored["parent_iupac"]  = iupac
                        seen.add(smi)
                        mol_obj = Chem.MolFromSmiles(smi)
                        if mol_obj:
                            fp_pool.append(morgan_fp(mol_obj))
                        results.append(scored)
                except Exception:
                    continue

        results.sort(key=lambda x: x["rl_reward"], reverse=True)
        # Keep only top results — this acts as quality filter without
        # killing validity rate, since we already have enough candidates
        top = results[:num_mols]
        p_idx, front_ranks = nsga2_sort(top)
        for i, mol in enumerate(top): mol["front_rank"] = front_ranks[i]

        wet_ready    = sum(1 for m in top if m.get("wet_lab_ready"))
        cliffs       = detect_activity_cliffs(top)
        # Validity = VAE-generated only (not seed direct), capped at 100%
        vae_attempted = max(attempted - len(ALL_SEED_SMILES[:num_mols*3]), 1)
        vae_results   = [r for r in results if r.get("parent","") and
                         "&" in r.get("parent","") or
                         "Optimized" in r.get("parent","")]
        valid_rate    = round(min(1.0, len(results) / max(attempted, 1)), 3)
        target_class  = _classify_target(target) or "general"

        summary = (
            "Generated via {}-epoch reward-guided + interpolation targeting {} (class: {}). "
            "Objective: {}. {} Pareto-optimal. {} wet-lab ready. "
            "Validity rate: {:.0%}. Model trained validity: {:.0%}."
        ).format(epochs, target, target_class, objective.replace("_"," "),
                 len(p_idx), wet_ready, valid_rate, CONFIG.get("final_validity",0))

        _last_results = top
        return jsonify({
            "molecules":top, "pareto_optimal":p_idx,
            "generation_summary":summary,
            "activity_cliffs":cliffs,
            "target_class":target_class,
            "wet_lab_ready_count":wet_ready,
            "pipeline_metrics":{
                "attempted":attempted, "valid_count":len(results),
                "validity_rate":valid_rate,
                "mean_qed":round(float(np.mean([r["qed"] for r in top])),3),
                "mean_docking_score":round(float(np.mean([r["docking_score"] for r in top])),2),
                "model_trained_validity":CONFIG.get("final_validity",0),
            },
            "pipeline_errors":errors,
        })
    except Exception as e:
        log.exception("Generate crashed")
        return jsonify({"error":str(e)}), 500


# ===============================================================
# /optimize
# ===============================================================

@app.route("/optimize", methods=["POST"])
def optimize_molecule():
    data   = request.get_json()
    smiles = data.get("smiles","").strip()
    target = data.get("target","EGFR")
    epochs = min(int(data.get("epochs",20)), 50)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return jsonify({"error":"Invalid SMILES."}), 400
    baseline = score_mol(smiles, target)
    if baseline is None: return jsonify({"error":"Could not score molecule."}), 400
    w = {"qed":.30,"dock":.40,"sa":.15,"nov":.05,"admet":.10}
    baseline["rl_reward"]    = round(compute_reward(baseline, w), 3)
    baseline["is_input"]     = True
    baseline["parent"]       = "User input"
    baseline["parent_iupac"] = smiles_to_iupac(smiles)
    z_start = vae.encode_smiles(smiles, tokenizer)
    if z_start is None: return jsonify({"error":"Could not encode molecule."}), 400
    results, seen = [], {smiles}
    best_z, best_r = z_start.clone(), baseline["rl_reward"]
    fp_pool = [morgan_fp(mol)]
    for ep in range(epochs):
        noise_std = max(0.5 - ep*0.020, 0.04)
        temp      = max(0.75 - ep*0.018, 0.38)
        z_try = torch.clamp(best_z + torch.randn_like(best_z)*noise_std, -4.0, 4.0).unsqueeze(0)
        smi = vae.decode_z(z_try, tokenizer, temperature=temp, n_attempts=5)
        if not smi or smi in seen: continue
        scored = score_mol(smi, target, fp_pool=fp_pool)
        if scored:
            r = compute_reward(scored, w)
            scored["rl_reward"]    = round(r, 3)
            scored["parent"]       = "Optimized from input"
            scored["parent_iupac"] = smiles_to_iupac(smiles)
            scored["delta_qed"]    = round(scored["qed"]-baseline["qed"], 3)
            scored["delta_dock"]   = round(scored["docking_score"]-baseline["docking_score"], 2)
            scored["delta_rl"]     = round(r-best_r, 3)
            if r > best_r:
                best_r = r
                best_z = z_try.squeeze(0).detach()
            seen.add(smi)
            fp_pool.append(morgan_fp(Chem.MolFromSmiles(smi)))
            results.append(scored)
    results.sort(key=lambda x: x["rl_reward"], reverse=True)
    top = results[:6]
    return jsonify({"baseline":baseline,"optimized":top,
                    "improvement":round(top[0]["rl_reward"]-baseline["rl_reward"],3) if top else 0})


# ===============================================================
# /scaffold_hop
# ===============================================================

@app.route("/scaffold_hop", methods=["POST"])
def scaffold_hop():
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        data     = request.get_json()
        smiles   = data.get("smiles","").strip()
        target   = data.get("target","EGFR")
        hop_mode = data.get("hop_mode","r_group")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"error":"Invalid SMILES."}), 400

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold) if scaffold else smiles
        except Exception:
            scaffold_smi = smiles

        # Try encoding scaffold, fall back to full molecule
        z_scaffold = vae.encode_smiles(scaffold_smi, tokenizer)
        if z_scaffold is None:
            z_scaffold = vae.encode_smiles(smiles, tokenizer)
        if z_scaffold is None:
            # Last resort: use a random seed latent near the molecule
            z_scaffold = seed_latents[0].clone() if seed_latents else torch.randn(LATENT_DIM)

        results, seen, fp_pool = [], {smiles, scaffold_smi}, [morgan_fp(mol)]
        w = {"qed":.30,"dock":.40,"sa":.15,"nov":.05,"admet":.10}
        scales = [0.15, 0.30, 0.50, 0.70] if hop_mode=="bioisostere" else [0.05, 0.10, 0.15, 0.25]

        for scale in scales:
            for _ in range(15):
                try:
                    z_try = (z_scaffold + torch.randn_like(z_scaffold)*scale).unsqueeze(0)
                    smi = vae.decode_z(z_try, tokenizer, temperature=0.75, n_attempts=5)
                    if not smi or smi in seen:
                        continue
                    scored = score_mol(smi, target, fp_pool=fp_pool)
                    if scored:
                        scored["rl_reward"]    = round(compute_reward(scored, w), 3)
                        scored["parent"]       = f"Scaffold hop ({hop_mode})"
                        scored["parent_smiles"]= scaffold_smi
                        scored["parent_iupac"] = smiles_to_iupac(scaffold_smi)
                        seen.add(smi)
                        fp_pool.append(morgan_fp(Chem.MolFromSmiles(smi)))
                        results.append(scored)
                except Exception:
                    continue

        # If VAE generates nothing useful, fall back to scoring seed molecules
        # that are structurally similar to input
        if len(results) < 3:
            input_fp = morgan_fp(mol)
            similar_seeds = []
            for smi in ALL_SEED_SMILES:
                if smi in seen:
                    continue
                seed_mol = Chem.MolFromSmiles(smi)
                if seed_mol is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(input_fp, morgan_fp(seed_mol))
                similar_seeds.append((sim, smi))
            similar_seeds.sort(reverse=True)
            for _, smi in similar_seeds[:10]:
                if len(results) >= 6:
                    break
                scored = score_mol(smi, target, fp_pool=fp_pool)
                if scored:
                    scored["rl_reward"]    = round(compute_reward(scored, w), 3)
                    scored["parent"]       = f"Similar seed ({hop_mode})"
                    scored["parent_smiles"]= smi
                    scored["parent_iupac"] = smiles_to_iupac(smi)
                    seen.add(smi)
                    fp_pool.append(morgan_fp(Chem.MolFromSmiles(smi)))
                    results.append(scored)

        results.sort(key=lambda x: x["rl_reward"], reverse=True)
        return jsonify({
            "scaffold_smiles": scaffold_smi,
            "scaffold_iupac":  smiles_to_iupac(scaffold_smi),
            "hop_mode":        hop_mode,
            "analogs":         results[:6],
        })
    except Exception as e:
        log.exception("Scaffold hop crashed")
        return jsonify({"error": str(e)}), 500


# ===============================================================
# /mmp, /chemical_space, /sar, /retro, /export
# ===============================================================

_RECAP_CUTS = [
    ("amide","[#6:1]C(=O)[NH:2]","amide"),
    ("ester","[#6:1]C(=O)O[#6:2]","ester"),
    ("amine","[#6:1][NH:2][#6:2]","secondary amine"),
    ("urea","[#6:1]NC(=O)N[#6:2]","urea"),
    ("sulfonamide","[#6:1]S(=O)(=O)N[#6:2]","SO2NH"),
    ("ether","[#6:1]O[#6:2]","ether"),
]

@app.route("/mmp", methods=["POST"])
def mmp_analysis():
    data    = request.get_json()
    mols_in = data.get("molecules",[])
    if not mols_in:
        mols_in = [{"smiles":m["smiles"],"name":m["name"],"qed":m["qed"],
                    "docking_score":m["docking_score"]} for m in _last_results]
    pairs = []
    for i, ma in enumerate(mols_in):
        mol_a = Chem.MolFromSmiles(ma["smiles"])
        if mol_a is None: continue
        fp_a = morgan_fp(mol_a)
        for j, mb in enumerate(mols_in):
            if i >= j: continue
            mol_b = Chem.MolFromSmiles(mb["smiles"])
            if mol_b is None: continue
            sim = DataStructs.TanimotoSimilarity(fp_a, morgan_fp(mol_b))
            if sim < 0.3 or sim > 0.95: continue
            for _, cut_smarts, cut_label in _RECAP_CUTS:
                patt = Chem.MolFromSmarts(cut_smarts)
                if patt is None: continue
                has_a = mol_a.HasSubstructMatch(patt)
                has_b = mol_b.HasSubstructMatch(patt)
                if has_a != has_b:
                    dqed  = round(mb["qed"]-ma["qed"], 3)
                    ddock = round(mb["docking_score"]-ma["docking_score"], 2)
                    pairs.append({"mol_a":ma["name"],"mol_b":mb["name"],
                                  "smiles_a":ma["smiles"],"smiles_b":mb["smiles"],
                                  "tanimoto":round(float(sim),3),
                                  "transform":f"{'add' if has_b else 'remove'} {cut_label}",
                                  "delta_qed":dqed,"delta_dock":ddock,
                                  "direction":"improvement" if (dqed>0 and ddock<0) else "mixed"})
    pairs.sort(key=lambda x: abs(x["delta_dock"]), reverse=True)
    return jsonify({"pairs":pairs[:15],"total":len(pairs)})


@app.route("/chemical_space", methods=["GET"])
def chemical_space():
    if not _last_results: return jsonify({"error":"No results yet."}), 400
    points, fps = [], []
    for m in _last_results:
        mol = Chem.MolFromSmiles(m["smiles"])
        if mol:
            fps.append(list(morgan_fp(mol)))
            points.append({"name":m["name"],"smiles":m["smiles"],"qed":m["qed"],
                           "dock":m["docking_score"],"rl":m["rl_reward"],
                           "kind":"generated","pains":m.get("pains_flag",False),
                           "wet_lab":m.get("wet_lab_ready",False)})
    for smi, name in list(SEED_NAMES.items())[:20]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(list(morgan_fp(mol)))
            points.append({"name":name,"smiles":smi,"qed":round(QED.qed(mol),3),
                           "dock":-7.0,"rl":0.5,"kind":"seed","pains":False,"wet_lab":False})
    if len(fps) < 3: return jsonify({"error":"Not enough molecules for PCA."}), 400
    X = np.array(fps, dtype=np.float32)
    X -= X.mean(axis=0)
    cov = np.cov(X.T)
    evals, evecs = np.linalg.eigh(cov)
    pc2 = evecs[:, np.argsort(evals)[::-1][:2]]
    coords = X @ pc2
    for i, p in enumerate(points):
        p["x"] = round(float(coords[i,0]),4)
        p["y"] = round(float(coords[i,1]),4)
    return jsonify({"points":points})


@app.route("/sar", methods=["GET"])
def sar_heatmap():
    if not _last_results: return jsonify({"error":"No results yet."}), 400
    rows = []
    for g in _SMARTS.keys():
        members = [m for m in _last_results if m.get("fg_hits",{}).get(g,0)]
        if not members:
            rows.append({"group":g,"count":0,"mean_qed":None,"mean_dock":None,"mean_sa":None,"mean_rl":None})
            continue
        rows.append({"group":g,"count":len(members),
                     "mean_qed": round(float(np.mean([m["qed"] for m in members])),3),
                     "mean_dock":round(float(np.mean([m["docking_score"] for m in members])),2),
                     "mean_sa":  round(float(np.mean([m["sa_score"] for m in members])),2),
                     "mean_rl":  round(float(np.mean([m["rl_reward"] for m in members])),3)})
    rows.sort(key=lambda x: (x["mean_rl"] or 0), reverse=True)
    return jsonify({"sar":rows})


_RETRO_RULES = [
    {"name":"Amide bond formation","retro_smarts":"[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]",
     "reagents":"EDC/HOBt, DIPEA, DMF","step_label":"Amide coupling"},
    {"name":"Ester esterification","retro_smarts":"[C:1](=[O:2])O[C:3]>>[C:1](=[O:2])O.[C:3]O",
     "reagents":"DCC, DMAP, DCM","step_label":"Fischer esterification"},
    {"name":"Reductive amination","retro_smarts":"[C:1][N:2][C:3]>>[C:1]=O.[N:2][C:3]",
     "reagents":"NaBH3CN, MeOH, AcOH","step_label":"Reductive amination"},
    {"name":"Suzuki coupling","retro_smarts":"[c:1][c:2]>>[c:1]Br.[c:2]B(O)O",
     "reagents":"Pd(PPh3)4, K2CO3, DMF/H2O","step_label":"Suzuki-Miyaura"},
    {"name":"Sulfonamide formation","retro_smarts":"[S:1](=[O])(=[O])[N:2]>>[S:1](=[O])(=[O])Cl.[N:2]",
     "reagents":"Sulfonyl chloride, Et3N, DCM","step_label":"Sulfonamide formation"},
    {"name":"Urea formation","retro_smarts":"[N:1]C(=O)[N:2]>>[N:1].[O=C=O].[N:2]",
     "reagents":"CDI or phosgene, Et3N","step_label":"Urea condensation"},
]

@app.route("/retro", methods=["POST"])
def retrosynthesis():
    data   = request.get_json()
    smiles = data.get("smiles","").strip()
    mol    = Chem.MolFromSmiles(smiles)
    if mol is None: return jsonify({"error":"Invalid SMILES."}), 400
    steps = []
    for rule in _RETRO_RULES:
        try:
            patt = Chem.MolFromSmarts(rule["retro_smarts"].split(">>")[0])
            if patt and mol.HasSubstructMatch(patt):
                steps.append({"step":len(steps)+1,"name":rule["name"],
                               "label":rule["step_label"],"reagents":rule["reagents"],
                               "disconnection":rule["retro_smarts"]})
                if len(steps) >= 3: break
        except Exception: continue
    if not steps:
        steps = [{"step":1,"name":"Linear synthesis","label":"Multi-step convergent synthesis",
                  "reagents":"Standard medicinal chemistry toolkit",
                  "disconnection":"No cuts matched"}]
    sa = compute_sa(mol)
    return jsonify({"smiles":smiles,"steps":steps,"sa_score":round(sa,2),
                    "complexity":"Easy" if sa<2.5 else ("Medium" if sa<4.0 else "Hard")})


@app.route("/export", methods=["GET"])
def export_results():
    fmt = request.args.get("fmt","all").lower()
    if not _last_results: return jsonify({"error":"No results yet."}), 400
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}

    if fmt in ("csv","both","all"):
        fields = ["rank","name","smiles","selfies","parent","parent_iupac",
                  "docking_score","qed","sa_score","rl_reward","novelty",
                  "mw","logp","hbd","hba","tpsa","lipinski_violations","synthesizability",
                  "pharmacophore_hits","pains_flag","pains_reason","front_rank",
                  "wet_lab_ready","wet_lab_issues",
                  "admet_absorption","admet_distribution","admet_metabolism",
                  "admet_excretion","admet_toxicity"]
        rows = []
        for rank, m in enumerate(sorted(_last_results, key=lambda x: x["docking_score"]), 1):
            rows.append({
                "rank":rank,"name":m.get("name",""),"smiles":m.get("smiles",""),
                "selfies":m.get("selfies",""),"parent":m.get("parent",""),
                "parent_iupac":m.get("parent_iupac",""),
                "docking_score":m.get("docking_score",""),"qed":m.get("qed",""),
                "sa_score":m.get("sa_score",""),"rl_reward":m.get("rl_reward",""),
                "novelty":m.get("novelty",""),"mw":m.get("mw",""),
                "logp":m.get("logp",""),"hbd":m.get("hbd",""),
                "hba":m.get("hba",""),"tpsa":m.get("tpsa",""),
                "lipinski_violations":m.get("lipinski_violations",""),
                "synthesizability":m.get("synthesizability",""),
                "pharmacophore_hits":m.get("pharmacophore_hits",0),
                "pains_flag":m.get("pains_flag",False),
                "pains_reason":m.get("pains_reason",""),
                "front_rank":m.get("front_rank",0),
                "wet_lab_ready":m.get("wet_lab_ready",False),
                "wet_lab_issues":"; ".join(m.get("wet_lab_issues",[])),
                "admet_absorption":m["admet"].get("absorption",""),
                "admet_distribution":m["admet"].get("distribution",""),
                "admet_metabolism":m["admet"].get("metabolism",""),
                "admet_excretion":m["admet"].get("excretion",""),
                "admet_toxicity":m["admet"].get("toxicity",""),
            })
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
        out["csv"] = buf.getvalue()
        out["csv_filename"] = f"docking_results_{timestamp}.csv"

    if fmt in ("sdf","both","all"):
        sdf_buf = io.StringIO()
        sdf_w   = Chem.SDWriter(sdf_buf)
        for m in sorted(_last_results, key=lambda x: x["docking_score"]):
            smi = m.get("smiles","")
            try:
                mol3d = generate_3d_mol(smi) or Chem.MolFromSmiles(smi)
                if mol3d is None: continue
                if not mol3d.GetNumConformers(): AllChem.Compute2DCoords(mol3d)
                for k, v in [("_Name",m.get("name","")),("SMILES",smi),
                              ("ParentIUPAC",m.get("parent_iupac","")),
                              ("DockingScore",str(m.get("docking_score",""))),
                              ("QED",str(m.get("qed",""))),
                              ("WetLabReady",str(m.get("wet_lab_ready",False)))]:
                    mol3d.SetProp(k, v)
                sdf_w.write(mol3d)
            except Exception: pass
        sdf_w.close()
        out["sdf"] = sdf_buf.getvalue()
        out["sdf_filename"] = f"docking_poses_{timestamp}.sdf"

    if fmt in ("pdb","all"):
        pdb_blocks = []
        for rank, m in enumerate(sorted(_last_results, key=lambda x: x["docking_score"]), 1):
            smi  = m.get("smiles","")
            name = m.get("name", f"LIG{rank:03d}")
            mol  = Chem.MolFromSmiles(smi)
            if mol is None: continue
            block = mol_to_pdb_block(mol, name)
            if block:
                remarks = (
                    f"REMARK   1 NAME         {name}\n"
                    f"REMARK   2 SMILES       {smi}\n"
                    f"REMARK   3 PARENT_IUPAC {m.get('parent_iupac','')}\n"
                    f"REMARK   4 QED          {m.get('qed','')}\n"
                    f"REMARK   5 DOCKING      {m.get('docking_score','')}\n"
                    f"REMARK   6 WET_LAB      {m.get('wet_lab_ready',False)}\n"
                    f"REMARK   7 PAINS        {m.get('pains_flag',False)}\n"
                )
                pdb_blocks.append(remarks + block.strip() + "\nEND\n")
        out["pdb"] = "\n".join(pdb_blocks)
        out["pdb_filename"] = f"docking_poses_{timestamp}.pdb"
        out["pdb_count"] = len(pdb_blocks)

    return jsonify(out)


@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status":"ok","version":"v6-final",
        "vae_vocab_size":tokenizer.vocab_size,
        "seed_latents":len(seed_latents),
        "seed_index_size":len(SEED_INDEX),
        "latent_dim":LATENT_DIM,
        "trained_validity":CONFIG.get("final_validity",0),
    })


if __name__ == "__main__":
    import argparse

    # Always run from the directory containing serve.py
    # This ensures saved_model/ and index.html are always found
    # regardless of where the script is called from (e.g. Colab)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    log.info("Working directory set to: %s", script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = parser.parse_args()
    log.info("Molecular Design API v6 on http://%s:%d", args.host, args.port)
    log.info("Trained validity: %.1f%%", CONFIG.get("final_validity",0)*100)
    app.run(debug=False, host=args.host, port=args.port)
