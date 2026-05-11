"""
docking.py
==========
Unified docking module for the Molecular Design VAE pipeline.

Routes docking calls based on the trained model's mode:

    Mode         Engine           Hardware    Time per molecule
    -----        ------           --------    ------------------
    basic        mock (RDKit)     CPU         instant
    moderate     mock (RDKit)     CPU         instant
    large        AutoDock Vina    CPU         ~25 sec
    large-gpu    DiffDock         GPU         ~30 sec (or 5-10 min CPU)

Falls back gracefully to mock docking if the real engine isn't installed.
The pipeline never breaks — it just downgrades scoring quality with a warning.

Target
------
EGFR Kinase, PDB ID 1IEP (Imatinib-bound). Open-source PDB structure,
no account needed. Active site box is the Imatinib binding pocket.

Setup
-----
    For Vina  (large mode):    bash setup_docking.sh --vina
    For DiffDock (large-gpu):  bash setup_docking.sh --diffdock

Both setup commands download the PDB structure and configure the receptor
once. After setup, real docking happens automatically when you run the
matching training mode.

Integration
-----------
serve.py imports `dock_for_mode()` and calls it instead of `mock_docking()`.
The mode is read from saved_model/config.json which the training scripts
write. No other changes to serve.py needed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors

log = logging.getLogger(__name__)

# ================================================================
# TARGET CONFIGURATION — EGFR Kinase (PDB 1IEP)
# Imatinib binding pocket coordinates
# ================================================================
EGFR_TARGET = {
    "pdb_id":      "1IEP",
    "name":        "EGFR Kinase",
    "pdb_file":    "receptors/1IEP.pdb",
    "pdbqt_file":  "receptors/1IEP.pdbqt",
    # Active site centre (Imatinib binding pocket)
    "center_x":    16.5,
    "center_y":    50.5,
    "center_z":    35.0,
    # Search box size (Angstroms)
    "size_x":      22.0,
    "size_y":      22.0,
    "size_z":      22.0,
}

# Cache of docking results — same molecule docked twice returns instantly
_DOCKING_CACHE: dict = {}


# ================================================================
# MAIN ENTRY POINT — called by serve.py
# ================================================================

def dock_for_mode(smiles: str, mode: str, target_name: str = "EGFR kinase") -> float:
    """
    Dock a molecule using the engine appropriate for the given mode.

    Args:
        smiles:      SMILES string of the molecule
        mode:        Training mode from config.json — "basic", "moderate",
                     "large", or "large-gpu"
        target_name: Target protein name (currently only EGFR is configured)

    Returns:
        Docking score in kcal/mol (more negative = stronger binding)
        Falls back to mock score if the real engine isn't available.
    """
    cache_key = (smiles, mode)
    if cache_key in _DOCKING_CACHE:
        return _DOCKING_CACHE[cache_key]

    if mode == "large":
        score = dock_with_vina(smiles)
    elif mode == "large-gpu":
        score = dock_with_diffdock(smiles)
    else:
        # basic, moderate — mock docking only
        score = mock_docking(smiles, target_name)

    _DOCKING_CACHE[cache_key] = score
    return score


# ================================================================
# MOCK DOCKING — RDKit-based estimate
# Used for basic and moderate modes, and as fallback for failures
# ================================================================

def mock_docking(smiles: str, target_name: str = "EGFR kinase") -> float:
    """
    Computational estimate of binding affinity based on molecular properties.
    NOT real protein docking — used when real docking isn't set up.

    Returns: estimated score in kcal/mol (range typically -3 to -12)
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return -5.0

    try:
        mw       = Descriptors.MolWt(mol)
        logp     = Descriptors.MolLogP(mol)
        rings    = rdMolDescriptors.CalcNumAromaticRings(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        hbd      = rdMolDescriptors.CalcNumHBD(mol)
        hba      = rdMolDescriptors.CalcNumHBA(mol)

        # Empirical scoring inspired by drug-likeness rules
        score = -5.0
        if 250 <= mw <= 450:
            score -= 1.5
        if 1.5 <= logp <= 4.5:
            score -= 1.0
        score -= 0.4 * rings
        score -= 0.1 * min(hba, 8)
        score += 0.2 * max(0, rotbonds - 8)
        # Add deterministic noise based on SMILES so same molecule always
        # gets the same score
        seed = sum(ord(c) for c in smiles) % 1000
        np.random.seed(seed)
        score += np.random.uniform(-1.0, 1.0)
        np.random.seed(None)

        return float(np.clip(score, -12.0, -2.0))
    except Exception as e:
        log.debug("mock_docking failed for %s: %s", smiles[:30], e)
        return -5.0


# ================================================================
# REAL DOCKING — AutoDock Vina (used by Large mode)
# ================================================================

def is_vina_available() -> bool:
    """Check whether Vina is installed and the receptor PDBQT exists."""
    try:
        import vina  # noqa: F401
        from meeko import MoleculePreparation  # noqa: F401
    except ImportError:
        return False
    if not os.path.exists(EGFR_TARGET["pdbqt_file"]):
        return False
    return True


def _prepare_ligand_pdbqt(smiles: str, output_path: str) -> bool:
    """Convert SMILES to PDBQT for Vina input."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        # 3D conformer
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=20) != 0:
            return False
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        prep = MoleculePreparation()
        prep.prepare(mol)
        pdbqt_string = PDBQTWriterLegacy.write_string(prep.setup)
        with open(output_path, 'w') as f:
            f.write(pdbqt_string)
        return True
    except Exception as e:
        log.debug("Ligand prep failed for %s: %s", smiles[:30], e)
        return False


def dock_with_vina(smiles: str) -> float:
    """
    Real molecular docking with AutoDock Vina against EGFR 1IEP.
    Falls back to mock_docking if Vina or receptor isn't set up.

    Returns: best binding affinity in kcal/mol
    """
    if not is_vina_available():
        log.warning("Vina or receptor not available — falling back to mock score. "
                    "Run 'bash setup_docking.sh --vina' to enable real docking.")
        return mock_docking(smiles)

    try:
        from vina import Vina

        # Prepare ligand
        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_path = os.path.join(tmpdir, "ligand.pdbqt")
            if not _prepare_ligand_pdbqt(smiles, ligand_path):
                log.warning("Ligand prep failed for %s — using mock score", smiles[:40])
                return mock_docking(smiles)

            # Run Vina
            v = Vina(sf_name='vina', cpu=2, verbosity=0)
            v.set_receptor(EGFR_TARGET["pdbqt_file"])
            v.set_ligand_from_file(ligand_path)
            v.compute_vina_maps(
                center=[EGFR_TARGET["center_x"], EGFR_TARGET["center_y"], EGFR_TARGET["center_z"]],
                box_size=[EGFR_TARGET["size_x"], EGFR_TARGET["size_y"], EGFR_TARGET["size_z"]],
            )
            v.dock(exhaustiveness=8, n_poses=5)
            energies = v.energies(n_poses=1)
            score = float(energies[0][0])  # best pose, total energy
            log.debug("Vina dock %s = %.2f kcal/mol", smiles[:30], score)
            return score
    except Exception as e:
        log.warning("Vina docking failed for %s: %s — using mock", smiles[:40], e)
        return mock_docking(smiles)


# ================================================================
# REAL DOCKING — DiffDock (used by Large GPU mode)
# ================================================================

def is_diffdock_available() -> bool:
    """Check whether DiffDock is installed and configured."""
    diffdock_path = os.environ.get("DIFFDOCK_PATH", "DiffDock")
    if not os.path.isdir(diffdock_path):
        return False
    inference_script = os.path.join(diffdock_path, "inference.py")
    if not os.path.exists(inference_script):
        return False
    if not os.path.exists(EGFR_TARGET["pdb_file"]):
        return False
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


def dock_with_diffdock(smiles: str) -> float:
    """
    Real molecular docking with DiffDock (deep-learning blind docking).
    Falls back to mock_docking if DiffDock isn't set up.

    Returns: best binding affinity in kcal/mol
    """
    if not is_diffdock_available():
        log.warning("DiffDock not available — falling back to mock score. "
                    "Run 'bash setup_docking.sh --diffdock' to enable.")
        return mock_docking(smiles)

    try:
        diffdock_path = os.environ.get("DIFFDOCK_PATH", "DiffDock")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write SMILES file
            smiles_csv = os.path.join(tmpdir, "ligands.csv")
            with open(smiles_csv, 'w') as f:
                f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
                f.write(f"egfr_lig,{os.path.abspath(EGFR_TARGET['pdb_file'])},{smiles},\n")

            # Run DiffDock inference
            output_dir = os.path.join(tmpdir, "out")
            cmd = [
                sys.executable,
                os.path.join(diffdock_path, "inference.py"),
                "--protein_ligand_csv", smiles_csv,
                "--out_dir", output_dir,
                "--inference_steps", "20",
                "--samples_per_complex", "10",
                "--batch_size", "6",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                    cwd=diffdock_path)
            if result.returncode != 0:
                log.warning("DiffDock failed for %s: %s — using mock",
                            smiles[:40], result.stderr[:200])
                return mock_docking(smiles)

            # Parse results — DiffDock outputs ranked SDF files
            # Confidence score from rank 1 -> approximate kcal/mol mapping
            complex_dir = os.path.join(output_dir, "egfr_lig")
            if not os.path.isdir(complex_dir):
                return mock_docking(smiles)

            # Find rank 1 result and read confidence
            rank1_files = [f for f in os.listdir(complex_dir) if f.startswith("rank1_")]
            if not rank1_files:
                return mock_docking(smiles)

            # DiffDock confidence: c = log(p / (1-p)). Map to estimated kcal/mol
            # using empirical relation: score ≈ -7 - 1.5 * confidence
            confidence_str = rank1_files[0].split("confidence")[-1].split(".sdf")[0]
            try:
                confidence = float(confidence_str)
                estimated_kcal = -7.0 - 1.5 * confidence
                return float(np.clip(estimated_kcal, -14.0, -3.0))
            except ValueError:
                return mock_docking(smiles)
    except subprocess.TimeoutExpired:
        log.warning("DiffDock timed out for %s — using mock", smiles[:40])
        return mock_docking(smiles)
    except Exception as e:
        log.warning("DiffDock error for %s: %s — using mock", smiles[:40], e)
        return mock_docking(smiles)


# ================================================================
# DIAGNOSTIC HELPER — used by setup_docking.sh
# ================================================================

def check_status() -> dict:
    """Report which docking backends are available."""
    return {
        "vina_installed":        _check_module("vina"),
        "meeko_installed":       _check_module("meeko"),
        "diffdock_installed":    is_diffdock_available(),
        "torch_geometric":       _check_module("torch_geometric"),
        "pdb_file":              os.path.exists(EGFR_TARGET["pdb_file"]),
        "pdbqt_file":            os.path.exists(EGFR_TARGET["pdbqt_file"]),
    }


def _check_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Quick diagnostic when run directly
    print("Docking module status:")
    for k, v in check_status().items():
        print(f"  {k}: {'YES' if v else 'NO'}")
    print()
    test_smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    print(f"Test molecule: {test_smi}")
    print(f"  Mock score:        {mock_docking(test_smi):.2f} kcal/mol")
    if is_vina_available():
        print(f"  Vina score:        {dock_with_vina(test_smi):.2f} kcal/mol")
    if is_diffdock_available():
        print(f"  DiffDock score:    {dock_with_diffdock(test_smi):.2f} kcal/mol")
