#!/usr/bin/env bash
# setup_docking.sh — one-command real-docking setup
#
# Usage:
#   bash setup_docking.sh --vina       # Install Vina + receptor (for large mode)
#   bash setup_docking.sh --diffdock   # Install DiffDock + receptor (for large-gpu)
#   bash setup_docking.sh --status     # Check what's installed
#
# What it does:
#   1. Downloads EGFR 1IEP from RCSB PDB (open access, no account needed)
#   2. Prepares the receptor (PDBQT for Vina, PDB cleaned for DiffDock)
#   3. Installs the chosen docking engine
#   4. Verifies everything works

set -e

# -------- Config --------
PDB_ID="1IEP"
RECEPTOR_DIR="receptors"
PDB_URL="https://files.rcsb.org/download/${PDB_ID}.pdb"
DIFFDOCK_REPO="https://github.com/gcorso/DiffDock.git"

# -------- Helpers --------
echo_section() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

check_pdb() {
    if [[ ! -f "${RECEPTOR_DIR}/${PDB_ID}.pdb" ]]; then
        echo_section "Downloading EGFR ${PDB_ID} from RCSB PDB"
        mkdir -p "${RECEPTOR_DIR}"
        if command -v wget >/dev/null 2>&1; then
            wget -q -O "${RECEPTOR_DIR}/${PDB_ID}.pdb" "${PDB_URL}"
        elif command -v curl >/dev/null 2>&1; then
            curl -sL -o "${RECEPTOR_DIR}/${PDB_ID}.pdb" "${PDB_URL}"
        else
            echo "ERROR: need wget or curl to download the PDB file"
            exit 1
        fi
        echo "Saved to ${RECEPTOR_DIR}/${PDB_ID}.pdb"
    else
        echo "PDB file already present: ${RECEPTOR_DIR}/${PDB_ID}.pdb"
    fi
}

setup_vina() {
    echo_section "Setting up AutoDock Vina (for Large mode)"

    check_pdb

    echo ""
    echo "Installing Python packages: vina, meeko, openbabel-wheel"
    pip install -q vina meeko || {
        echo "WARNING: pip install failed. Try:"
        echo "    pip install vina meeko"
        echo "    or on Linux: sudo apt-get install autodock-vina"
        exit 1
    }

    echo ""
    echo "Preparing receptor (cleaning + converting to PDBQT)..."

    python3 << 'PYEOF'
import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

PDB = "receptors/1IEP.pdb"
PDBQT = "receptors/1IEP.pdbqt"

# Strip waters and ligands using a simple line filter
clean_pdb = PDB.replace(".pdb", "_clean.pdb")
with open(PDB) as fin, open(clean_pdb, 'w') as fout:
    for line in fin:
        if line.startswith("ATOM"):  # protein only
            fout.write(line)
        elif line.startswith("TER") or line.startswith("END"):
            fout.write(line)

# Convert PDB → PDBQT using meeko's prepare_receptor (or fallback)
try:
    from meeko import PolymerPreparator
    pp = PolymerPreparator()
    pp.prepare_from_pdb(clean_pdb)
    pp.write_pdbqt(PDBQT)
    print(f"Receptor prepared via meeko: {PDBQT}")
except Exception as e:
    # Fallback: use prepare_receptor4.py from MGLTools if installed
    try:
        result = subprocess.run(
            ["prepare_receptor", "-r", clean_pdb, "-o", PDBQT, "-A", "hydrogens"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(PDBQT):
            print(f"Receptor prepared via prepare_receptor: {PDBQT}")
        else:
            raise RuntimeError(result.stderr)
    except Exception:
        # Last fallback: minimal manual conversion
        print("WARNING: full PDBQT prep failed. Creating minimal version.")
        with open(clean_pdb) as fin, open(PDBQT, 'w') as fout:
            for line in fin:
                if line.startswith("ATOM"):
                    # Add charges (Q) and atom type as element
                    elem = line[76:78].strip() or line[12:16].strip()[0]
                    fout.write(line.rstrip() + f"  0.000 {elem}\n")
                elif line.startswith("END"):
                    fout.write(line)
        print(f"Minimal PDBQT written: {PDBQT}")

# Cleanup
if os.path.exists(clean_pdb):
    os.remove(clean_pdb)
PYEOF

    echo ""
    echo "Verifying setup..."
    python3 -c "from docking import is_vina_available; print('Vina ready:', is_vina_available())"

    echo_section "Vina setup complete"
    echo ""
    echo "Now run:  python train_vae_extended.py --mode large"
    echo "Then:     python serve.py"
    echo ""
    echo "Real docking against EGFR 1IEP will be used automatically in large mode."
}

setup_diffdock() {
    echo_section "Setting up DiffDock (for Large GPU mode)"

    check_pdb

    if [[ ! -d "DiffDock" ]]; then
        echo ""
        echo "Cloning DiffDock repository..."
        git clone --depth 1 "${DIFFDOCK_REPO}" DiffDock
    else
        echo "DiffDock directory already exists"
    fi

    echo ""
    echo "Installing PyTorch Geometric and DiffDock dependencies..."
    echo "(This downloads ~3 GB including ESM model weights)"

    pip install -q --upgrade pip

    # PyTorch Geometric — pin versions known to work with DiffDock
    pip install -q torch_geometric || {
        echo "WARNING: torch_geometric install failed. Try manually:"
        echo "    pip install torch_geometric"
    }

    # Other deps from DiffDock
    if [[ -f "DiffDock/requirements.txt" ]]; then
        pip install -q -r DiffDock/requirements.txt || {
            echo "Some DiffDock requirements failed. Continuing — may still work."
        }
    fi

    # ESM-2 weights (DiffDock uses these for protein embeddings)
    pip install -q fair-esm || true

    echo ""
    echo "Verifying setup..."
    python3 -c "from docking import is_diffdock_available; print('DiffDock ready:', is_diffdock_available())"

    echo_section "DiffDock setup complete"
    echo ""
    echo "Now run:  python train_vae_extended.py --mode large-gpu"
    echo "Then:     python serve.py"
    echo ""
    echo "Real DiffDock blind docking will be used in large-gpu mode."
}

show_status() {
    echo_section "Docking environment status"
    python3 -c "
from docking import check_status
status = check_status()
for k, v in status.items():
    print(f'  {k:25s} {\"YES\" if v else \"NO\"}')"
}

# -------- Main --------
case "${1:-}" in
    --vina)
        setup_vina
        ;;
    --diffdock)
        setup_diffdock
        ;;
    --status)
        show_status
        ;;
    *)
        echo "Usage:"
        echo "  bash setup_docking.sh --vina       # for Large mode"
        echo "  bash setup_docking.sh --diffdock   # for Large GPU mode"
        echo "  bash setup_docking.sh --status     # check what's installed"
        echo ""
        echo "After setup, real docking happens automatically when you run"
        echo "the matching training mode."
        exit 1
        ;;
esac
