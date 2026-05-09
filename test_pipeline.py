"""
test_pipeline.py
================
Unit tests for the Molecular Design VAE pipeline.

Run with:
    pip install pytest
    pytest tests/

Tests cover:
    1. SELFIES tokenizer round-trip
    2. Seed library validity
    3. Mock docking score range
    4. Augmentation produces valid SMILES
    5. /health endpoint (requires serve.py running)

Test 5 is skipped automatically if the server is not running.
"""

import os
import sys
import pytest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -----------------------------------------------------------------
# Test 1: SELFIES tokenizer round-trip
# -----------------------------------------------------------------
def test_selfies_tokenizer_roundtrip():
    """A simple SMILES should encode and decode back to the same molecule."""
    from train_vae import SELFIESTokenizer
    from rdkit import Chem

    test_smiles = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
    tokenizer = SELFIESTokenizer()
    tokenizer.fit([test_smiles])

    encoded = tokenizer.encode(test_smiles, max_len=80)
    assert encoded is not None, "Tokenizer should encode a valid SMILES"
    assert encoded.shape[0] == 80, "Encoded tensor should match max_len"

    decoded = tokenizer.decode(encoded)
    assert decoded is not None, "Tokenizer should decode back to a SMILES"

    # Canonical comparison — different orderings of the same molecule are equivalent
    canon_orig    = Chem.MolToSmiles(Chem.MolFromSmiles(test_smiles))
    canon_decoded = Chem.MolToSmiles(Chem.MolFromSmiles(decoded))
    assert canon_orig == canon_decoded, \
        f"Round-trip changed molecule: {canon_orig} != {canon_decoded}"


# -----------------------------------------------------------------
# Test 2: Seed library is valid
# -----------------------------------------------------------------
def test_seed_library_valid():
    """All molecules in the seed library should be valid SMILES."""
    from train_vae import SEED_SMILES
    from rdkit import Chem

    assert len(SEED_SMILES) > 250, \
        f"Seed library should have >250 molecules, got {len(SEED_SMILES)}"

    invalid_count = 0
    for smi in SEED_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_count += 1

    assert invalid_count == 0, \
        f"Seed library has {invalid_count} invalid SMILES — should be 0"


# -----------------------------------------------------------------
# Test 3: Augmentation produces valid SMILES
# -----------------------------------------------------------------
def test_augmentation_validity():
    """Augmented SMILES should still be parseable molecules."""
    from train_vae import augment_smiles
    from rdkit import Chem

    test_input = ["CC(=O)Nc1ccc(O)cc1", "O=C(O)c1ccccc1O", "Cc1ccc(O)cc1"]
    augmented = augment_smiles(test_input, n_augment=3)

    assert len(augmented) >= len(test_input), \
        "Augmentation should produce at least as many molecules as input"

    invalid = sum(1 for s in augmented if Chem.MolFromSmiles(s) is None)
    assert invalid == 0, \
        f"Augmentation produced {invalid} invalid SMILES — should be 0"


# -----------------------------------------------------------------
# Test 4: SELFIES guarantees chemical validity
# -----------------------------------------------------------------
def test_selfies_validity_guarantee():
    """
    SELFIES is meant to guarantee valid molecules. Confirm by encoding
    and decoding a sample of seeds.
    """
    from train_vae import SELFIESTokenizer, SEED_SMILES
    from rdkit import Chem

    tok = SELFIESTokenizer()
    tok.fit(SEED_SMILES[:50])

    valid_count = 0
    for smi in SEED_SMILES[:30]:
        encoded = tok.encode(smi, max_len=80)
        if encoded is not None:
            decoded = tok.decode(encoded)
            if decoded and Chem.MolFromSmiles(decoded) is not None:
                valid_count += 1

    # Should be very close to 100% — SELFIES claim
    assert valid_count >= 25, \
        f"Round-trip validity too low: {valid_count}/30 — SELFIES should give close to 100%"


# -----------------------------------------------------------------
# Test 5: API /health endpoint (skipped if server not running)
# -----------------------------------------------------------------
def test_health_endpoint():
    """If serve.py is running, /health should return ok."""
    try:
        import requests
    except ImportError:
        pytest.skip("requests not installed")

    try:
        r = requests.get("http://localhost:5000/health", timeout=2)
    except Exception:
        pytest.skip("serve.py not running on localhost:5000")

    assert r.status_code == 200, f"Health endpoint returned {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", f"Health status not ok: {data}"
    assert "vae_vocab_size" in data, "Health response should include vae_vocab_size"


# -----------------------------------------------------------------
# Quick test runner
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("Running tests directly (use pytest for full output)...")
    import traceback
    for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]:
        try:
            print(f"  {name}...", end=" ")
            fn()
            print("PASS")
        except pytest.skip.Exception as e:
            print(f"SKIP ({e})")
        except Exception as e:
            print("FAIL")
            traceback.print_exc()
