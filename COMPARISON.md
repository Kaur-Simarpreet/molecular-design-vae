# Comparison with Existing Tools

This document compares the Molecular Design VAE pipeline with publicly available tools in de novo drug discovery and protein–ligand modelling.

---

## 1. Feature comparison

| Feature | This project | NVIDIA MolMIM | DiffDock | GNINA | AlphaFold 3 | Boltz-2 | EquiBind |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| De novo molecule generation | ✓ | ✓ | – | – | – | – | – |
| Runs locally on consumer hardware | ✓ | – | – | – | – | – | – |
| CPU-only execution supported | ✓ | – | – | – | – | – | – |
| Open-source code | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| Free, no account required | ✓ | – | ✓ | ✓ | – | ✓ | ✓ |
| Molecular docking scoring | ✓ | ○ | ✓ | ✓ | ○ | ○ | ✓ |
| Multi-objective Pareto (NSGA-II) | ✓ | – | – | – | – | – | – |
| Interactive browser UI with 3D viewer | ✓ | – | – | – | – | – | – |
| SAR / activity cliffs / MMP | ✓ | – | – | – | – | – | – |
| Scaffold hopping | ✓ | – | – | – | – | – | – |
| Retrosynthesis route | ✓ | – | – | – | – | – | – |
| PAINS / ADMET filtering | ✓ | – | – | ○ | – | – | – |
| SDF / PDB export | ✓ | – | ○ | ✓ | ✓ | ✓ | ○ |

✓ supported · ○ partial support · – not supported

---

## 2. Hardware requirements

| Tool | Minimum GPU | Runs on student laptop? | Approximate cost |
|------|-------------|:----------------------:|------------------|
| **This project** | None (CPU sufficient) | Yes | Free |
| NVIDIA MolMIM | A100 / H100 / L40S | No | $2–4/hr cloud or $10k+ hardware |
| DiffDock | RTX 3080+ | Slow | Gaming GPU required |
| GNINA | Any CUDA GPU | Partial | Gaming GPU required |
| AlphaFold 3 | A100 / TPU | No | Cloud only |
| Boltz-2 | GPU recommended | Slow | Gaming GPU required |
| EquiBind | GPU recommended | Slow | Gaming GPU required |

### Note on NVIDIA MolMIM

NVIDIA's BioNeMo NIM documentation lists supported GPUs as A100 (40 GB or 80 GB), H100, H200, B200, GH200, and L40S. These are data-centre accelerators, not consumer or gaming GPUs. The full BioNeMo virtual screening pipeline additionally requires at least 1.3 TB of fast NVMe storage. Running MolMIM also requires an NGC account and API key. The cost is therefore substantial: an A100 80 GB runs $10,000–15,000 used, and cloud rental is $2–4/hr. This places MolMIM outside practical reach of student researchers without HPC infrastructure.

---

## 3. Intended purpose

| Tool | Primary purpose | Does not do |
|------|----------------|-------------|
| **This project** | Generate novel drug-like molecules with a complete analysis platform | Wet-lab validation; structure-based generation |
| NVIDIA MolMIM | Generate molecules at scale on HPC | Run on consumer hardware |
| DiffDock | Predict binding pose without grid box | Generate molecules |
| GNINA | Score docking poses with CNN | Generate molecules |
| AlphaFold 3 | Predict protein–ligand structure | Generate molecules |
| Boltz-2 | Predict biomolecular structures | Generate molecules |
| EquiBind | Predict binding pose efficiently | Generate molecules |

---

## 4. Discussion

The de novo molecular generation space contains two clearly distinguishable categories of tools.

The first category **generates** new molecules. Among publicly available tools this currently includes only NVIDIA MolMIM and the present project. The second category — DiffDock, GNINA, AlphaFold 3, Boltz-2, EquiBind — does not generate molecules; these are docking, scoring, or structure-prediction tools that operate on user-supplied molecules.

The closest comparator is therefore NVIDIA MolMIM. Both tools encode molecules into a continuous latent space and decode novel structures from sampled latent vectors. The key difference is deployment. MolMIM is designed for high-performance computing environments using A100 or H100 accelerators; the present pipeline targets consumer hardware. This trade-off is intentional — MolMIM trains on substantially more data and produces a richer latent space, while the present project trades some diversity for accessibility.

Among the docking-only tools, GNINA and DiffDock could complement rather than replace this pipeline. GNINA's CNN scoring function performs well on novel chemotypes — exactly the molecules this project generates. DiffDock's blind docking is also well-suited as an optional final step for the highest-scoring generated candidates.

The unique contribution of this project is the combination of generative modelling with a complete analysis platform — chemical-space visualisation, structure–activity relationship analysis, activity-cliff detection, matched molecular pair analysis, scaffold hopping, retrosynthesis, and 3D structure viewing — accessible in a single locally-runnable interface.

---

## 5. Sources

- DiffDock: Corso, Stärk, Jing, Barzilay, and Jaakkola (2022)
- GNINA: McNutt et al. (2021)
- AlphaFold 3: Abramson et al. (2024)
- Boltz-2: MIT (2024)
- EquiBind: Stärk et al. (2022)
- NVIDIA BioNeMo Framework documentation, accessed 2026

Comparison reflects publicly available information at the time of writing.
