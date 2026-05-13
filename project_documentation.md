# Molecular Design VAE

**How We Built It, What We Used, and Where It's Going**

Simarpreet Kaur (SR No: 26727) · M. Bhagya Sri (SR No: 27003)
Generative AI for Research · 2026
[github.com/Kaur-Simarpreet/molecular-design-vae](https://github.com/Kaur-Simarpreet/molecular-design-vae)

---

## 1. What Is This Project?

We built a tool that generates new drug-like molecules from scratch. You type in a target — say, "EGFR kinase" — and the system produces novel chemical structures that are likely to bind that target, are drug-like, and are reasonably easy to synthesise.

The whole thing runs on a student laptop. No GPU needed for the basic version, no cloud account, no expensive software. Just Python, a bit of patience, and about 8 minutes of training time.

We built four versions of the same tool, each trained on progressively more data, producing better and more diverse molecules at the cost of longer training time.

| Mode | Dataset | Training Time | Docking | Where it runs |
|------|---------|--------------|---------|---------------|
| Basic | 295 hand-picked seeds | 5–8 min (CPU) | Estimated scores | Hosted app + local |
| Moderate | ~25,000 ZINC molecules | 60–90 min (CPU) | Estimated scores | Local or Colab |
| Large | ~100,000 ZINC molecules | 4–6 hours (CPU) | Real Vina docking | Local (run overnight) |
| Large GPU | ~100,000 ZINC molecules | 20–40 min (GPU) | Real DiffDock | Colab T4 or local GPU |

---

## 2. SELFIES — How We Represent Molecules

Before we can train a model to generate molecules, we need a way to represent them as text. The obvious choice is SMILES — the standard notation used everywhere in cheminformatics. But SMILES has a serious problem for generative models: most random SMILES strings are chemically invalid. If the model generates even one wrong character, the whole molecule breaks.

We use SELFIES instead. SELFIES (Self-Referencing Embedded Strings, Krenn et al. 2020) is a representation where every possible string — no matter what — decodes to a valid molecule. The grammar is designed so you literally cannot write an invalid SELFIES. This is a huge deal for a VAE, because the decoder can generate freely without worrying about producing garbage.

> **Why this matters in practice:** With SMILES, a VAE might generate 40–60% valid molecules. With SELFIES, we get 95–100% validity by construction. That means the model spends its capacity learning meaningful chemistry rather than learning to avoid syntax errors. Our basic mode achieves 100% validity rate on the training set.

### How the tokeniser works

Each SELFIES symbol — like `[C]`, `[=O]`, `[N]`, `[Branch1]` — becomes one integer token. We build the vocabulary from the training molecules, which gives us 28 tokens for basic mode and up to 52+ tokens for extended modes. Sequences are padded to a fixed length with a `[nop]` (no-operation) token.

We also use atom-reorder augmentation: the same molecule can be written in different orderings, and each ordering is a valid training example. This multiplies our training data without adding new molecules.

```
# One molecule, three valid SELFIES encodings (all decode to the same molecule):
[C][C](=[O])[N][c1cc][c][c][c1]   ← original ordering
[N][C](=[O])[C][c1cc][c][c][c1]   ← start from nitrogen
[c1cc][c][c][c1][N][C](=[O])[C]   ← start from ring

# All three go into the training set — 3x data from 1 molecule
```

---

## 3. The Beta-VAE — The Heart of the System

The VAE is what learns the drug-like chemical space. Think of it as learning a map of molecule-land: similar molecules end up near each other, and you can navigate the map to find new molecules with desired properties. The "latent space" is that map — a compressed representation where every point corresponds to a molecule.

### 3.1 Encoder

The encoder reads a SELFIES token sequence and compresses it down to a point in the latent space. It doesn't output a single point — it outputs a probability distribution (a mean and a variance), which is what makes it a variational autoencoder rather than a regular one.

```
Input:   SELFIES token sequence  [t1, t2, ..., t80]
Embed:   Each token → 64-dimensional vector
Encode:  Bidirectional LSTM reads the sequence both ways
Output:  μ (mean vector, dim 128)  ← "where this molecule lives"
         σ² (variance, dim 128)   ← "how uncertain we are"
```

### 3.2 Reparameterisation Trick

We need to sample a latent point z from that distribution, but sampling is not differentiable — gradients can't flow through it. The reparameterisation trick fixes this by reformulating the sample as a deterministic function plus noise:

```
z = μ + σ × ε     where ε ~ N(0, I)   (random noise)

Now z is differentiable with respect to μ and σ.
The randomness comes from ε, which has no trainable parameters.
```

This is one of the clever little tricks that makes VAEs trainable at all. It looks like a minor implementation detail but it's actually the core insight that the original VAE paper (Kingma & Welling 2014) was built around.

### 3.3 Decoder

The decoder takes the latent point z and reconstructs the original SELFIES sequence token by token. It's autoregressive — each token is predicted using the previous one. During training we use teacher forcing (feed the true previous token, not the predicted one) because it speeds up learning dramatically. We gradually reduce teacher forcing over training so the model learns to work on its own.

```
Teacher forcing ratio schedule:
  Epoch 1:   100% teacher forcing   ← training wheels fully on
  Epoch 75:  62.5% teacher forcing  ← training wheels loosened
  Epoch 150: 50% teacher forcing    ← minimum, model mostly self-reliant
```

### 3.4 Training Objective (ELBO)

The model is trained to maximise the Evidence Lower Bound — a balance between two things: reconstructing the molecule well, and keeping the latent space well-organised. If we only optimised reconstruction, the latent space would be a mess and navigation would be impossible. If we only optimised the latent space, the reconstructions would be bad.

```
ELBO = Reconstruction quality - β × KL divergence

Reconstruction quality = cross-entropy between predicted and true tokens
KL divergence          = how far the latent distribution is from N(0, I)
β                      = how much we weight the regularisation (0.15–0.20)
```

We use β < 1, which means we care more about reconstruction quality than latent space regularity. This is a conscious choice — for drug discovery, we want the decoder to be accurate.

### 3.5 Model Sizes

| Setting | Basic / Moderate | Large / Large-GPU |
|---------|-----------------|-------------------|
| Latent dimension | 128 | 256 |
| LSTM hidden units | 256 | 512 |
| Embedding dimension | 64 | 128 |
| Max sequence length | 80 SELFIES tokens | 100 SELFIES tokens |
| Total parameters | ~3.7 million | ~4.9 million |

---

## 4. Cyclic KL Annealing — Keeping the Latent Space Alive

There's a nasty failure mode in VAE training called posterior collapse. It happens when the encoder just ignores the input and outputs the prior distribution N(0,I) for everything. The KL term goes to zero (the model is "happy"), but the latent space contains no information about the molecule. The decoder has to reconstruct without any signal, so it just memorises the most common output.

If posterior collapse happens, our tool is broken — every generation call produces the same molecule regardless of what target you typed in.

> **How we prevent it:** We use cyclic cosine annealing for the KL weight β (Fu et al. 2019). Instead of a fixed β, we repeatedly cycle between β = 0 (reconstruction phase — model learns to use the latent space) and β = β_max (regularisation phase — latent space gets organised). This prevents the encoder from collapsing while still producing a structured latent space.

```python
# Warmup: β ramps up from 0
if epoch < warmup_epochs:
    β = β_max × (epoch / warmup_epochs)

# Cyclic cosine: β oscillates between 0 and β_max
else:
    t = (epoch - warmup_epochs) % cycle_len
    β = β_max × 0.5 × (1 - cos(π × t / cycle_len))
```

| Mode | β_max | Warmup | Cycle length | Total cycles |
|------|-------|--------|-------------|-------------|
| Basic | 0.20 | 20 epochs | 30 epochs | ~4.3 cycles |
| Moderate | 0.18 | 30 epochs | 40 epochs | ~5.5 cycles |
| Large | 0.15 | 40 epochs | 50 epochs | ~7.2 cycles |

---

## 5. Reinforcement Learning — Current and Future

### 5.1 Why We Need RL at All

A VAE trained only on reconstruction loss is a great mimic — it learns to reproduce the training distribution. But it doesn't know anything about drug-likeness, binding affinity, or synthesisability. If you just sample randomly from the latent space, you get drug-like molecules about as often as the training set did — which might not be good enough.

Reinforcement learning adds a reward signal that guides the model toward better molecules. Instead of just sampling randomly, we actively search the latent space for points that decode to high-scoring molecules.

### 5.2 What We Actually Built (Honest Version)

> **We did not implement PPO.** The optimisation component is called "Reward-Guided Latent Space Exploration" in our code and documentation — not PPO. It is a directed hill-climbing procedure. We made this naming change deliberately because calling it PPO would be inaccurate. PPO requires a policy network, a value network, a rollout buffer, and advantage estimation. None of those exist in the current version.

What we actually do is hill-climbing in the latent space with a reward signal. Here's the algorithm in plain English:

- Start at a seed molecule's latent vector z
- Add a small random noise vector to get a new point z'
- Decode z' to a molecule and score it
- If the score improved, move there. If not, try again.
- After 5 failed attempts, jump to a different seed molecule and try again
- Repeat for RL_EPOCHS iterations, keep all the good molecules found

The noise gets smaller over time — early on we explore broadly, later we refine locally.

```
# Noise schedule: big early, small late
σ(epoch) = σ_max × 0.5 × (1 + cos(π × epoch / RL_EPOCHS)) + σ_min

σ_max = 0.8   ← broad exploration at the start
σ_min = 0.1   ← fine refinement at the end
```

### 5.3 The Reward Function

The reward is a weighted sum of five property scores. All scores are normalised to [0, 1] so they're comparable. The weights change based on what objective the user selects from the dropdown.

```
reward = w_qed   × QED(molecule)
       + w_dock  × normalised_docking_score
       + w_sa    × (1 - SA_score/10)
       + w_nov   × novelty_vs_seed_library
       + w_admet × admet_pass
```

| Objective | QED | Docking | SA | Novelty | ADMET |
|-----------|-----|---------|-----|---------|-------|
| Binding affinity | 30% | 40% | 15% | 5% | 10% |
| Multi-objective | 30% | 25% | 15% | 15% | 15% |
| ADMET optimised | 30% | 15% | 10% | 5% | 40% |
| Novelty focused | 25% | 20% | 10% | 35% | 10% |
| Synthesisability | 25% | 15% | 45% | 5% | 10% |

### 5.4 What It Does Well and What It Doesn't

**It works well because:**
- Runs on CPU with no extra training required — just runs on top of the frozen VAE
- Consistently improves mean reward scores compared to pure random sampling
- Simple enough to explain to anyone: decode, score, keep the good ones

**It doesn't work well because:**
- It has no memory across runs — every generation starts from scratch
- It can get stuck in local optima even with random restarts
- Knowledge from one target (EGFR) doesn't transfer to another target (GPCR)
- The exploration is random, not learned — it doesn't know which directions are productive

### 5.5 Future Plan: Real PPO

Proximal Policy Optimisation (Schulman et al. 2017) is what we want to upgrade to. The difference is that PPO trains an actual neural network (the policy) that learns which directions in latent space lead to better molecules. Instead of random noise, the policy makes intelligent moves.

The three things PPO needs that we don't currently have:
- **Policy network πθ:** takes the current latent vector z and outputs a direction to move in
- **Value network Vφ:** estimates how good a latent position is (expected future reward from here)
- **PPO update:** trains both networks using a clipped objective that prevents destabilising updates

```
# PPO clipped objective (the key equation):
L = -min(
    r_t × A_t,
    clip(r_t, 1-ε, 1+ε) × A_t
)

where r_t = new_policy / old_policy   (how much the policy changed)
      A_t = advantage (was this move better or worse than expected?)
      ε   = 0.2  (don't let the policy change too dramatically at once)
```

The clipping is the key innovation of PPO over earlier policy gradient methods. It stops the policy from making huge jumps that destabilise training.

| Aspect | Current (Hill-Climbing) | Future (PPO) |
|--------|------------------------|-------------|
| Algorithm | Directed random walk | Learned policy network |
| Policy network | None | MLP: z → action Δz |
| Value network | None | MLP: z → scalar V |
| Learning across runs | No — starts fresh every time | Yes — policy improves |
| Target transfer | None | Partial generalisation |
| Training overhead | Zero | ~2–3 weeks to implement + train |
| Expected reward gain | Baseline | +15–25% (per REINVENT paper) |
| Current status | Done and running | Planned future work |

Why didn't we implement PPO now? Honestly, time. A correct PPO implementation needs a policy network, value network, rollout buffer, advantage estimation, reward shaping, and careful hyperparameter tuning. That's 2–3 weeks of focused work on top of everything else in this project. Hill-climbing is effective enough for what we need right now, and it's honest — we document exactly what it is.

---

## 6. NSGA-II — Ranking Molecules When Nothing Is Perfect

Here's the problem: we have 4 objectives (QED, docking, SA, novelty) and they conflict. A molecule with great docking might have terrible SA. A highly novel molecule might have mediocre QED. If we collapse everything into one score, we lose information about these trade-offs.

NSGA-II (Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002) handles this properly. Instead of one ranked list, it produces fronts — groups of molecules that represent different positions on the trade-off curve.

### How the ranking works

A molecule A "dominates" molecule B if A is at least as good as B on every objective and strictly better on at least one. NSGA-II sorts molecules into fronts based on domination:

- **Front 1 (Pareto optimal):** no other molecule dominates these. These are your best candidates.
- **Front 2:** dominated by at least one molecule in Front 1, but not by anything else.
- **Front 3 and beyond:** progressively worse trade-offs.

Within each front, molecules are ranked by crowding distance — how different they are from their neighbours on the trade-off curve. This preserves diversity and stops the ranking from collapsing to a single point in objective space.

In our UI, the molecule cards are colour-coded by Pareto front. Front 1 molecules get the best visual treatment. The user can see at a glance which molecules represent the best trade-offs.

---

## 7. The 8-Stage Quality Filter

After generation and NSGA-II ranking, every molecule passes through 8 sequential filters before the user sees it. This is the gatekeeping layer — SELFIES guarantees the molecule is chemically valid, but these filters check that it's actually drug-like and useful.

About 50% of generated molecules pass all 8 filters. The rest are valid but not drug-like enough to be worth showing.

| Stage | Filter | What it checks | Why it matters |
|-------|--------|---------------|----------------|
| 1 | Molecular weight | 150–500 Da | Too small = not complex enough. Too large = poor bioavailability. |
| 2 | Tanimoto diversity | Not too similar to existing results | Keeps the output chemically diverse. |
| 3 | Lipinski Rule of Five | MW≤500, LogP≤5, HBD≤5, HBA≤10 | The classic drug-likeness filter from 1997. Still industry standard. |
| 4 | QED minimum | QED > 0.25 | Removes molecules with very poor overall drug-likeness. |
| 5 | PAINS filter | No pan-assay interference | PAINS compounds look active in assays but aren't real hits. |
| 6 | ADMET screening | Absorption, distribution, metabolism, excretion, toxicity | Basic pharmacokinetic viability. |
| 7 | Novelty filter | Not identical to seed molecules | We're generating new drugs, not reproducing training data. |
| 8 | Wet-lab viability | No highly reactive groups | Removes molecules that would be unstable or dangerous to handle. |

---

## 8. Molecular Docking — What's Real, What's Estimated

Docking is how we estimate whether a molecule will bind to a target protein. It's the bridge between "this molecule looks drug-like" and "this molecule might actually work against EGFR kinase".

We're honest about this in the documentation: basic and moderate modes use estimated docking scores, not real protein docking. Large and Large-GPU modes use real docking against a real PDB structure.

### Basic and Moderate: Estimated Scores

The estimated score is calculated from molecular properties using an empirical formula. It's fast (instant), consistent (same molecule always gets the same score), and useful for ranking molecules within a batch. But it's not real docking — we don't actually place the molecule inside the protein pocket and calculate binding energy.

```python
# Simplified mock scoring (from serve.py):
score = -5.0
if 250 <= MW <= 450:     score -= 1.5   # drug-like MW bonus
if 1.5 <= LogP <= 4.5:   score -= 1.0   # good lipophilicity bonus
score -= 0.4 × n_aromatic_rings          # more rings = better docking estimate
score += 0.2 × max(0, rotbonds - 8)     # too flexible = penalty
score += uniform(-1.0, 1.0)             # deterministic noise from SMILES hash
```

### Large: Real AutoDock Vina

AutoDock Vina (Trott & Olson 2010) is the gold standard for fast molecular docking. It places the molecule in 3D, rotates it in the protein's binding pocket, and calculates the binding energy using a force field. We dock against EGFR kinase crystal structure 1IEP (Imatinib-bound, from RCSB PDB).

- Active site centre: (16.5, 50.5, 35.0) Å — the Imatinib binding pocket
- Search box: 22 × 22 × 22 Å
- Time: ~25 seconds per molecule on CPU
- Output: binding affinity in kcal/mol (more negative = stronger binding)

### Large-GPU: Real DiffDock

DiffDock (Corso et al. 2022) is a deep learning docking model that uses diffusion to predict binding poses without needing a predefined grid box — it's called blind docking. This is state-of-the-art for novel molecules that haven't been seen during training. It's better for the kinds of molecules our VAE generates precisely because they're novel.

- No grid box needed — DiffDock predicts where binding happens
- Time: ~30 seconds per molecule on GPU
- Output: confidence score mapped to estimated kcal/mol
- Requires: EGFR 1IEP PDB structure, torch_geometric, ESM-2 model weights (~3 GB)

| Mode | Engine | Type | Time per molecule | Realistic? |
|------|--------|------|------------------|-----------|
| Basic | RDKit estimate | Property-based formula | Instant | Useful for ranking only |
| Moderate | RDKit estimate | Property-based formula | Instant | Useful for ranking only |
| Large | AutoDock Vina | Real force-field docking | ~25 sec | Yes — real binding energies |
| Large-GPU | DiffDock | Deep learning blind docking | ~30 sec | Yes — state-of-the-art |

---

## 9. The Analysis Platform — 8 Tabs

The web frontend is built in pure HTML/CSS/JavaScript — no React, no Vue, no build step. It communicates with the Flask API via fetch calls. Here's what each tab does and why it's useful:

| Tab | What it does | Why it matters |
|-----|-------------|----------------|
| Generate | De novo molecule generation with reward-guided RL | The main pipeline. Start here. |
| Optimize Molecule | Paste any SMILES, get improved analogues via latent space perturbation | Useful when you have a lead compound and want to improve it. |
| Scaffold Hop | Keep the core scaffold, explore different substituents | Classic med-chem strategy: change the 'frame' while preserving the binding mode. |
| Chemical Space | 2D PCA of generated molecules, coloured by score | Visualises chemical diversity. Spread-out clouds = diverse output. |
| SAR Heatmap | Which functional groups correlate with better scores | Structure-Activity Relationship: tells you what's driving activity. |
| Activity Cliffs | Pairs of similar molecules with very different scores | Where the interesting biology is. Small structural change = big potency change. |
| MMP Analysis | Matched Molecular Pairs — single-bond changes and their effect | The most rigorous SAR analysis. Used in industry drug discovery. |
| Retrosynthesis | Suggested synthetic routes for generated molecules | Checks whether the molecule can actually be made in a lab. |

The BACKEND URL is defined in the HTML head so it's available before any button is clicked. It auto-detects whether the app is running locally or on Hugging Face Spaces and connects accordingly.

---

## 10. The Dataset

### Curated Seed Library

We hand-picked 295 drug-like molecules from ChEMBL 34 and ZINC-250K to cover 12 scaffold families: kinase inhibitors, GPCR ligands, protease inhibitors, amino acid derivatives, heterocyclics, urea/hydroxamic acids, sulfonamides, fluorinated compounds, morpholine/piperazine, pyrimidines, fused rings, and indole/benzimidazole. Every molecule was validated with RDKit and passes Lipinski's Rule of Five.

We also included 16 named reference drugs (Aspirin, Ciprofloxacin, Celecoxib, Adenosine, Salbutamol, Nicotine, Paracetamol, and others) as anchor points. These help the model learn the "feel" of real approved drugs.

### ZINC-250K (Extended Modes)

For moderate, large, and large-GPU modes we pull from ZINC-250K — the standard benchmark dataset in molecular generation research (Gómez-Bombarelli et al. 2018). It's 250,000 drug-like SMILES strings, freely available, no login required.

- Source: aspuru-guzik-group on GitHub, permanent public URL
- Filter applied: MW 150–500 Da, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
- Moderate mode uses the first 25,000 entries
- Large and Large-GPU modes use the first 100,000 entries
- Downloaded automatically on first run, cached locally for offline reuse

---

## 11. How We Compare to Existing Tools

The honest answer is that there are two categories of tools in this space: tools that generate molecules, and tools that dock/score existing molecules. Most tools people cite — DiffDock, GNINA, AlphaFold 3, Boltz-2 — are in the second category. They don't generate new molecules at all. They take molecules you give them and predict how they bind.

The only other publicly available tool that actually generates molecules like we do is NVIDIA MolMIM. The difference is that MolMIM requires A100 or H100 data-centre GPUs and an NGC account. Our tool runs on a student laptop.

| Tool | Generates molecules? | CPU only? | Free? | Browser UI? |
|------|---------------------|----------|-------|-------------|
| This project | Yes | Yes | Yes | Yes — 8 analysis tabs |
| NVIDIA MolMIM | Yes | No — A100/H100 required | No — NGC account needed | No |
| DiffDock | No | No | Yes | No |
| GNINA | No | Partial | Yes | No |
| AlphaFold 3 | No | No | No | No |
| Boltz-2 | No | No | Yes | No |
| EquiBind | No | No | Yes | No |

---

## 12. Limitations — The Honest List

We think being upfront about what the tool doesn't do is more useful than overselling it. Here's what you should know:

- Basic and moderate modes use estimated docking scores, not real protein docking. Large and Large-GPU use real Vina and DiffDock.
- The RL optimisation is hill-climbing, not PPO. It's effective but it has no memory and doesn't learn across runs.
- The seed library in basic mode (295 molecules) is kinase-heavy. If you're designing for a target very different from kinases, the moderate or large modes will give you better diversity.
- No wet-lab validation. Everything we produce is a computational prediction. Whether a molecule actually binds in the real world requires experimental testing.
- The model doesn't condition on protein structure. It doesn't "see" the protein pocket when generating — the target name only affects the reward weighting, not the latent space navigation.
- About 50% of generated molecules pass all 8 quality filters. The rest are chemically valid (thanks to SELFIES) but not drug-like enough to be useful.

---

## 13. References

- Krenn M. et al. (2020). SELFIES: a robust molecular string representation. *Machine Learning: Science and Technology* 1(4).
- Kingma D.P. & Welling M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.
- Higgins I. et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*.
- Fu H. et al. (2019). Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing. *NAACL-HLT 2019*.
- Schulman J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Schulman J. et al. (2016). High-Dimensional Continuous Control Using Generalised Advantage Estimation. *ICLR 2016*.
- Olivecrona M. et al. (2017). Molecular de-novo design through deep reinforcement learning. *Journal of Cheminformatics* 9(1):48.
- Deb K. et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation* 6(2):182–197.
- Gómez-Bombarelli R. et al. (2018). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. *ACS Central Science* 4(2):268–276.
- Trott O. & Olson A.J. (2010). AutoDock Vina: improving the speed and accuracy of docking. *Journal of Computational Chemistry* 31(2):455–461.
- Corso G. et al. (2022). DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking. *ICLR 2023*.
- Mendez D. et al. ChEMBL 34. *European Bioinformatics Institute*, 2024.
- NVIDIA BioNeMo Framework documentation (2026). MolMIM NIM deployment guide.

---

*GitHub: [github.com/Kaur-Simarpreet/molecular-design-vae](https://github.com/Kaur-Simarpreet/molecular-design-vae)*
