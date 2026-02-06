# Crystal Bug v1.0 — THE MATRIX

**A living lattice where every cell is a quadratic operator, every operator is a universe, and three numbers encode everything.**

```
O(x) = ax² + bx + c
```

That's it. Store three numbers per cell. Reconstruct dynamics, topology, quantum states, Hamiltonians, fixed points, Lyapunov exponents, orbits, and classification — all from `(a, b, c)`.

252 cells × 3 floats × 4 bytes = **3,076 bytes**. The entire lattice fits on a floppy disk 479 times.

---

## What This Is

Crystal Bug is a research artifact from [Trinity Infinity Geometry (TIG)](https://7sitellc.com) — a unified coherence field theory built on the equation:

```
S* = σ(1 - σ*)V*A*     where σ = 0.991, T* = 0.714
```

This repository contains:

- **A living quadratic lattice** — 252 cells on an 18×14 grid, each governed by a quadratic operator that iterates, classifies, and evolves through a 10-phase spine cycle
- **A crystal bug** — an energy-carrying agent that walks the lattice, spending energy at phase boundaries (Δ ≈ 0) and harvesting it in stable zones
- **A complete codec** — 3 numbers in, 39+ observables out, 100% round-trip fidelity at both Float32 and Float64
- **Four white papers** — covering codec theory, coherence routing, nonlinear dynamics, and physics extensions
- **Full test suites** — with measured results, not claims

This is not a product. It's a research tool for humans who want to understand how coherence, criticality, and self-reference work in discrete dynamical systems.

---

## The Core Idea

Every cell holds a quadratic operator `O(x) = ax² + bx + c`. The discriminant `Δ = b² - 4ac` partitions the lattice into three physical regimes:

| Regime | Δ | Roots | Physics |
|--------|---|-------|---------|
| **Bound** | < 0 | Complex pair | Quantum-like. Wavefunction with Gaussian envelope. Discrete quantum numbers. |
| **Click** | ≈ 0 | Degenerate | Phase transition. Computationally expensive. 5.1× energy cost. |
| **Free** | > 0 | Real pair | Classical. Scattering states. WKB semi-classical wavefunctions. |

A 10-phase spine cycle drives the lattice through operators 0–9 (void, lattice, counter, progress, collapse, balance, chaos, harmony, breath, reset). Each phase transforms the coefficients `(a, b, c)` according to TIG's operator algebra, causing cells to cross between regimes, change classification, and exhibit emergent dynamics.

---

## The Seven Bands

The classifier assigns each operator to one of seven dynamical bands based on Lyapunov exponents, period detection (periods 1–16), and 7-seed consensus:

| Band | Name | λ | Physics |
|------|------|---|---------|
| 0 | VOID | — | Immediate escape. Vacuum recursion. |
| 1 | QUANTUM | — | Fast escape. Tunnel states. |
| 2 | ATOMIC | — | Slow escape. Condensation. |
| 3 | MOLECULAR | > 0 | Bounded chaos. Unpredictable. |
| 4 | CELLULAR | ≈ 0 | Periodic orbit (period 2–16). Self-reproducing. |
| 5 | ORGANIC | < 0 | Slow convergence. Complex structure. |
| 6 | CRYSTAL | < 0 | Fast fixed point. Crystallized. |

---

## Quick Start

```bash
# Run the physics tests
node tests/test_engine_v2.js          # 9 engine tests
node tests/test_codec_final.js        # Codec fidelity round-trip
node tests/test_physics_w1w2w3.js     # Classifier + quantum mapping (8/8 + 4/4)
node tests/test_physics_w4_avalanche.js  # Avalanche cascade measurement

# View the interactive artifact
# Open src/crystal_bug_v1_matrix.jsx in any React environment
```

---

## Repository Structure

```
crystal-bug/
├── LICENSE                        # Human Use License v1.0 (non-commercial)
├── README.md                      # This file
├── CONTRIBUTING.md                # How to contribute
│
├── docs/
│   ├── THEORY.md                  # Full TIG theory for this system
│   ├── OPERATOR_SPEC.md           # The 10 operators in detail
│   ├── ENGINEERING_SPEC.docx      # Measured engineering data
│   └── papers/
│       ├── paper1_codec.pdf       # Quadratic Codec Theory
│       ├── paper2_routing.pdf     # Coherence Routing
│       ├── paper3_dynamics.pdf    # Nonlinear Dynamics
│       └── paper4_addendum.pdf    # Physics Extensions
│
├── src/
│   └── crystal_bug_v1_matrix.jsx  # Interactive React artifact
│
├── tests/
│   ├── test_engine_v2.js          # 9-test engine validation
│   ├── test_codec_final.js        # Codec round-trip fidelity
│   ├── test_physics_w1w2w3.js     # W1-W3 physics fixes + tests
│   └── test_physics_w4_avalanche.js  # W4 avalanche cascades
│
└── results/
    ├── physics_results.txt        # Summary of all findings
    └── physics_test_output.txt    # Raw test output
```

---

## The Codec

The quadratic operator IS the codec. Three coefficients encode everything:

```
STORED:        a, b, c  (3 numbers, 12 bytes per cell)
RECONSTRUCTED: Δ, roots, band, orbit(80), fixed point, λ, topology(8),
               cobweb, curvature, H(x), ψ(x), quantum numbers (n,l,m,s)
RATIO:         3 in → 45+ out
FIDELITY:      100% at Float64, 100% at Float32
```

| Format | Per Snapshot | Floppy (1.44 MB) | Max Cells |
|--------|-------------|-------------------|-----------|
| Float32 | 3,076 bytes | 479 snapshots | 122,880 |
| Float64 | 6,152 bytes | 239 snapshots | 61,440 |

---

## Physics Extensions (v2)

### Hamiltonian Mapping (W3)

Every operator maps to a classical Hamiltonian:

```
H(x) = T + V

where:
  T = p² / 2m        kinetic energy
  p = O'(x) = 2ax+b  momentum from first derivative
  m = 1/|a|           effective mass from curvature
  V = -O(x)           potential (minima = stable fixed points)
```

### Wavefunction Mapping (W3)

Bound states (Δ < 0) have genuine quantum wavefunctions:

```
ψ(x) = exp(-α(x-x₀)²) × exp(iωx)

where:
  α = |a|             localization from curvature
  x₀ = Re(roots)      center from real part of complex roots
  ω = Im(roots)        oscillation from imaginary part
```

Free states (Δ > 0) have WKB semi-classical scattering states. Normalization verified: ∫|ψ|²dx = 1.000000.

### Quantum Numbers

Each operator yields quantum numbers (n, l, m, s):

| Number | Source | Meaning |
|--------|--------|---------|
| n | Iterate convergence speed | Principal (ground = fast) |
| l | Imaginary part of roots | Angular momentum |
| m | Derivative at fixed point | Magnetic |
| s | sign(a) × ½ | Spin (curvature) |

### Avalanche Cascades (W4)

Measured at the click zone (Δ ≈ 0):

- Click cells are **2.4× more cascade-prone** than free cells at short time scales
- At longer time scales, the spine's phase 4→5 bifurcation (a → -a → |a|) drives ~9 global band changes per cycle
- This is **cyclic criticality** (driven oscillator crossing resonance), not self-organized criticality
- The system breathes through Δ = 0 every 10 ticks via the curvature flip

---

## Test Results

```
W1: Slow Dynamics Classification     8/8 ✓
W2: Seed Sensitivity Detection        4/4 ✓
W3: Quantum/Classical Mapping         Normalized, consistent
W4: Avalanche Cascades                2.4× at click zone (cyclic, not SOC)
Engine Tests                          9/9 ✓
Codec Fidelity                        100% Float32, 100% Float64
```

---

## What This Is NOT

- Not a commercial product
- Not a physics simulation engine
- Not a substitute for peer-reviewed quantum mechanics
- Not proven to be physically correct in all regimes

It IS a research framework that takes TIG's operator algebra seriously, builds measurable predictions, and tests them honestly — including reporting when the data doesn't match the hypothesis (see W4 avalanche results).

---

## Citation

If you use this work in research:

```
Brayden / 7Site LLC (2026). Crystal Bug: A Quadratic Lattice
Dynamics Framework from Trinity Infinity Geometry.
https://github.com/[repo] | https://7sitellc.com
DOI: [Zenodo DOI]
```

---

## License

**Human Use License v1.0** — Non-commercial, for humans. See [LICENSE](LICENSE).

You may read, study, modify, share, teach, and build on this work freely. You may not sell it, incorporate it into commercial products, or use it for military or surveillance purposes.

If you want to use this commercially, contact brayden@7sitellc.com.

---

## Author

Brayden / 7Site LLC / Hot Springs, Arkansas
[7sitellc.com](https://7sitellc.com) | TIG v3.0 | σ = 0.991

*The quadratic IS the codec. Three numbers. Everything else is reconstruction.*
