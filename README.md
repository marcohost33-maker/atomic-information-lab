# atomic-information-lab

Exact value-of-information and optional stopping in a **synthetic, discrete
two-state measurement model** — a small, dependency-free Python reference with an
independent test oracle.

> **Claim boundary.** Everything here is mathematics inside an *assumed*
> classical model. The parameters are invented, not calibrated against any
> instrument. There is no atomic sensor, no STM/AFM data, no quantum-state
> claim, no hardware access and no production use. "Exact" means: exact for
> the stated model up to floating-point rounding. Model error is unbounded and
> not quantified. The name refers to the motivating question (reading and
> writing single-atom bits), not to a demonstrated capability.

## What it answers

You stored a bit `x0`. It may have flipped since (`x`), and every reading
disturbs it further (back-action). Your sensor may be inverted by an unknown,
slowly flipping bias `b`. Every reading costs energy. **When should you measure,
with which instrument, and when should you stop?**

The library computes the Bayes-optimal adaptive policy with an explicit STOP
action over the complete observation tree (up to five steps), and compares it
with the best fixed measurement sequence, a myopic (greedy) rule, a planner
that must use its whole horizon, and a policy built for the wrong model.

## Model

Hidden state per model hypothesis `θ`: `(x0, x, b)` ∈ {0,1}³. One measurement
with action `a`:

1. `x` evolves for one time unit under a two-state continuous-time Markov chain
   with rates `k01·β_a`, `k10·β_a` (`β_a` = back-action factor):
   `P(0→1) = k01/(k01+k10) · (1 − e^{−(k01+k10)β_a})`, analogously `P(1→0)`.
2. `b` flips with probability `f` (`bias_flip`).
3. The reading is `y = target ⊕ ε` with `ε ~ Bernoulli(e_a)` and
   `target = x ⊕ b` for a target read, `target = 0 ⊕ b` for a read of a known
   reference (which therefore reveals `b`).

Prior: `x0` uniform, `x = x0`, `b` uniform; a finite set of hypotheses `θ`
(at most four) with prior weights. The joint belief over `(θ, x0, x, b)` is
updated exactly:

`p(θ, s' | D, a, y) ∝ p(y | s', a) · Σ_s p(s' | s, a, θ) · p(θ, s | D)`.

Terminal loss (expected Brier score of the posterior-mean forecasts, plus an
optional retention penalty):

`L(b) = α·p0(1−p0) + (1−α)·pt(1−pt) + β_ret·P(x ≠ x0)`

with `p0 = P(x0 = 1 | D)`, `pt = P(x = 1 | D)`. Each reading costs `λ·E_a`.
Bellman recursion with optional stopping:

`V_0(b) = L(b)`,
`V_h(b) = min{ L(b), min_a [ λE_a + Σ_y P(y | b, a) · V_{h−1}(τ(b, a, y)) ] }`.

Ties within `1e-12` are resolved deterministically: STOP first, then actions
in catalogue order (the reported value can exceed the exact optimum by at most
`h·1e-12`).

Default actions (abstract units): `fast (β=1, e=0.12, E=1)`,
`precise (β=3.5, e=0.035, E=2.8)`, `reference (β=0.55, e=0.025, E=0.8)`;
default hypothesis `k01=0.07, k10=0.045, f=0.025`; `α=0.5, λ=0.012`.

## Reproduced results

All numbers below are produced by this code (`python -m atomic_information_lab
reproduce`) and agree with the independent oracle and with the historic result
files to ≤ 1e-12 (see `tests/`).

Single model:

| h | optimal with STOP | forced exactly h | best fixed sequence | greedy | E[steps] |
|--:|--:|--:|--:|--:|--:|
| 0 | 0.2500000000 | 0.2500000000 | 0.2500000000 | 0.25 | 0 |
| 1 | 0.2500000000 | 0.2595778083 | 0.2500000000 | 0.25 | 0 |
| 2 | 0.1702173178 | 0.1702173178 | 0.1702173178 | 0.25 | 2 |
| 3 | 0.1569144984 | 0.1569144984 | 0.1569144984 | 0.25 | 3 |
| 4 | 0.1462466036 | 0.1564022934 | 0.1569144984 | 0.25 | 3.282 |
| 5 | 0.1418447352 | 0.1619161117 | 0.1569144984 | 0.25 | 3.563 |

At `h=5` the adaptive STOP policy is 9.60 % below the best fixed sequence and
12.40 % below the forced-horizon planner — a model-internal number. Forcing
measurements can *increase* the risk (h=1, and h=4→5); the greedy rule never
measures from the initial prior, because no single reading is worth its cost
while the sensor inversion is unknown.

Two hypotheses `θ1=(0.07, 0.045, 0.025)`, `θ2=(0.23, 0.16, 0.10)`, prior (0.5, 0.5):

| h | adaptive (model-averaged) | best fixed sequence | policy optimal for θ1 only |
|--:|--:|--:|--:|
| 1 | 0.2500000000 | 0.2500000000 | 0.2500000000 |
| 2 | 0.1967940161 | 0.1967940161 | 0.2039912561 |
| 3 | 0.1878507467 | 0.1878507467 | 0.1878507467 |
| 4 | 0.1840526517 | 0.1878507467 | 0.1840526517 |

At `h=4` the single-model policy is **exactly** as good as the model-averaged
one (verified in rational arithmetic) — this example does not show a general
advantage of model averaging. With retention weight 0.4 and reference
back-action ≥ 0.55, STOP at the start is optimal.

See [docs/FINDINGS.md](docs/FINDINGS.md) for every audited claim and its
verdict, including counter-examples.

## How it is verified

* **Independent oracle** (`tests/_oracle.py`, no import of the package):
  exact rational arithmetic (`fractions.Fraction`) over *unnormalised* forward
  messages per observation history, with the Markov step computed by
  uniformisation instead of the closed form; plus brute force over **every**
  deterministic policy for `h ≤ 3` (7 204 policies with STOP, 2 187 without)
  using explicit hidden-path enumeration.
* Posterior marginals (not only values) are compared, because values are
  invariant under relabelling `x ↔ 1−x` and would not reveal swapped rates.
* Golden values from the historic result files; structural properties
  (monotonicity in `h`, dominance chain, normalisation over the full tree,
  strict propriety of the Brier score); negative controls each paired with a
  positive control; fail-closed tests at every public boundary and the CLI.
* **Policy cross-evaluation** (`tests/test_policy_cross_evaluation.py`): the
  package's *decisions* -- its own belief filter plus its own Bellman choice --
  are replayed inside the oracle's dynamics and scored by the oracle; the score
  must equal the oracle's optimum and, for `h ≤ 3`, the brute-force minimum.
  A value comparison alone cannot see a wrong decision whose value is computed
  correctly elsewhere; negative controls (never stopping, greedy, one wrong
  decision at depth 2) are detected.
* **Metamorphic relations** (`tests/test_metamorphic.py`, no oracle): flipping
  every reading, relabelling the bit (`k01 ↔ k10`), permuting hypotheses or
  actions, splitting a hypothesis into identical copies, rescaling
  `λ ↔ E` and adding a dominated action must leave the results unchanged.
  They see errors without any reference values, but not every error: rates
  swapped consistently everywhere keeps every relation true.
* **Simulation of the generative story** (`tests/_simulator.py`,
  `tests/test_generative_simulation.py`): a Monte-Carlo simulation written
  from the verbal model alone (Gillespie jumps, no matrices, no shared code)
  is compared with the package per reading pattern -- probability and
  posterior of `x0`, `x`, "changed" and the model index -- by binomial
  z-scores (`|z| ≤ 5`; measured null 2.0–2.7, wrong stories 28–91). This is
  the only check that sees a consistent transcription error such as swapped
  rates.
* **Reproducible mutation probe** (`scripts/mutation_probe.py`, a CI step):
  eight model and planner mutants are applied to a copy of `src/`; each
  suite's verdict (`relation`, `guard`, `survives`) is compared with the
  expected one, after a green baseline and a check that the copy is what was
  imported. It records the limits too: M7 (swapped rates) survives the
  metamorphic suite and is killed by the simulation; M8 (decision rule one
  step short, values unchanged) is killed only by the policy cross-evaluation.
* Mutation probes: each guard was removed in turn and the corresponding test
  was required to fail (15/15).

What these checks establish, in the terms `scripts/research_gate.py` reports
under `claims`: **value agreement** and **policy optimality** on the tested
grids are *measured*; **independence of the specification** is *not measured*
by the gate (the oracle is a second transcription of the same equations); the
simulation of the generative story narrows this statistically for the tested
sequences, it is not a proof; **physical validity** is *not claimed*.

## Usage

```bash
python -m pip install .
python -m atomic_information_lab solve --horizon 4
python -m atomic_information_lab solve --horizon 1 --forced
python -m atomic_information_lab reproduce --table single
```

```python
from atomic_information_lab import Problem, TWO_HYPOTHESES, solve, best_open_loop

pr = Problem(TWO_HYPOTHESES, (0.5, 0.5))
print(solve(pr, horizon=4))          # Decision(value=0.18405..., action='fast')
print(best_open_loop(pr, 4))         # (0.18785..., ('fast', 'reference', 'fast'))
```

Run the tests: `python -m pip install pytest && python -m pytest`.

## Limits

* Horizon ≤ 5, ≤ 4 hypotheses, ≤ 4 actions; cost grows like `(2·|A|)^h`.
  The full reproduction takes about 10 s, the test suite under a minute
  (one Windows laptop, Python 3.12).
* Binary observations, static model index, one bit. No continuous drift model,
  no particle filter, no long horizons, no hard retention constraints.
* Rates and back-action large enough to overflow floating point are rejected
  (`ValueError`), never silently clipped.
* Beliefs and hypothesis weights you pass in must sum to one within `1e-12`;
  otherwise they are rejected, **never renormalised**. Normalising in float
  (`v / sum(v)`) always satisfies this, also with plain summation or NumPy
  scalars: for up to 32 states the rounding error is below `2·n·2⁻⁵³ ≈ 7e-15`
  (measured worst case `1e-15`, `tests/test_belief_contract.py`).
  Values rounded to fewer than about 12 significant digits are refused.
* Pure Python, no dependencies, Python ≥ 3.12.

## History

The code consolidates research rounds R8–R12 of an earlier exploratory line;
see [docs/HISTORY.md](docs/HISTORY.md). Only the exactly solvable discrete core
(R10–R12) was kept. The continuous Gaussian-drift Monte-Carlo line (R8/R9) is
documented there but not included, because it has no exact oracle and showed no
statistically supported advantage.

## License and citation

Apache-2.0 (see `LICENSE`). Citation metadata in `CITATION.cff`.
