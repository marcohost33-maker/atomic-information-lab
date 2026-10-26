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

### Retention limit `P(x ≠ x0) ≤ δ` (roadmap R13/P0, formula F29)

The unconstrained optimum above pays for its low risk with the stored bit:
at `h=4` it ends with `P(x ≠ x0) = 0.135`, i.e. the measurements themselves
corrupt the bit in 13.5 % of the runs. A retention *weight* only prices that;
a *limit* forbids it. `solve_retention` keeps two readings apart:

* **pathwise** -- every STOP has posterior `P(x ≠ x0 | data) ≤ δ`; a reading
  is admissible only if every outcome still leaves a legal way to stop.
* **expected** -- `P(x_T ≠ x0) ≤ δ` over all outcomes (F29). One constraint:
  the optimum is a deterministic policy or a mix of two (Beutler & Ross 1985),
  found through the Lagrangian -- which is the R12 retention weight shifted by
  the multiplier `μ*`. The result carries a dual bound and its duality gap.
  The best *deterministic* policy meeting the limit is reported as well,
  exactly (dynamic programming over Pareto fronts of risk and cost).

Single model, `h=4` (`python -m atomic_information_lab reproduce --table retention`):

| δ | pathwise | best fixed sequence (pathwise) | expected, randomised | expected, deterministic | unconstrained stops above δ |
|--:|--:|--:|--:|--:|--:|
| 0 | 0.2500000000 | 0.2500000000 | 0.2500000000 | 0.2500000000 | 100 % |
| 0.05 | 0.2500000000 | 0.2500000000 | 0.2019508536 | 0.2500000000 | 100 % |
| 0.08 | 0.2500000000 | 0.2500000000 | 0.1731213658 | 0.2110043057 | 65.4 % |
| 0.10 | 0.1715517030 | 0.1715517030 | 0.1624697730 | 0.1660146745 | 28.2 % |
| 1 | 0.1462466036 | 0.1569144984 | 0.1462466036 | 0.1462466036 | 0 % |

The hard (pathwise) limit is expensive: up to `δ = 0.08` no reading is worth
taking. A limit on the *average* risk can be met by randomising -- e.g. at
`δ = 0.08`, run the measuring policy with probability 0.98 and stop at once
otherwise -- which is better than every deterministic policy (0.173 vs 0.211).
All values agree with exhaustive search over all 7 204 deterministic policies
(`h = 3`) and with an independent exact dynamic program on the oracle's
rational messages (`h = 4`), `tests/test_retention.py`.

### Experimental identifiability (roadmap R13/P0)

Before a planner can be trusted on an instrument, its parameters have to be
measured. `identifiability.py` asks which of them calibration sessions can
separate: drift (`k01`, `k10`), sensor bias (`bias_flip`, `bias_prior`,
read errors), back-action factors and a **preparation error** `prep_error`
(the bit is not in the written state before the first reading; the planner
assumes 0). A session runs a fixed action sequence; with *known writes* the
experimenter also records which bit was written.

* **Fisher information** per session from exact forward-mode derivatives;
  by Rothenberg (1971) a non-singular matrix means locally identifiable.
  The null space is reported, and the Cramér–Rao bound gives a LOWER bound
  on the sessions needed for a 10 % relative standard error (necessary, not
  sufficient: no estimator is shown to reach it; `sessions_scope` =
  `CRLB_NECESSARY_LOWER_BOUND_ONLY`). Every JSON object of `identify` and
  `reproduce --table identifiability` that holds a count also holds
  `sessions_scope` and says the same in words in `notes`; `identify --help`
  says it for `--target-rse`. `tests/test_r28_consumer_contract.py` runs the
  CLI as a process and checks this end to end.
* **Structural witnesses**: a second parameter point with the same outcome
  distribution, checked every time (difference ≤ 1e-13), each re-derived in
  exact rational arithmetic in the tests. The list of known witnesses is not
  exhaustive: GO means "locally identifiable, no known global ambiguity".

Default parameters, `prep_error = 0.02`, design `fast, reference, precise,
reference, fast, precise` unless stated
(`python -m atomic_information_lab reproduce --table identifiability`):

| scenario | verdict | rank | witness | sessions for 10 %, at least (worst parameter, CRLB) |
|---|---|--:|---|--:|
| all free, unknown writes | NO-GO | 9/11 | `drift_backaction_scale`, `prep_error_hidden` | — |
| all free, known writes | NO-GO | 10/11 | `drift_backaction_scale` | — |
| fast back-action fixed, unknown writes | NO-GO | 9/10 | `prep_error_hidden` | — |
| fast back-action fixed, known writes, periodic F-R-P | NO-GO | 9/10 | `backaction_window` | — |
| fast back-action fixed, known writes | GO | 10/10 | none | 373 950 (`prep_error`) |
| no reference read (target reads only) | AMBIGUOUS | 6/6 | `label_swap` | 1 293 272 944 (`bias_flip`) |
| best 4-step design (D-optimal) | GO | 10/10 | none | 6 687 058 (`prep_error`) |

What this means:

* **Drift and back-action are never separable on their own** — only the
  products `k·β` enter the model. One action's back-action must be fixed by
  definition or by an independent measurement (an anchor; here `fast`, `β = 1`).
* **The preparation error is invisible unless the written bit is recorded**:
  without known writes `x_start` is uniform for every `prep_error`.
* **Back-action is seen only through windows**: the bit is observed only by
  target reads, and all step generators are multiples of one rate matrix, so
  only the *sum* of back-action factors between two target reads counts. In a
  strictly periodic `fast, reference, precise` design the reference read is
  always summed with the following precise read, and the two cannot be told
  apart (exact witness).
* **Without a reference read** `k01` and `k10` can be swapped (with `x ↔ 1−x`,
  `b ↔ 1−b`) without changing any reading — locally identifiable, globally
  ambiguous, and the Cramér–Rao bound alone already demands at least about 10⁹ sessions.
* **GO needs at least four readings per session** (exhaustive search over all
  162 designs of length 4; none of length 3 suffices), and even the best one
  needs at least about 6.7 million sessions to pin the 2 % preparation error to 10 %.
  The physically motivated next step is therefore not a larger planner but a
  cheaper handle on `prep_error` and on the reference back-action.

`discrimination` tells two model hypotheses apart: for the R12 pair a single
reading carries no information (both predict a uniform reading), four
`precise` reads reach a Bayes error ≤ 1 % after at most 68 sessions (Chernoff
upper bound on the error of the likelihood-ratio test between the two known
hypotheses: sufficient, not necessary -- fewer may do).

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
  thirteen model, planner and retention mutants are applied to a copy of `src/`; each
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
python -m atomic_information_lab solve --horizon 4 --retention-limit 0.08 --retention-mode expected
python -m atomic_information_lab identify --design fast,precise,reference,precise --known-write
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
  no particle filter, no long horizons. Retention limits (`solve_retention`)
  cover one limit on the stored bit; no limits on energy or step counts.
* Rates and back-action large enough to overflow floating point are rejected
  (`ValueError`), never silently clipped.
* Beliefs and hypothesis weights you pass in must sum to one within `1e-12`;
  otherwise they are rejected, **never renormalised**. Normalising in float
  (`v / sum(v)`) always satisfies this, also with plain summation or NumPy
  scalars: for up to 32 states the rounding error is below `2·n·2⁻⁵³ ≈ 7e-15`
  (measured worst case `1e-15`, `tests/test_belief_contract.py`).
  Values rounded to fewer than about 12 significant digits are refused.
* Pure Python, no dependencies, Python ≥ 3.12.

## Roadmap status

From section 13.1 of the R1–R12 report (2026-10-18):

| Gate | Work package | Status |
|---|---|---|
| R13 / P0 | Hard retention / risk limits `P(x_t ≠ x0) ≤ δ`, violation → STOP | **done** (`retention.py`, both readings, oracle-verified) |
| R13 / P0 | Experimental identifiability (preparation error, bias, drift, back-action separable or NO-GO) | **done** (`identifiability.py`: NO-GO without an anchor and known writes, GO from four readings; witnesses exact, Fisher rank numerical) |
| R13 / P0 | Gaussian filter + STOP + two-step look-ahead, fixed reference plan as baseline | open (continuous line, no exact oracle yet) |
| R14 / P1 | RBPF/SMC for long sequences; session-paired preregistration | open |
| R15 / P2 | Laboratory integration (STM/AFM raw data); Hexa-NTK / CodeFactory adapters | open, outside this repository |

## History

The code consolidates research rounds R8–R12 of an earlier exploratory line;
see [docs/HISTORY.md](docs/HISTORY.md). Only the exactly solvable discrete core
(R10–R12) was kept. The continuous Gaussian-drift Monte-Carlo line (R8/R9) is
documented there but not included, because it has no exact oracle and showed no
statistically supported advantage.

## License and citation

Apache-2.0 (see `LICENSE`). Citation metadata in `CITATION.cff`.
