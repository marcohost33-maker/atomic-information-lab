# atomic-information-lab

Exact value-of-information and optional stopping in a **synthetic, discrete
two-state measurement model** (plus a linear-Gaussian analogue, `gaussian.py`) — a
small, dependency-free Python reference with independent test oracles.

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
  The null space is reported, and the Cramér–Rao bound gives a lower bound
  on the sessions an **unbiased** regular estimator needs for a 10 % relative
  standard error, at this parameter point of this model. It is not
  sufficient, and it is no lower bound for biased estimators, whose mean
  squared error can be smaller (`sessions_scope` =
  `CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY`, AIL-R29-01). Every JSON object of `identify` and
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
hypotheses: sufficient, not necessary -- fewer may do; `Discrimination`
carries `sessions_scope = CHERNOFF_SUFFICIENT_KNOWN_PAIR_ONLY`). A Monte-Carlo
check (`scripts/chernoff_check.py`, seed 20261104) puts the actual error at
68 sessions at 0.0021 ± 0.0002 and below 1 % already between 40 and 50
sessions; on a small binary case the exact Bayes error confirms the direction
(reported 47 / 162, exact minimum 32 / 112, `tests/test_r29_estimator_scope.py`).

### Does an actual estimator reach the Cramér–Rao counts? (R29 P1)

`estimation.py` fits the maximum-likelihood estimator to simulated outcome
counts (exact multinomial draws; Fisher scoring started 20 % away from the
truth; failed fits counted, not dropped) and measures bias, relative standard
error and coverage of the 95 % Wald interval against the bound. Protocol
fixed in [docs/ESTIMATOR_PROTOCOL.md](docs/ESTIMATOR_PROTOCOL.md) before the
run (separate commit); results in `docs/ESTIMATOR_CALIBRATION_RESULTS.json`
(`python scripts/estimator_calibration.py`, R = 1000 per N, about 5 min).
6-step design of the table, known writes, anchor `fast`:

| sessions N | converged | `prep_error` RSE (CRLB) | `backaction:reference` bias / RSE (CRLB) | coverage range, 10 parameters |
|--:|--:|--:|--:|--:|
| 23 372 (count / 16) | 992 / 1000 | 0.381 (0.400) | +7.3 % / 0.397 (0.352) | 0.942–0.964 |
| 93 488 (count / 4) | 1000 / 1000 | 0.201 (0.200) | +0.3 % / 0.179 (0.176) | 0.936–0.956 |
| **373 950 (count)** | 1000 / 1000 | **0.098** (0.100) | +0.4 % / 0.088 (0.088) | 0.940–0.953 |
| 1 495 800 (count · 4) | 1000 / 1000 | 0.049 (0.050) | +0.1 % / 0.043 (0.044) | 0.939–0.956 |

Pre-registered primary endpoint met: at the Cramér–Rao count the MLE reaches
the 10 % target on `prep_error` (0.098 ± 0.0023 Monte-Carlo SE, 5.2 SE below
the 0.11 limit). Model-internal: the data come from the fitted model, one
point, one design -- this calibrates the count as a planning figure *inside
the model*, not on an instrument.

Post hoc, not confirmatory ([docs/ESTIMATOR_POSTHOC.json](docs/ESTIMATOR_POSTHOC.json),
`scripts/estimator_posthoc.py`; a new run first reproduces all 517 registered
values exactly):

* From a quarter of the count on, all 30 ratios empirical RSE / Cramér–Rao RSE
  have a 95 % Monte-Carlo interval containing 1, and no coverage leaves the
  family-wise band (Bonferroni over 40 cells) or the registered screen
  [0.93, 0.97]. That -- and only that -- is meant by "efficient within
  Monte-Carlo error" here; no equivalence margin was registered.
* At a sixteenth the asymptotics break: 8 of 1000 fits fail at the boundary,
  4 of 10 ratio intervals exclude 1, `backaction:reference` is biased by +7.3 %.
  `prep_error` lies *below* the bound (ratio 0.951, interval 0.910–0.993) with
  a bias of +2.9 % -- a biased estimator is not bound by Cramér–Rao.
* Coverage at that N is over the 992 converged fits; over all 1000
  replications it lies between 0.934 and 0.964 whatever the failed fits did
  (`coverage_all_low/high`).

The chain protocol → results is checked from git
(`python scripts/verify_preregistration.py`, also in CI): the protocol text
above "Deviations" and the runner are byte-identical from the registration
commit `1097c2d` to the results commit, and the recorded setting matches the
protocol. Re-running at `1097c2d` reproduces the registered file
bit-identically (Python 3.13). Git proves content and order, not wall-clock
time; commit dates are self-declared.

See [docs/FINDINGS.md](docs/FINDINGS.md) for every audited claim and its
verdict, including counter-examples.

### Separate outputs: risk, information, retention, energy, steps (R29-02)

`solve` returns one number, expected terminal loss plus priced energy.
`policy_profile` (`metrics.py`) evaluates one policy on the full observation
tree and reports each quantity on its own: the Bayes risk and its three
ingredients, the mutual information `I(X0; D_T)` between the written bit and
everything recorded up to the stopping time (bits), `P(x_T ≠ x0)`, energy and
the distribution of the number of readings. `value = bayes_risk +
energy_cost` is checked on every call. The information figure is reported
next to the risk; it is **not** the objective.

`python -m atomic_information_lab reproduce --table profile` (default model):

| h | policy | value | Bayes risk | I(X0; D) bit | P(x_T ≠ x0) | E[energy] | E[steps] |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | forced | 0.259578 | 0.249978 | 0.0000 | 0.0306 | 0.800 | 1.000 |
| 2 | optimal with STOP | 0.170217 | 0.127017 | 0.2120 | 0.1862 | 3.600 | 2.000 |
| 3 | optimal with STOP | 0.156914 | 0.123314 | 0.3680 | 0.1271 | 2.800 | 3.000 |
| 4 | optimal with STOP | 0.146247 | 0.109266 | 0.3855 | 0.1348 | 3.082 | 3.282 |
| 5 | optimal with STOP | 0.141845 | 0.096847 | 0.3866 | 0.1511 | 3.750 | 3.563 |
| 5 | forced | 0.161916 | 0.101540 | 0.4179 | 0.1994 | 5.031 | 5.000 |
| 5 | best fixed sequence | 0.156914 | 0.123314 | 0.3680 | 0.1271 | 2.800 | 3.000 |

What the single number hides (finding N-32, all model-internal):

* **More information, worse outcome.** At `h = 5` the forced planner learns
  more about the written bit than the optimum (0.418 vs 0.387 bit) and is
  worse on value, on retention (0.199 vs 0.151) and on energy.
* **The value-optimal sequence is not the most informative one.** At `h = 2`
  the optimum `reference, precise` yields 0.212 bit; `fast, reference` yields
  0.290 bit with half the energy (1.8 vs 3.6) and less than half the
  retention risk (0.082 vs 0.186), at a value 0.0027 worse.
* **Risk can fall with zero information** (N-2, now measured): with
  `bias_flip = 0.5` every reading is noise about `x0`, `I(X0; D) = 0` at every
  horizon, yet the risk falls because back-action makes the current bit more
  predictable.
* From `h = 3` to `h = 5` the optimum gains 0.019 bit and its retention risk
  rises from 0.127 to 0.151: the later readings buy accuracy on the *current*
  bit. It stops after three readings (probability 0.718) or uses all five
  (0.282), never four.

No mutual information is reported for the current bit `X_T`: its marginal
changes with every reading, so "prior minus posterior entropy" would not be a
mutual information of two fixed variables; `entropy_current_bits` is the
remaining uncertainty.

### Pareto list of calibration designs and the Fisher spectrum (R29-03)

`design_search` ranks by verdict and then by `log det` of the Fisher
information (D-optimality) and returns one winner. `design_pareto` keeps
every design that no other design beats at once on verdict, `log det`, the
Cramér–Rao bound for the worst parameter and energy per session. Each score
also carries `campaign_energy_crlb` (bound × energy per session: the energy of
the smallest campaign the bound allows an unbiased regular estimator -- it
inherits `sessions_scope`, it is not a budget) and the Fisher spectrum: rank,
smallest eigenvalue and condition number `λmax/λmin` of the scale-free matrix.

`python -m atomic_information_lab designs --length 5 --pareto` (anchored
parameters, known writes; 486 designs, 84 of full rank, 5 non-dominated):

| design | log det | sessions, at least (CRLB) | energy / session | campaign energy (CRLB) | condition |
|---|--:|--:|--:|--:|--:|
| `fast, precise, reference, reference, precise` (D-optimal) | 4.163 | 1 517 144 | 8.2 | 12.44 M | 2385 |
| `fast, precise, reference, reference, fast` | 4.138 | 1 671 322 | 6.4 | 10.70 M | 1603 |
| `fast, reference, precise, precise, reference` | 3.663 | 1 382 893 | 8.2 | 11.34 M | 2099 |
| `fast, reference, precise, reference, fast` | 3.197 | 1 647 073 | 6.4 | 10.54 M | 1493 |
| `fast, fast, reference, precise, precise` | 0.102 | 1 364 627 | 8.4 | 11.46 M | 168 |

The D-optimal design has neither the smallest bound (11 % above the best)
nor the cheapest bounded campaign (18 % above the best), and it is 14 times
worse conditioned than the last row (finding N-33). Among full-rank designs
the condition number spans 763–35 242 (length 4, 12 designs) and 147–14 337
(length 5, 84 designs): a full rank says the parameters separate, not how
evenly (N-34). All counts are lower bounds for unbiased regular estimators
at this parameter point; the list ranks necessary effort and does not say any
design achieves it. Only the default catalogue and parameter point were
searched.

### Profile likelihood: practical identifiability beyond the local picture (R29 / P1)

The Cramér–Rao counts read the curvature of the log-likelihood at one point.
`likelihood_profile.py` follows Raue et al. (2009,
doi:10.1093/bioinformatics/btp358): fix one parameter at `c`, re-optimise all
others, and see how fast the fit degrades. It does so on the **expected** data
of N sessions (the "Asimov" data set of Cowan et al. 2011,
doi:10.1140/epjc/s10052-011-1554-0), where the likelihood-ratio statistic is
exactly `N · π(c)` with the per-session `π(c) = 2 · min KL(p* ‖ p_c)`. So one
profile answers every N:

* **95 % interval** after N sessions: the connected set around the truth with
  `π(c) ≤ 3.841 / N` (Wilks, one degree of freedom, pointwise);
* **practically identifiable** in Raue's sense once that interval is bounded
  on both sides -- `bounded_sessions` is the smallest such N within the scan
  (64 × the truth or the domain bound);
* **profile count**: the smallest N whose `Δ = 1` interval lies within
  ±10 % -- equal to the Cramér–Rao count if the profile were exactly
  quadratic. `profile/CRLB` measures how far the local picture is off,
  `asymmetry = π(+10 %) / π(−10 %)` in which direction (< 1: flatter above).

Registered calibration design (`fast, reference, precise, reference, fast,
precise`, known writes, fast back-action as anchor), every parameter
(`python -m atomic_information_lab reproduce --table profile-likelihood`,
about eleven minutes; `docs/PROFILE_LIKELIHOOD.json`):

| parameter | sessions, at least (CRLB) | profile count | profile/CRLB | asymmetry | 95 % interval bounded from N = |
|---|--:|--:|--:|--:|--:|
| `k01` | 46 451 | 46 658 | 1.004 | 0.991 | 1 710 |
| `k10` | 51 648 | 52 443 | 1.015 | 0.970 | 1 710 |
| `bias_flip` | 4 708 | 4 867 | 1.034 | 0.934 | 120 |
| `prep_error` | 373 950 | 376 115 | 1.006 | 1.012 | 15 249 |
| `bias_prior` | 128 | 128 | 0.999 | 1.000 | 6 |
| `error:fast` | 870 | 891 | 1.024 | 0.954 | 32 |
| `error:precise` | 17 239 | 17 324 | 1.005 | 0.991 | 654 |
| `error:reference` | 4 902 | 5 075 | 1.035 | 0.932 | 127 |
| `backaction:precise` | 37 442 | 45 971 | 1.228 | 0.648 | 1 756 |
| `backaction:reference` | 290 269 | 312 114 | 1.075 | 0.863 | 4 257 |

What it shows (findings N-36–N-39, all model-internal):

* **The worst-case planning figure holds beyond the local picture.** For
  `prep_error` the profile count is 0.6 % above the Cramér–Rao count, so the
  registered 373 950 sessions are not an artefact of the quadratic
  approximation. All parameters except the two back-action factors stay
  within 3.5 %.
* **The back-action factors are not quadratic.** For `backaction:precise`
  the local picture understates the effort by 23 %, for
  `backaction:reference` by 7.5 %; both profiles are flatter above the truth.
  On the registered design these are exactly the two parameters whose MLE was
  biased upwards (> 3 Monte-Carlo SE) at count/16 (+2.9 %, +7.3 %; post hoc,
  N-37). **A pre-registered transfer to a second design did not hold up**
  (N-43): the back-action effect appeared where predicted and was absent where
  predicted absent, but its size was twice the prediction, and the drift rates,
  predicted unbiased, came out −9 % in a regime where 37 % of the fits failed.
  Read the profile as a diagnostic of *which* parameters are non-quadratic, not
  as a predictor of the MLE's bias.
* **Practical identifiability sets in early but unevenly.** With fewer than 15 249 sessions the 95 % interval of `prep_error` runs to the lower end of the scan (2⁻¹⁰ of the true value) -- the data cannot exclude a perfect preparation; the drift rates need 1 710 and `backaction:reference` 4 257 sessions on that side, while `backaction:precise` needs 1 756 sessions before its interval closes *above* (within 64 × its value). `bias_prior` is bounded from 6 sessions.
  At the registered counts (23 372 and 373 950) every 95 % interval is
  bounded and every profile is unimodal, e.g. `prep_error` at 373 950:
  [0.01606, 0.02390] around 0.02 (-19.7 % / +19.5 %, the Wald interval would be ±19.6 %), and `backaction:reference` at 23 372: [0.246, 1.063] around 0.55 (-55 % / +93 %).
* **Structural controls.** Where `identifiability` knows an exact witness,
  the profile is flat (≤ 1e-31; drift/back-action scale, hidden preparation
  error) or falls back to a second zero (label swap, `π = 3e-32` at the
  swapped point) and is flagged as non-monotone.

Limits (`pl_scope`): expected data -- the intervals here describe the
typical experiment; their coverage on sampled data is measured below; one
parameter point and design; "bounded" means within the scan grid, and counts
read off the grid are conservative; the inner optimisation is local
(continuation in both directions plus a start at the truth; checked against a
global search of the oracle for one nuisance parameter and by a multi-start
benchmark, N-45). 18 of 341 far points are reported as not converged and enter
no figure.

### Intervals from observed data, and why not Wald (R33, pre-registered)

`likelihood_ratio_interval(start, design, free, counts, name)` gives the
likelihood-ratio (profile) interval of one parameter from **observed** outcome
counts, next to the Wald interval (CLI `lr-interval --counts counts.json`). Its
behaviour was measured in a protocol committed before the run
(`docs/PL_FALSIFICATION_PROTOCOL.md`, results `docs/PL_FALSIFICATION_RESULTS.json`),
on the very data of the registered estimator run (same seeds, reproduced bit
for bit):

| N | parameter | Wald coverage | Wald misses below / above | LR coverage | LR misses below / above |
|--:|---|--:|--:|--:|--:|
| 23 372 | `prep_error` | 0.962 | 29 / 9 | 0.962 | 27 / 11 |
| 23 372 | `backaction:precise` | 0.964 | 0 / 36 | 0.954 | 27 / 19 |
| 23 372 | `backaction:reference` | 0.955 | 0 / 45 | 0.952 | 29 / 19 |
| 373 950 | `prep_error` | 0.942 | 32 / 26 | 0.943 | 31 / 26 |
| 373 950 | `backaction:precise` | 0.953 | 18 / 29 | 0.955 | 24 / 21 |
| 373 950 | `backaction:reference` | 0.950 | 16 / 34 | 0.950 | 24 / 26 |

* LR coverage lies in 0.939–0.962 for all ten parameters at both N (screen
  [0.93, 0.97]; finding N-42).
* **Two-sided Wald coverage hid a one-sided failure.** Where the profile is
  skewed, the Wald interval never misses below and misses above 3.6–4.5 % of
  the time instead of 2.5 %; the LR interval splits its misses. The registered
  run's coverage screen (two-sided) could not see this.
* It does not repair every skew: for `prep_error` at 23 372 both intervals
  miss more often below than above (Wald 29 / 9, LR 27 / 11).
* A cross-family review (Codex) found three real defects in this round --
  sparse observed counts, a nuisance start on 0, unfinished restricted fits in
  the coverage runner -- all fixed with tests; the corrected re-run of part A
  keeps every conclusion (findings N-42, N-46).
* So for skewed parameters report the LR interval, and do not read a one-sided
  bound off either interval at small N. Its coverage is a model-internal
  measurement at one parameter point, not a guarantee.

### The continuous line: Kalman filter, STOP and look-ahead (R13/P0, R34, pre-registered)

The earlier R8/R9 rounds used a Gaussian drift model with reference slots and found "no
statistically supported advantage over a fixed reference plan"; they were left out because they
had no exact oracle. `gaussian.py` is the linear-Gaussian analogue of the model above: the
stored value `x` drifts (variance `drift_var · β_a` per reading, back-action speeds it up), the
sensor offset `b` drifts, a target read sees `x + b + noise`, a reference read `b + noise`.
Loss: `α·Var(x0|D) + (1−α)·Var(x|D) + β_ret·E[(x−x0)²]` plus `λ·E_a` per reading; prior
variance 1/4, so the loss before any reading is 0.25 as in the discrete model.

**Adaptivity is worth exactly zero here -- under three conditions** (N-47): linear-Gaussian
dynamics and readings, known parameters, and a loss that depends on the data only through the
posterior covariance (as the variance loss above does). Then the Kalman covariance does not
depend on the data and the Bayes-optimal policy with STOP is an open-loop plan computed in
advance (Meier, Peschon & Dressler 1967, doi:10.1109/TAC.1967.1098668; no dual effect,
Bar-Shalom & Tse 1974, doi:10.1109/TAC.1974.1100635). The third condition is what matters
most: with the **same dynamics** but a 0-1 loss for the sign of `x0`, the terminal risk
`Φ(−|m|/√v)` depends on the posterior mean, and the Bayes-optimal adaptive policy (`h = 2`)
beats the best fixed plan by 0.08 %, 0.50 % and 2.33 % at `λ = 0.004, 0.012, 0.036` (N-50,
`scripts/sign_adaptivity.py`, exploratory; quadrature error ≤ 9e-7, Monte-Carlo check of the
adaptive values). So the discrete model's 9.6 % adaptive gain is not explained by
"non-Gaussian dynamics" alone: its Brier loss `p(1−p)` depends on the posterior itself. Whether
the R9 null result has this cause cannot be checked here (the R9 code is not in this
repository); note that in this model class the difference between an online rule and a fixed
plan is a deterministic number, computable exactly. The covariance fact gives this line its
exact oracle:
`tests/_gaussian_oracle.py` recomputes every posterior by batch least squares in rational
arithmetic (no Kalman recursion), agreement ≤ 1e-12; a generative simulation checks the
reported variance against the empirical error, with a negative control that it can fail.

Default point (`python -m atomic_information_lab reproduce --table gaussian`):

| h | optimum (STOP) | plan | two-step look-ahead | greedy | best forced `h` | fixed reference plan |
|--:|--:|---|--:|--:|--:|--:|
| 1 | 0.1645397 | fast | 0.1645397 | 0.1645397 | 0.1645397 | 0.2651000 |
| 2 | 0.1152415 | reference, precise | 0.1152415 | 0.1182078 | 0.1152415 | 0.1161737 |
| 3 | 0.1051982 | fast, reference, fast | 0.1105364 | 0.1051982 | 0.1051982 | 0.1244557 |
| 6 | 0.1051982 | fast, reference, fast | 0.1105364 | 0.1051982 | 0.1235152 | 0.1252184 |

Pre-registered grid (`docs/GAUSSIAN_PROTOCOL.md`, committed before the run; 27 points of
offset prior × drift × energy price, `h = 6`; `docs/GAUSSIAN_GRID_RESULTS.json`):

* **Look-ahead depth is not monotone** (G2 confirmed; compare the "lookahead pathology" of game
  search, Nau 1983): the two-step rule is worse than the greedy rule at 11 of 27 points; on average it is closer to the optimum (3.1 % vs 4.0 %, G3), but its
  worst case is 11.8 % above the optimum -- the registered bound of 10 % is **refuted** (G4).
  Greedy's worst case is 40.6 %. Neither heuristic dominates (N-48).
* The fixed reference plan `reference, fast, reference, …` is at least 5 % above the optimum at
  26 of 27 points (G5); with a non-small offset prior every optimal plan that reads the target
  also reads the reference (G6).
* Post hoc, descriptive: 22 of 27 optimal plans start with a target read and put the reference
  read after it (bracketing); in 10 of the 15 points where the two-step rule loses, it reads the
  reference first, and in 7 it uses the same readings as the optimum in a worse order (N-49).

Practical consequence inside this model class: the exact search costs `Σ_k |A|^k` covariance
updates (1 092 for three actions and `h = 6`, about 0.03 s; `h = 10`: about 2.6 s), so there is no
reason to use either heuristic. A retention limit on the data-averaged damage
`E[(x−x0)²] ≤ δ` is available (`retention_limit`, the expected reading of F29). It is **not** a
pathwise limit: the posterior damage `E[(x−x0)²|D]` (`conditional_damage`) depends on the data
(e.g. 0.037 or 0.149 after the same two readings) and is unbounded under Gaussian tails, so a
hard pathwise version would admit only readings that carry no information about `x − x0`
(N-51). CLI: `gaussian --horizon 6 [--depth 2] [--retention-limit δ] [--alpha a]
[--retention-weight w]`. Every figure carries `scope = MODEL_INTERNAL_EXACT_LINEAR_GAUSSIAN`.

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
* **Reproducible mutation probe** (`scripts/mutation_probe.py`, its own CI job):
  thirteen model, planner and retention mutants are applied to a copy of `src/`; each
  suite's verdict (`relation`, `guard`, `survives`) is compared with the
  expected one, after a green baseline and a check that the copy is what was
  imported. It records the limits too: M7 (swapped rates) survives the
  metamorphic suite and is killed by the simulation; M8 (decision rule one
  step short, values unchanged) is killed only by the policy cross-evaluation.
* **Separate outputs and Pareto list** (R31): `policy_profile` against the exact
  rational oracle for optimal, forced, greedy and every fixed sequence up to
  length 3, and against explicit hidden-path enumeration; the Pareto list
  against a second, differently written front with soundness and completeness
  checks; Fisher eigenvalues, determinant and condition number against exact
  rational linear algebra (`tests/test_r31_*.py`; mutants M28–M33, all killed
  by assertions).
* **Profile likelihood** (R32, `tests/test_r32_profile_likelihood.py`): the
  profile statistic against the hidden-path oracle with its own global search
  (log grid and golden section) over the nuisance parameter; the local limit
  against the Cramér–Rao curvature with the cubic term measured; flat profiles
  at every exact witness and the label swap as a second zero; the definitions
  of `bounded_sessions`, nested intervals and the Wilks threshold exactly;
  non-converged points never close an interval; the consumer contract of the
  CLI; mutants M34–M38. R33 adds regressions for four defects of the inner
  optimisation found in self-review (M39–M42), the observed-data statistic
  against log-likelihood differences of `fit_mle` (M43), and the
  pre-registered results.
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
python -m atomic_information_lab solve --horizon 4 --profile
python -m atomic_information_lab designs --length 5 --pareto
python -m atomic_information_lab profile-likelihood --design fast,reference,precise --known-write \
    --free k01,backaction:precise --sessions 1000,100000
python -m atomic_information_lab lr-interval --design fast,reference,precise --known-write \
    --free k01,backaction:precise --param k01 --counts counts.json
```

```python
from atomic_information_lab import Problem, TWO_HYPOTHESES, solve, best_open_loop

pr = Problem(TWO_HYPOTHESES, (0.5, 0.5))
print(solve(pr, horizon=4))  # Decision(value=0.18405..., action='fast')
print(best_open_loop(pr, 4))  # (0.18785..., ('fast', 'reference', 'fast'))

from atomic_information_lab import optimal_policy, policy_profile

profile = policy_profile(pr, optimal_policy(pr), 4)
print(profile.bayes_risk, profile.information_initial_bits, profile.p_changed, profile.expected_energy)
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
| R29 / P1 | Does the MLE reach the Cramér–Rao counts? Pre-registered, chain checked from git, scope on every CRLB figure (cross-family R30) | **done, model-internal** (`estimation.py`; findings N-28–N-31) |
| R29-02 | Separate outputs Bayes risk / MI(X0;Y) / P(X_T≠X0) / energy / steps | **done** (`metrics.py`, oracle-verified; finding N-32; same-family, cross-family review pending) |
| R29-03 | Pareto list in `design_search` instead of a single D-optimal winner | **done** (`design_pareto`, Fisher spectrum on every report; findings N-33, N-34; same-family, cross-family review pending) |
| R29 / P1 | Profile likelihood for practical identifiability (Raue et al. 2009, doi:10.1093/bioinformatics/btp358) | **done** (`likelihood_profile.py`: expected-data profiles, observed-data LR intervals; LR coverage measured pre-registered; findings N-36–N-45; same-family) |
| R13 / P0 | Gaussian filter + STOP + two-step look-ahead, fixed reference plan as baseline | **done** (`gaussian.py`: exact open-loop optimum with STOP, rational batch oracle; pre-registered grid G1–G6, G4 refuted; findings N-47–N-51 incl. self-review R34b; same-family, cross-family review pending) |
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
