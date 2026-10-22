# Findings — audit of the R1–R12 research line

This file records, claim by claim, what the consolidation of the earlier research
rounds (R8–R12, 2026-10-17) found. Verdicts: **CONFIRMED**, **REFUTED**,
**PARTIAL**, **UNCHECKED**. "Independent oracle" means `tests/_oracle.py`: exact
rational arithmetic over unnormalised forward messages, transitions by
uniformisation instead of the closed form, and brute force over all policies up
to `h = 3` (7204 policies with STOP, 2187 without). It shares no code with the
audited sources.

Claim boundary: synthetic classical two-state models only. No real atomic
sensor, no quantum or laboratory evidence, no hardware calibration.

## 1. Integrity of the source bundles

| Item | Verdict | Evidence |
|---|---|---|
| SHA-256 manifests of the five bundles (R8 13, R9 9, R10 13, R11 10, R12 13 entries) | CONFIRMED | 58/58 entries match; CRC of all five archives clean |
| R11 archive digest cited in the R12 audit (`e8891a3c…`) | CONFIRMED | recomputed |
| Version chain | CONFIRMED | R9 carries `r8_core` unchanged; R10 carries R8/R9 unchanged plus `r10_*`; R11 carries the R10 oracle unchanged; R12 carries the R10 oracle and the R11 module unchanged |
| Manifests prove authorship | REFUTED (as the audit itself says, F-08) | SHA-256 manifests protect against accidental change, not against a forger; no signature |

## 2. Reproduction (Python 3.12.10)

| Item | Verdict | Evidence |
|---|---|---|
| Original test suites R8 15/15, R9 19/19, R10 17/17, R11 12/12, R12 12/12 | CONFIRMED | 75/75, exit 0 each |
| R10: forcing a measurement at `h = 1` costs `0.2595778083` vs. STOP `0.25` (F-01) | CONFIRMED | independent oracle |
| R11 table `h = 0..5` | CONFIRMED | max deviation 4.4e-16 |
| R11: greedy stop stays at `0.25` for every `h`; `h = 5` optimal `0.1418447352` vs. best fixed sequence `0.1569144984` (F-03) | CONFIRMED | independent oracle |
| R12 table and the 12 sensitivity cases | CONFIRMED | max deviation 1.1e-16 |
| R12: model averaging beats the nominal policy at `h = 4` | REFUTED for this example | rational difference exactly 0 |
| Monte-Carlo figures of R8/R9/R10 with the stored seeds | CONFIRMED | all metrics identical |
| Reported percentages 12.40 %, 9.60 %, 2.02 %; BSC 0.531004 bit and the series 0.5310/0.8624/0.9569/0.9954; 0.015936 in² = 0.10281 cm²; R8 figures; R9 confidence intervals containing 0 | CONFIRMED | recomputed |

## 3. Counter-examples and new findings

| ID | Finding | Verdict | Evidence / treatment here |
|---|---|---|---|
| N-1 | The audit calls the solvers "fail-closed". With `k01 = 1e308`, R10/R11/R12 return a wrong result silently: `rate·3.5` overflows to inf, the posterior becomes NaN, `if z > 0` drops both branches, and only the energy price is reported as risk, `(0.0336, 'precise')`. | REFUTED | this package rejects rate overflow and its mass guard is NaN-safe (tests in `test_validation.py`) |
| N-2 | Risk reduction is not information: with `bias_flip = 0.5`, `λ = 0`, the posterior of `x0` stays exactly 0.5, yet `V₁` drops to `0.2493513986`, because back-action makes the current state more predictable. | new | documented; the objective is end risk, not information gain |
| N-3 | Seeds reproduce the metrics but not the split of actions between REF0 and REF1 (R9: 6 counters, R10: 10 counters differ). The two references tie exactly (difference 0.0); rounding decides. Consistent with F-07. | new | fixed tie rule here: tolerance 1e-12, STOP first, then catalogue order |
| N-5 | The earlier "independent" oracles share code with the code they check (R11 uses `branch()`, R12 imports `matrix`). | new | the oracle here shares nothing with `src/` |
| N-6 | The 15 R11 configurations were computed only at `h = 3`; in 14 the adaptive policy equals the fixed sequence, the only gain is 1.15e-4. | new | stated as a limit of the earlier evidence |
| — | "exact for `h ≤ 5`" | PARTIAL | exact within the model, but not under the overflow of N-1 |
| — | "< 4 s per test run" | PARTIAL | R11 took 7.1 s here |
| — | R12 "negative tests NaN" | PARTIAL | a direct NaN is rejected, a NaN produced by overflow was not |
| F-06 | paired random streams ≠ equal potential outcomes | PARTIAL | consistent, not tested |
| F-01…F-05, F-07 (indirectly), F-08, F-09 | audit findings | CONFIRMED | see sections 2 and 3 |
| PR-01, PR-02 (model-internal: without a reference `V = 0.25` for all `h`), NR-02, NR-8 | earlier pattern records | CONFIRMED | |

Found in this package's own development and fixed (same class "silent NaN /
silent underflow"): the first mass guard was NaN-blind (mutation W04), and the
oracle returned `p01 = 0` for a large `q·t` (underflow of `exp`); both now
refuse. A pure value test could not see swapped `k01`/`k10` (values are
symmetric under relabelling 0↔1); a test now compares posterior marginals
against the oracle (mutation W15).

## 4. R11 follow-up (2026-10-21, same-family; cross-family review pending)

| ID | Finding | Verdict | Evidence / treatment |
|---|---|---|---|
| N-7 | Value tests do not show optimal *decisions*: the package policy was scored by the package itself. A decision rule one step short (`solve()` values unchanged) passed every value comparison against the oracle. | new | package decisions replayed and scored inside the oracle and against brute force (`test_policy_cross_evaluation.py`, gate `policy_optimality_on_grid`); mutant M8 killed |
| N-8 | A consistent transcription error (rates swapped everywhere) is invisible to all metamorphic relations and shared by any oracle written from the same equations. | new | Gillespie simulation of the verbal model (`test_generative_simulation.py`); M7 killed (z 28–82) |
| N-9 | 1e-12 belief tolerance: satisfiable by any float-normalised belief (bound `2·n·2⁻⁵³`), also without compensated `sum()` (float subclasses such as `numpy.float64` are summed plainly by CPython ≥ 3.12). | CONFIRMED | `test_belief_contract.py`, worst `1e-15` at n = 32 |
| N-10 | Three CLI subprocess tests depended on an installed package. | fixed | `PYTHONPATH` as in `test_cli.py` |
| N-11 | A claimed "6/6 mutants killed by metamorphic relations" included one mutant caught only by the mass guard. | REFUTED (own claim) | probe classifies `relation` vs `guard`; mutant replaced |

## 5. R13/P0 retention limit (2026-10-21, same-family; cross-family review pending)

| ID | Finding | Verdict | Evidence |
|---|---|---|---|
| N-12 | The unconstrained optimum corrupts the stored bit it is asked about: `P(x_T != x0) = 0.127` (`h = 3`) and `0.135` (`h = 4`); with a limit of 8 % it stops above the limit in 65 % of the runs. The headline risk 0.146 is therefore an optimistic figure as soon as retention matters. | new | `solve_retention(...).unconstrained_*`, README table |
| N-13 | A hard (pathwise) limit is expensive: for `delta <= 0.08` no reading is admissible (value 0.25 = STOP). | new | exhaustive search `h = 3`, independent exact DP `h = 4` |
| N-14 | Under an average (expected) limit, randomising between two policies is strictly better than every deterministic policy (e.g. 0.173 vs 0.211 at `delta = 0.08`, `h = 4`); the feasible side of the Lagrangian mixture is NOT the best deterministic policy (gap up to 0.058 in the tested grid). | new | Pareto-front DP vs exhaustive deterministic optimum |
| N-15 | The R12 retention weight is exactly the Lagrange multiplier of F29: the expected-limit optimum is the R12 planner with weight `beta + mu*`, mixed. | CONFIRMED | duality gap 0 on all tested limits |

## 6. R13/P0 experimental identifiability (2026-10-22, same-family; cross-family review pending)

Model: the planning model plus a preparation error `prep_error` and a start
probability `bias_prior` of the sensor inversion; sessions of a fixed design,
optionally with the written bit recorded. Default parameters, `prep_error = 0.02`.

| ID | Finding | Verdict | Evidence |
|---|---|---|---|
| N-16 | Drift and back-action are not separable: only `k·β` enters, so `(c·k, β/c)` gives the same distribution for every design. Separation needs an anchor (one back-action factor fixed). | new (NO-GO without anchor) | exact by construction; witness checked in every report; tangent lies in the Fisher null space |
| N-17 | The preparation error carries zero information unless the written bit is recorded (`x_start` uniform for every `prep_error`). | new (NO-GO without known writes) | exact rational equality, control with known writes differs |
| N-18 | Back-action is only visible as window sums between two target reads (the step generators commute). A periodic `fast, reference, precise` design cannot separate the reference from the precise back-action; a back-action factor after the last target read carries no information. | new | exact rational equality with a control that breaks it; window null space by exact elimination |
| N-19 | Without a reference read, `k01 ↔ k10` (with `x ↔ 1−x`, `b ↔ 1−b`) is a global ambiguity although the Fisher matrix is regular; this is the identifiability side of PR-01 (without a reference `V = 0.25`). | new | exact rational equality; reference read as control |
| N-20 | With anchor and known writes, the ten remaining parameters are locally identifiable from four readings per session on (none of the 54 designs of length 3, 12 of 162 of length 4). Cramér–Rao: the best 4-step design needs at least about 6.7 million sessions for 10 % on `prep_error`, the 6-step design of the table at least 374 000 (lower bounds, see N-23). Identifiable in principle, very expensive in practice. | new | exhaustive design search; CRLB checked against an exact inverse |
| N-22 | Cross-family R25 (AIL-R25-01): a bare `GO` invites a global reading. Every verdict now carries a machine-readable `scope` (`LOCAL_ONLY_GLOBAL_UNPROVEN`, `LOCAL_ONLY_GLOBAL_REFUTED_BY_WITNESS`, `NOT_LOCALLY_IDENTIFIABLE`) in the report, the design search, the CLI JSON and the reproduce table. R25's product-only counter-example is N-16. | fixed | `tests/test_r25_scope.py` (5 tests); DK-20261022T220919-6ebfe9f2a8ab 5/5 |
| N-23 | Cross-family R27 (AIL-R27-01): every session count from the Fisher information is a Cramér–Rao bound, i.e. a variance bound for unbiased estimators under regularity. It is necessary, not sufficient; `design_search` ranks designs by it. Report, design search, CLI JSON and reproduce table now carry `sessions_scope = CRLB_NECESSARY_LOWER_BOUND_ONLY`, and README/FINDINGS say "at least". Open: Monte-Carlo calibration of bias/MSE/coverage of an actual estimator, profile likelihood. | fixed (scope), open (calibration) | `tests/test_r27_sessions_scope.py` |
| N-21 | The two R12 hypotheses are indistinguishable from one reading (both predict a uniform reading); four `precise` reads need 68 sessions for a Bayes error ≤ 1 % (Chernoff bound). | new | Chernoff information vs brute-force grid |

Verification: package distribution vs explicit hidden-path enumeration
(uniformisation, not the closed form) and vs chained `Problem.branches`;
Fisher information vs central differences of the oracle; rank threshold
`1e-9` on the scale-free matrix, zero-information threshold with a measured
gap of 26 orders of magnitude; mutants M14–M18 (`scripts/mutation_probe.py`).
Limits: Rothenberg's theorem is local and needs interior points (an error of
0 is refused); a full rank is a numerical statement, a witness an exact one;
only the default action catalogue and parameter point were searched.

## 7. Not checked

R6 figures, rounds R1–R7, the literature list of the earlier report, any
statement about other projects, Python < 3.12.
