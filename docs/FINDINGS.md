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

## 6. R13/P0 experimental identifiability (2026-10-22, same-family; cross-family rounds R25–R29 folded in)

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
| N-24 | Cross-family R28 Q2: a field name read back does not prove the consumer chain. Inventory of every consumer of a session count: the CLI JSON (`identify`, `reproduce --table identifiability` and `all`) already carried `sessions_scope` at every count (now checked by walking the parsed output of a real process), but no consumer said it in words -- the CLI printed the counts under the keys `sessions_for_target` / `sessions_for_10pct` with nothing but the token, `identify --help` was silent about `--target-rse`, and the module docstring called the count "the number of sessions needed". Now the report's `notes` carry `SESSIONS_SCOPE_TEXT` ("lower bounds ... necessary, not sufficient") whenever a count exists, the reproduce rows pass `notes` through, and the help text says it. JSON keys unchanged (compatibility). | fixed (consumer text), open (key names, calibration) | `tests/test_r28_consumer_contract.py` |
| N-21 | The two R12 hypotheses are indistinguishable from one reading (both predict a uniform reading); four `precise` reads reach a Bayes error ≤ 1 % after at most 68 sessions (Chernoff upper bound on the error of the likelihood-ratio test: sufficient, not necessary; wording corrected under N-24, the count "need" read as necessary). | new | Chernoff information vs brute-force grid |
| N-25 | Cross-family R29 (AIL-R29-01): the R27 token `CRLB_NECESSARY_LOWER_BOUND_ONLY` called the count "necessary" without its condition. The Cramér–Rao bound limits the variance of *unbiased* regular estimators at one point; a biased estimator can do better there (exact: the constant estimator 1/2 has MSE 0 < p(1−p)/n at p = 1/2, and is biased elsewhere). Token now `CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY`, text names the condition. R28 E1 (rename `sessions_for_*` to `min_sessions_*_crlb`): **not done** -- "min" would restate the universal reading this finding refutes; the keys stay, the scope travels with them. | fixed | `tests/test_r29_estimator_scope.py`; mutant M19 |
| N-26 | R28 Q2 inventory, Python API: `Discrimination.sessions_for_error` (a sufficient count) reached Python consumers with no scope at all; the CLI never prints it, so the process-level check of N-24 could not see it. Now `sessions_scope = CHERNOFF_SUFFICIENT_KNOWN_PAIR_ONLY` plus `notes`; an introspection test requires a scope on every public result type that holds a count (positive control: finds exactly the three types). | fixed | mutant M20 |
| N-27 | R28 Q2 "Chernoff direction": correct. Equal-prior Bayes error `0.5·Σ min(P,Q) ≤ 0.5·Σ P^λ Q^(1−λ)`, so `N ≥ ln(1/(2α))/C` is sufficient. Exact binomial enumeration (two cases): the bound holds at every N, the reported 47 / 162 meet α, the exact minimum is 32 / 112. README pair (68): Monte-Carlo error 0.0021 ± 0.0002 at 68, below 1 % between 40 and 50 sessions. | CONFIRMED (sufficient, ~1.4–1.6× conservative) | exact test; `scripts/chernoff_check.py`; mutant M21 |
| N-28 | R29 P1, pre-registered: at the Cramér–Rao count (373 950) the MLE reaches RSE 0.098 on `prep_error` (target 0.10), all ten parameters unbiased within MC error, Wald coverage 0.940–0.953. From count/4 on the MLE is efficient within MC error; at count/16, 0.8 % of fits fail at the boundary and `backaction:reference` is biased by +7.3 % (RSE ratio 1.13). The count is therefore a good *model-internal* planning figure for the MLE at this point -- not a sufficiency proof elsewhere or on an instrument. Profile likelihood still open. | new (P1 met) | `docs/ESTIMATOR_PROTOCOL.md` committed before `docs/ESTIMATOR_CALIBRATION_RESULTS.json`; mutants M22–M23 |
| N-29 | Cross-family R30 (AIL4-01 / AIL-01): `ParameterCalibration.crlb_rse` -- a Cramér–Rao figure -- reached Python consumers and the results JSON (written by `dataclasses.asdict`) with no `sessions_scope`; the R28 invariant held for the CLI only. Now carries `CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY`; an introspection test requires a scope on every public type with a `crlb` field, a JSON walker on every `crlb` site in `docs/`. The registered results file keeps the gap on purpose (frozen evidence, SHA-256 pinned in the test). | fixed | `tests/test_r30_crossfamily_ail4.py`; mutant M24 |
| N-30 | Cross-family R30 (AIL4-03 / AIL-02) qualifies N-28: "efficient within MC error" was stated without the MC error of the RSE, and coverage at count/16 was over survivors. Post hoc (labelled so): from count/4 on all 30 RSE-ratio 95 % MC intervals contain 1 and no coverage leaves the family-wise band (Bonferroni, 40 cells) or the screen; at count/16, 4 of 10 exclude 1, and `prep_error` lies *below* the bound (0.951 [0.910, 0.993], bias +2.9 %) -- a concrete case of N-25. Coverage over all replications at count/16: within [0.934, 0.964]. P1 margin: 5.2 MC SE. No equivalence margin was registered; the intervals are descriptive. Found on the way: with 2–3 converged fits the new MC error would have aborted the report (fixed before release, mutant M27). | CONFIRMED with stated limits | `docs/ESTIMATOR_POSTHOC.json`; mutants M25–M27 |
| N-31 | Cross-family R30 (AIL4-04 / AIL-03): pre-registration chain, now checked rather than asserted. `scripts/verify_preregistration.py` C1–C5 pass on the real history: protocol in `1097c2d` without results; results first added in descendant `35aef81`; protocol above "Deviations" byte-identical at registration, results commit and head; runner and estimator unchanged; recorded setting = protocol setting. `identifiability.py` changed in between (scope text only), and re-running at `1097c2d` reproduces the registered JSON bit-identically. Not shown by git: wall-clock order (commit dates are self-declared; `7a5075e` and `35aef81` share one timestamp) -- needs an external anchor. Future runs carry `provenance` (HEAD, dirty flag, file digests). | CONFIRMED (content and order); wall-clock UNCHECKED | negative controls on fabricated histories |

Verification: package distribution vs explicit hidden-path enumeration
(uniformisation, not the closed form) and vs chained `Problem.branches`;
Fisher information vs central differences of the oracle; rank threshold
`1e-9` on the scale-free matrix, zero-information threshold with a measured
gap of 26 orders of magnitude; mutants M14–M27 (`scripts/mutation_probe.py`).
Limits: Rothenberg's theorem is local and needs interior points (an error of
0 is refused); a full rank is a numerical statement, a witness an exact one;
only the default action catalogue and parameter point were searched.

## 7. R31 separate outputs and Pareto list (2026-11-18, same-family; cross-family review pending)

Roadmap R29-02 and R29-03, plus the gate of the cross-family round of 2026-11-15 ("Fisher rank,
smallest eigenvalue and condition number as functions of the design"). All figures are
model-internal: default synthetic model, one parameter point, no instrument.

| ID | Finding | Verdict | Evidence |
|---|---|---|---|
| N-32 | The value `solve` returns mixes quantities that move in opposite directions. (a) At `h = 5` the forced planner gains more information about the written bit than the optimum (`I(X0; D)` 0.418 vs 0.387 bit) and is worse on value (0.1619 vs 0.1418), retention (`P(x_T ≠ x0)` 0.199 vs 0.151) and energy (5.03 vs 3.75). (b) At `h = 2` the value-optimal fixed sequence `reference, precise` yields 0.212 bit; `fast, reference` yields 0.290 bit with half the energy and a retention risk of 0.082 instead of 0.186, at a value 0.0027 worse. (c) N-2 measured: with `bias_flip = 0.5`, `I(X0; D) = 0` (≤ 1e-14) at `h = 1..3` while the Bayes risk falls. (d) The optimum at `h = 5` stops after three readings (0.718) or uses all five (0.282), never four. Mutual information is therefore reported next to the risk and is not the objective. | new | `tests/test_r31_policy_profile.py`: exact rational oracle (optimal, forced, greedy, all 39 fixed sequences up to length 3, two-model case with other weights), hidden-path enumeration; mutants M28–M30 |
| N-33 | R29-03: a single D-optimal winner hides the trade-off. Length 5, anchored, known writes: 486 designs, 84 of full rank, 5 non-dominated. The D-optimal design `fast, precise, reference, reference, precise` has a Cramér–Rao bound of 1 517 144 sessions; `fast, fast, reference, precise, precise` has 1 364 627 (−10 %) at `log det` 0.10 instead of 4.16; the smallest bounded campaign energy belongs to `fast, reference, precise, reference, fast` (10.54 M vs 12.44 M energy units, −15 %). Length 4: 3 non-dominated of 12; the D-optimal design needs 49.5 M, `fast, reference, fast, precise` 41.1 M (−17 %). Every figure is a lower bound for unbiased regular estimators (`sessions_scope`); whether an estimator reaches it on these designs was not measured (N-28 measured one design only). | new | `tests/test_r31_design_pareto.py`: second front (sort and sweep), soundness and completeness, extreme designs are members; mutants M31–M32 |
| N-34 | Full rank is not the whole story. The condition number `λmax/λmin` of the scale-free Fisher matrix spans 763–35 242 over the 12 full-rank designs of length 4 and 147–14 337 over the 84 of length 5. The D-optimal design is never the best conditioned one (4074 vs 763; 2385 vs 147). The design with the smallest bound at length 5 has condition 168. Reports and scores now carry rank, eigenvalues and condition number; the condition number is withheld (`None`) exactly when the rank is deficient or a parameter has no information. | new | eigenvalues vs power iteration and exact rational inverse; determinant vs exact rational elimination; mutant M33 |
| N-35 | Own errors of this round, caught by measurement before release. (1) A test asserted that the best length-5 design is more than ten times better conditioned than every length-4 design; measured: 5.2 times (763 vs 147). The claim was dropped, the measured values are pinned. (2) `math.fsum` for the session energy was motivated by order-dependent rounding; measured: the built-in `sum` is already compensated on Python ≥ 3.12 and gave identical values for the default catalogue up to seven readings. The change stays as a precaution and is described as such. (3) Two further test expectations were wrong guesses (a design assumed GO was NO-GO; a prior entropy threshold) and were corrected to measured values, not the code. | negative result (process) | test history of this round |

Limits: one parameter point and the default action catalogue; the Pareto comparison is exact on
stored floats (designs whose criteria differ only by rounding are separated by that rounding); the
profile is exact for a given policy and horizon ≤ 5 only; same-family work, no cross-family round yet.

## 8. R32 profile likelihood (2026-11-21, same-family; cross-family review pending)

Roadmap R29 / P1 (Raue et al. 2009). Profiles on the expected data of N sessions; registered
calibration design `fast, reference, precise, reference, fast, precise`, known writes, fast
back-action as anchor, ten free parameters (`docs/PROFILE_LIKELIHOOD.json`). All figures are
model-internal: default synthetic model, one parameter point, no instrument.

| ID | Finding | Verdict | Evidence |
|---|---|---|---|
| N-36 | The Cramér–Rao count of the worst parameter survives the non-local check: for `prep_error` the likelihood-ratio count is 376 115 against 373 950 (+0.6 %). Over all ten parameters the ratio profile/CRLB lies in 0.999–1.035 except the two back-action factors: `backaction:precise` 1.228 (asymmetry 0.648) and `backaction:reference` 1.075 (0.863), both flatter above the truth. | new | profile against the hidden-path oracle with a global search; local limit against the Cramér–Rao curvature (cubic term measured); mutants M34, M38 |
| N-37 | Post hoc, descriptive, not confirmatory: the two parameters with the largest profile/CRLB ratio are exactly the two with a positive MLE bias above 3 Monte-Carlo SE at count/16 in the registered run (`backaction:precise` +2.85 %, 6.2 SE; `backaction:reference` +7.32 %, 5.8 SE; RSE ratios 1.14 and 1.13), and both have asymmetry < 1 (flatter above ↔ biased upwards). The expected-data profile names the parameters that leave the local picture without a single simulation. One parameter point, one design; the size of the bias is not predicted. | new (descriptive) | `test_post_hoc_the_two_most_non_quadratic_parameters_are_the_two_biased_ones` reads both pinned JSON files |
| N-38 | Practical identifiability (Raue: 95 % interval bounded on both sides) is reached within the scan from N = 6 (`bias_prior`) to N = 15 249 (`prep_error`). Below that, most intervals are open towards 0 (to the end of the scan at 2⁻¹⁰ of the value; e.g. a perfect preparation cannot be excluded), but `backaction:precise` is open *above* until N = 1 756 (scan up to 64 × its value). At both registered counts every interval is bounded and every profile is unimodal on both sides; intervals are asymmetric, e.g. [0.01606, 0.02390] around 0.02 (-19.7 % / +19.5 %, the Wald interval would be ±19.6 %), and `backaction:reference` at 23 372: [0.246, 1.063] around 0.55 (-55 % / +93 %). | new | `bounded_sessions` checked against intervals at N−1 and N; mutant M37 |
| N-39 | Structural witnesses seen by the profile: flat (≤ 1e-31 against `FLAT_TOL` 1e-12) for the drift/back-action scale and the hidden preparation error; the label swap is a second exact zero (`π = 3e-32` at `k01 = 0.045`) and the side is flagged non-monotone. A connected interval alone would hide it. | CONFIRMED (consistent with N-16, N-17, N-19) | `tests/test_r32_profile_likelihood.py`; mutant M38 |
| N-40 | Own errors of this round, caught by measurement before release. (1) Fisher scoring in linear coordinates crept along the drift/back-action valley (`k → 0`, `β → ∞` with `k·β` fixed): 7 of 34 points of the `prep_error` profile hit the iteration limit (e.g. 0.1 % too high at `prep_error = 0.26`, after about 1 900 evaluations); log coordinates converge in 6–61 evaluations. (2) A nuisance parameter on its domain bound (e.g. a read error at 0.5) stalled the step halving; now a projected active set. (3) Interval end points were bisected to 1e-4 of the parameter value instead of their distance from the truth; at 10⁹ sessions the interval came out 8 % too wide -- caught by the Wald half-width test. (4) The module name `profile_likelihood` was shadowed by the exported function of the same name; renamed to `likelihood_profile`. (5) One of 341 grid points of the reference table (`bias_flip = 0.381`, statistic 1.21, five orders of magnitude above every interval threshold) does not converge: the read errors run to 0 and a back-action factor to infinity -- the infimum is not attained in the parameter space (R33 tried snapping such parameters onto the bound; it led into a higher valley elsewhere and was withdrawn). It is reported (`not_converged`) and enters no figure; a test pins that. R33 found more defects of the same family; see N-44. | negative result (process) | test history of this round |

Limits: expected data only (no spread over repeated experiments, Wilks' finite-N coverage not
measured -- the next step would be likelihood-ratio intervals on the sampled data of the registered
run); one parameter point and design; "bounded" means within the scan grid (64 × the truth or the
domain bound), and counts read off the grid are conservative; the inner optimisation is local (two
starts per point; one-nuisance cases checked against a global search). Same-family work.

## 9. R33 self-review and pre-registered falsification (2026-11-22, same-family)

Critical review of R32 (blind spots: Wilks' coverage never measured, N-37 post hoc with two
parameters, inner optimisation checked globally only for one nuisance parameter, convergence
tests). Hypotheses and predictions were committed before any run: `docs/PL_FALSIFICATION_PROTOCOL.md`
and `docs/PL_TRANSFER_PREDICTIONS.json` in `dbee072`, the runner in `01048a6`; results
`docs/PL_FALSIFICATION_RESULTS.json` (provenance: head `01048a6`, bound files clean).

| ID | Finding | Verdict | Evidence |
|---|---|---|---|
| N-41 | In-sample calibration on the registered run, before the protocol: the expected-data profile (`Delta = 1` interval midpoint and half-width, `scripts/pl_predict.py`) predicts the **direction** of the back-action bias at count/16 (+1.8 % / +4.7 % predicted, +2.9 % / +7.3 % measured, within 3 MCSE) and **not** the variance inflation (RSE ratio 1.02 predicted, 1.14 / 1.13 measured). A quantitative claim from the profile shape alone would have been false; the protocol registers it as expected to fail (B5). | negative result | `scripts/pl_predict.py` on the registered design, N = 23 372 |
| N-42 | Pre-registered, part A, on the registered data (A0: reproduced bit for bit, deviation 0.0). The 95 % likelihood-ratio interval covers 0.939–0.962 for all ten parameters at N = 23 372 and 373 950 (A1 pass). Where the profile is skewed the **Wald interval misses on one side only**: `backaction:precise` 0 below / 36 above, `backaction:reference` 0 / 45 at N = 23 372 (upper error 3.6 % and 4.5 % instead of 2.5 %, lower 0 %); the LR interval splits its misses 27 / 19 and 29 / 19 (A3 pass). The registered coverage screen was two-sided and could not see it. `prep_error` at 23 372 misses more below for both intervals (Wald 29 / 9, LR 27 / 11): the LR interval does not repair every skew. Cross-family review (Codex) found that unfinished restricted fits were counted as hits or misses; corrected re-run on the same seeds (`docs/PL_FALSIFICATION_RESULTS_A_CORRECTED.json`): 3 unresolved cells at N = 23 372, conclusions unchanged. | CONFIRMED (A0, A1, A3; also after the correction) | `docs/PL_FALSIFICATION_RESULTS.json` part A; observed-data intervals now in the package (`likelihood_ratio_interval`, CLI `lr-interval`), their statistic checked against log-likelihood differences of `fit_mle` (mutant M43) |
| N-43 | Pre-registered, part B: transfer of N-37 to the 4-step design `fast, precise, reference, precise` (never used in a Monte-Carlo run). B2 pass (at N = 30 894 both back-action factors biased upwards, +18.4 % and +18.5 %, predicted +9.4 % and +9.2 %), B3 pass (no flag at N = 417 942), B6 pass (365 against 9 failed fits). **B1 fails**: at N = 30 894 `k01` and `k10` are biased by −8.9 % and −9.1 % where +0.2 % and +0.7 % were predicted. B4 14 / 19, B5 16 / 19. By the registered decision rule N-37 is **not transferred**; the README no longer claims that the profile predicts the MLE's bias. Post hoc only, not a rescue: the failure lies in the regime the profile itself marks as practically non-identifiable (the `Delta = 1` interval of `prep_error` is open there; 36.5 % of the fits fail at the boundary; the biases are conditional on convergence). | REFUTED (as registered) | `docs/PL_FALSIFICATION_RESULTS.json` part B |
| N-46 | Cross-family review of the consolidated PR #7 (Codex, 2026-11-22), four findings, each verified before acting. (1) P1 real: KL on observed frequencies skipped outcomes with zero count, which is wrong for sparse data (30 sessions: 0.363 instead of 0.592); fixed, mutant M44. (2) P2 real: a nuisance start on 0 (default `prep_error`) was frozen in log coordinates while convergence was certified (0.128 instead of 0.105); fixed by deciding "no information" on the linear scale and projecting a 0 start onto the domain bound. These are two independent defences (the line search also projects every value into the open domain), so the single-edit mutant M45 that removes one of them survives by design -- recorded as a limit, not a kill. Own error in the first fix, caught by the pinned table row: raising every start to >= 1e-8 moved far points of `bias_prior` into a higher valley (1.159 instead of 0.673); the floor is now the bound 1e-12 and no table point is worse (full table regenerated and compared). The corrected part-A run is not affected (it uses `fit_mle` only). (3) P2 real: the falsification runner counted unfinished restricted fits as hits or misses; fixed and part A re-run, conclusions unchanged (N-42). (4) P1 not reproduced: the pre-registration verifier passes on a non-fast-forward merge of the head into `main` with full history; it fails only on a squashed history, which is the intended fail-closed outcome -- hence merge commits, never squash. Expected-data figures were unaffected by (1)–(3). | 3 fixed, 1 not reproduced | review threads on PR #7; `tests/test_r32_profile_likelihood.py` |
| N-44 | Self-review of the R32 inner optimisation, four defects, each now a regression test and a mutant: (1) of two starts the converged one was preferred even when the other was lower -- every feasible point bounds the minimum from above, so the lowest value now wins (an intermediate R33 build showed a higher local minimum, 1.29 against 0.84, on the far side of `k01`); (2) the Newton decrement counted as convergence also for damped steps; (3) a tiny step after many halvings counted as convergence (stagnation reported as an optimum); (4) a single forward continuation can stay in a higher valley -- each side is now swept outwards and back. Effect on the published table: every count, ratio, asymmetry and interval unchanged; nine far points of `k01` lower by about 5e-6 relative; 18 instead of 1 far points honestly reported as not converged (values equal to R32 within 1e-9, all at least 300 times above every interval threshold). | fixed | mutants M39–M42; `tests/test_r32_profile_likelihood.py` |
| N-45 | Multi-start benchmark (`scripts/pl_multistart.py`, `docs/PL_MULTISTART.json`), an attempt to refute the table: at 40 points (for every parameter the points at -r and +r and the outermost point of each side) the inner optimisation ran from 12 random starts each (every nuisance parameter its true value times exp(N(0, 1)), 480 runs, 879 s on one core). No start ended lower than the table by more than 1e-6 relative; the largest improvement was 3.1e-11 (rounding). | CONFIRMED (0 of 40 refuted) | `tests/test_r32_profile_likelihood.py` pins the summary and that every benchmarked value is the table's |

Limits: model-internal throughout; one parameter point; two designs; 1000 replications per cell
(Monte-Carlo error about 0.007 on a coverage); the part-B biases at N = 30 894 are over the 63.5 %
converged fits. Same-family work.

## 10. R34 continuous line: Kalman filter, STOP, look-ahead; R34b self-review (2026-11-24, same-family; cross-family review pending)

Module, oracle, tests, protocol `docs/GAUSSIAN_PROTOCOL.md` and runner `scripts/gaussian_grid.py`
were committed together in `f1eb353` before the grid was computed; results
`docs/GAUSSIAN_GRID_RESULTS.json` in `48048e2` (provenance: head `f1eb353`, clean tree).

| ID | Finding | Verdict | Evidence |
|---|---|---|---|
| N-47 | In the linear-Gaussian analogue with known parameters and a loss that depends on the data only through the posterior covariance, the covariance, and hence every terminal loss, does not depend on the data; the Bayes-optimal policy with STOP is an open-loop plan and the value of adaptivity is exactly 0 (Meier, Peschon & Dressler 1967; no dual effect, Bar-Shalom & Tse 1974). **Scope (corrected in R34b):** the first version said this "explains" the R9 null result and attributed the discrete model's adaptive gain to "non-Gaussian structure". Neither is supported: the R9 code is not in this repository, an online rule and a fixed plan differ by a deterministic amount in this class (G5: the fixed reference plan is >= 5 % worse at 26 / 27 points), and N-50 shows that the loss, not the dynamics, decides. | CONFIRMED within its three conditions | covariance invariance test; package optimum = rational brute force over all plans up to `h = 4`, two models, <= 1e-12; generative simulation (6000 runs, abs(z) < 4) with a negative control (offset prior off by 4x: z > 4) |
| N-48 | Look-ahead depth is not monotone. At the default point the greedy rule is optimal (0.10520) and the two-step rule is 5.1 % worse (exploratory, before registration). On the registered grid: two-step worse than greedy at 11 / 27 points (G2 confirmed, 10 excluding the default point), closer to the optimum on average (3.1 % vs 4.0 %, G3 confirmed), worst case 11.8 % (G4 **refuted**: registered bound 10 %); greedy worst case 40.6 %. | G1–G3, G5, G6 CONFIRMED; G4 REFUTED | `docs/GAUSSIAN_GRID_RESULTS.json`; the test re-runs the grid and compares every row |
| N-49 | Post hoc, descriptive only (not registered): 22 / 27 optimal plans start with a target read and place the reference read after it; where the two-step rule loses (15 points) it reads the reference first in 10 and uses the optimum's readings in a worse order in 7. The order matters because the offset drifts between readings. | post hoc | same file |
| N-50 | Counter-case to an over-general reading of N-47 (R34b, exploratory, not pre-registered; a prototype was seen first). Same dynamics, readings and energies; terminal decision "sign of `x0`", 0-1 loss. The risk of a fixed plan is `1/2 - arctan(sqrt((v0 - v)/v))/pi` (closed form, checked by simulation); the Bayes-optimal adaptive value for `h = 2` follows from the Gaussian martingale of the posterior mean. Adaptive beats the best of all fixed plans of length <= 2 at every price: 0.213575 vs 0.213756 (`lambda = 0.004`, +0.08 %), 0.230658 vs 0.231819 (0.012, +0.50 %), 0.268622 vs 0.275019 (0.036, +2.33 %); quadrature error (step halving) <= 9e-7, at least 200x below each gap. Monte Carlo of the verbal model with the Kalman filter reproduces each adaptive value (abs(z) <= 1.4; its SE 0.002 checks the value, not the gap). Control: the same problem with the variance loss has adaptive = open-loop. | CONFIRMED (exploratory) | `scripts/sign_adaptivity.py`, `docs/SIGN_ADAPTIVITY.json`; tests re-run the quadrature coarser and compare |
| N-51 | The R34 docstring and README said that pathwise and expected retention limits coincide in the Gaussian line. False: the posterior damage `E[(x - x0)^2 | D] = Var(x - x0 | D) + E[x - x0 | D]^2` depends on the data (0.037 vs 0.149 after the same two `fast` readings with data 0/0 and 2/-2); only its average is the deterministic damage (law of total expectation, checked by 20 000 simulations). Under Gaussian tails it is unbounded, so a hard pathwise limit would admit only readings carrying no information about `x - x0`. `retention_limit` is now documented as the expected limit; `conditional_damage` added. | REFUTED (own claim), corrected | `tests/test_r34_gaussian.py` |
| N-52 | Cross-family finding (Codex review of PR #8, P2). `solve_gaussian(..., retention_limit=δ)` was documented as the Bayes-optimal policy under the *expected* limit `E[(x − x0)²] ≤ δ`; it is only the best single plan. Every policy, randomised or data-dependent, induces a distribution over the reading sequences it carries out, and its value and damage are the matching averages of the plan values and damages (posterior variances do not depend on the data; a reading's drift is independent of all earlier decisions). The optimum is the lower convex hull of the plan points at δ: one plan or a mix of two (`solve_gaussian_retention`, with multiplier, dual bound over all plans and duality gap). Codex's case (default model, `h = 1`, δ = 0.01): STOP 0.25 vs ½ STOP + ½ `fast` 0.2072698413. Largest gap seen: 36.6 % above the optimum at `h = 2`, δ = 0.03 (single plan `fast` 0.16454 vs 0.03 STOP + 0.97 `reference, fast` 0.12049). | CONFIRMED (review correct), fixed | `tests/test_r35_retention_mixture.py`: rational oracle that checks every plan and every pair of plans (no hull), two models, `h` ≤ 3, 10 limits each, ≤ 1e-12; dual gap ≤ 1e-10, value non-increasing and convex in δ on 41 limits; mutants M53–M54 |
| N-53 | Refinement of N-47 (exploratory, R35, not pre-registered; found by the prototype of N-52). Under an expected limit, adaptivity is worth more than zero even in this model class: when both hull plans share their first readings, splitting on the standardised innovation `e/√s` of the last shared reading (`Φ(e/√s) < w`) realises the mix without an external coin, as a deterministic, data-dependent rule that beats every single plan. Example: default model, `h = 2`, δ = 0.08: 0.02 `reference, fast` + 0.98 `reference, precise` = 0.115260 vs best single plan `reference, fast` 0.116174 (−0.79 %). Generative simulation of the verbal model, 20 000 runs: value z = −0.97, damage z = 0.66; negative control with swapped weights: damage z = −138. | CONFIRMED (exploratory) | `tests/test_r35_retention_mixture.py`; mutant M55 |

**R34b correction record (self-review of R34, same day).** Besides N-47 (scope) and N-51:
(1) `gaussian_table` crashed for any catalogue without the default action names; (2)
`reproduce --table gaussian --max-horizon N` rejected every `N > 5` and named the wrong range;
(3) `GaussianState` accepted a negative variance (loss -1.0); (4) the README table was first typed
from rounded prototype values (7th digit off; now pinned by a test); (5) the PR text named the
wrong Python version; (6) the R34 registration chain was checked only by a test that *skips* when
the commit is missing, i.e. fail-open under a squash merge -- `verify_preregistration.py --chain
all` now runs in CI; (7) the release state was misreported as "no tag": tag `v0.1.0` exists
(`84cb661`, 2026-11-22) while the CHANGELOG said "not released" (defect A6 of the 2026-11-24
Lagebild, fixed only locally there); (8) Vitus et al. had been cited with an unverified venue
(removed). Each code defect has a regression test; mutants M51–M52 below.

New mutants M46–M50 (reference read sees the target, back-action ignored, STOP removed, Kalman
mean sign, look-ahead executes the wrong action), all killed by relations; M49 leaves every
planning figure exact and is seen only by the oracle means and the generative simulation.
M51 (the determinant check of `GaussianState` removed) and M52 (the table's reference plan
ignores the model's catalogue) reproduce defects (3) and (1) of R34b; M51 is killed by a
relation, M52 by the fail-closed guard (unknown action name), which is how the defect showed.

Limits: model-internal, synthetic, one model class with three actions; the grid varies three of
the eight parameters; `h = 6`. No statement about an instrument.

**R35 correction record (cross-family: Codex review of PR #8 at `c74cc69`).**
P2 (expected retention limit) was correct and is fixed as N-52/N-53: the single-plan result keeps
its function under its correct name (best plan under the limit); the CLI `gaussian
--retention-limit` reports the exact (possibly mixed) optimum as `optimal` and the single plan as
`best_plan_under_limit`. P1 (registration ancestry) was reproduced and needs no history change:
on the pushed branch (`f1eb353 → 48048e2 → 87523e0 → c74cc69`) and on a merge commit of it, the
shape GitHub checks out for `pull_request` (`refs/pull/N/merge`), every chain passes; the review's
commit `2d1080d` (parent `84cb661`) is not in this repository and has the shape of a squash, for
which R34 C2 and C4 fail exactly as reported. The gate is meant to fail there and stays
fail-closed; `verify_preregistration.py` now names the cause under `diagnosis` (and when the
registration commit is missing from a clone), and a test squashes the branch in a scratch clone
and checks both outcomes. Consequence kept from R34: merge with a merge commit, never squash or
rebase. Mutants M53 (mixture weights swapped) and M54 (hull keeps non-convex points) are caught
by the result's own certificate (guard); M55 (shared prefix inverted) by a relation.

## 11. Not checked

R6 figures, rounds R1–R7, the literature list of the earlier report, any
statement about other projects, Python < 3.12.
