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
| N-40 | Own errors of this round, caught by measurement before release. (1) Fisher scoring in linear coordinates crept along the drift/back-action valley (`k → 0`, `β → ∞` with `k·β` fixed): 7 of 34 points of the `prep_error` profile hit the iteration limit (e.g. 0.1 % too high at `prep_error = 0.26`, after about 1 900 evaluations); log coordinates converge in 6–61 evaluations. (2) A nuisance parameter on its domain bound (e.g. a read error at 0.5) stalled the step halving; now a projected active set. (3) Interval end points were bisected to 1e-4 of the parameter value instead of their distance from the truth; at 10⁹ sessions the interval came out 8 % too wide -- caught by the Wald half-width test. (4) The module name `profile_likelihood` was shadowed by the exported function of the same name; renamed to `likelihood_profile`. (5) One of 341 grid points of the reference table (`bias_flip = 0.381`, statistic 1.21, five orders of magnitude above every interval threshold) does not converge: the read errors run to 0 and a back-action factor to infinity. It is reported (`not_converged`) and enters no figure; a test pins that. | negative result (process) | test history of this round |

Limits: expected data only (no spread over repeated experiments, Wilks' finite-N coverage not
measured -- the next step would be likelihood-ratio intervals on the sampled data of the registered
run); one parameter point and design; "bounded" means within the scan grid (64 × the truth or the
domain bound), and counts read off the grid are conservative; the inner optimisation is local (two
starts per point; one-nuisance cases checked against a global search). Same-family work.

## 9. Not checked

R6 figures, rounds R1–R7, the literature list of the earlier report, any
statement about other projects, Python < 3.12.
