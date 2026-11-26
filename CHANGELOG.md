# Changelog

All notable changes to this project are documented here. Format: Keep a
Changelog; versioning: SemVer.

## [Unreleased]

### Fixed -- R35b: self-review of R35 (correction record in FINDINGS §10)
- `GaussianRetentionResult.deterministic` held the best single plan, not the best deterministic
  policy; renamed `best_single_plan`. The deterministic class is now solved exactly
  (`deterministic_value`, `deterministic_damage`, `deterministic_components`: best first reading,
  then a split on the innovation), with a rational oracle and a simulation.
- Claims corrected with exact suprema: single plan up to 52.76 % above the optimum (was "largest
  gap seen 36.6 %"), 41.63 % above the deterministic policy; the coin is worth up to 29.03 %.
  N-53 rescoped (data-dependent beats every fixed plan, not randomised plans).
- New `gaussian_retention_frontier` (the hull for every δ). Test linking the multiplier to the
  retention weight (Gaussian analogue of N-15).
- Beutler & Ross 1985 cited with DOI and scoped as the infinite-horizon analogue (also in
  `retention.py`). Mutant M56; expected verdicts of M54/M55 updated (stronger tests).

### Fixed -- R35: cross-family review of PR #8 (Codex, at `c74cc69`; findings N-52, N-53)
- P2: under the expected limit `E[(x - x0)^2] <= delta` the Gaussian line reported the best
  single plan as the optimum. New `solve_gaussian_retention` (exact over all randomised and
  data-dependent policies: lower convex hull of the plan points, at most two plans, multiplier,
  dual bound over all plans, duality gap, comparison with the best single plan and the
  unconstrained optimum); new `innovation`. `solve_gaussian` / `lookahead_plan` docstrings scope
  their retention limit to single plans. CLI `gaussian --retention-limit`: `optimal` is now the
  exact optimum, the single plan moves to `best_plan_under_limit`.
- N-53: when both hull plans share their first readings, a threshold on the standardised
  innovation realises the mix without a coin (simulation-checked, with a negative control).
- P1: reproduced as the shape of a squash (`2d1080d` is not in this repository); history kept,
  gate kept fail-closed. `verify_preregistration.py` reports the cause under `diagnosis` (also for
  a clone that lacks the registration commit); a test squashes the branch in a scratch clone.
- Mutants M53–M55; `tests/test_r35_retention_mixture.py` (rational pairwise oracle, certificate,
  shape, generative simulation, squash diagnosis).

### Added -- R34: continuous line, Kalman filter + STOP + look-ahead (roadmap R13/P0)
- `gaussian` module: linear-Gaussian analogue of the model (drifting stored value and sensor
  offset, target and reference reads, back-action on the drift); Kalman filter in Joseph form;
  `solve_gaussian` (exact Bayes-optimal plan with STOP, optional retention limit),
  `lookahead_plan` (receding horizon, depth 1 = greedy, depth 2 = two-step look-ahead),
  `reference_plan` / `fixed_plan` (R8/R9 reference-slot baseline), `best_fixed_plan`,
  `gaussian_table`. `scope = MODEL_INTERNAL_EXACT_LINEAR_GAUSSIAN` on every plan.
- Independent oracle `tests/_gaussian_oracle.py`: batch information form in rational arithmetic.
- Pre-registered grid: `docs/GAUSSIAN_PROTOCOL.md`, `scripts/gaussian_grid.py`,
  `docs/GAUSSIAN_GRID_RESULTS.json` (G1–G3, G5, G6 confirmed; G4 refuted). Findings N-47–N-49.
- CLI `gaussian --horizon H [--depth D] [--retention-limit δ]`; `reproduce --table gaussian`
  (not part of `--table all`).
- Mutants M46–M50.

### Fixed -- R34b self-review of R34 (findings N-50, N-51; correction record in FINDINGS §10)
- `gaussian_table` used the default action names and crashed for any other catalogue; it now
  takes the model's own reference and target action (`model_reference_plan`).
- `reproduce --table gaussian --max-horizon N` rejected every `N > 5` (the discrete range check ran
  first) and named the wrong range in the error.
- `GaussianState` accepted invalid states (a negative variance gave a negative loss); it now checks
  finiteness, symmetry, positive semi-definiteness and `damage >= 0` on construction.
- The documented claim that pathwise and expected retention limits coincide was false: the
  posterior damage depends on the data. New `conditional_damage`; `retention_limit` is documented
  as the expected (data-averaged) limit only (N-51).
- N-47 was stated too generally (and claimed to explain the R9 null result, which cannot be
  checked here); now scoped to its three conditions, with the counter-case N-50: under a 0-1 loss
  for the sign of `x0` adaptive stopping beats every fixed plan (`scripts/sign_adaptivity.py`,
  `docs/SIGN_ADAPTIVITY.json`).
- `scripts/verify_preregistration.py --chain r29|r33|r34|all`; CI checks every chain (a squash
  merge of a registration commit now fails CI instead of skipping a test). R33 has no C5 parser and
  says so under `not_checked`.
- CLI `gaussian` gains `--alpha` and `--retention-weight`.
- Mutants M51–M52.
- CHANGELOG: the tagged release 0.1.0 gets its own heading (tag `v0.1.0` → `84cb661`, 2026-11-22);
  `tests/test_changelog_tags.py` keeps every tag and its heading in step.

## [0.1.0] - 2026-11-22

Tag `v0.1.0` (lightweight) points at `84cb661` (merge of #7). No GitHub release and no DOI yet.
Everything below down to the consolidation base was developed after it and before the tag.

### Added -- R33: self-review, observed-data intervals, pre-registered falsification
- `likelihood_ratio_interval` / `likelihood_ratio_statistic` / `DataInterval`: likelihood-ratio
  (profile) intervals from OBSERVED outcome counts, next to the Wald interval; CLI `lr-interval
  --counts counts.json`. `pl_scope = OBSERVED_DATA_LIKELIHOOD_RATIO_WILKS_MODEL_INTERNAL`.
- Pre-registered protocol `docs/PL_FALSIFICATION_PROTOCOL.md` (committed before the run), runner
  `scripts/pl_falsification.py`, predictions `scripts/pl_predict.py` /
  `docs/PL_TRANSFER_PREDICTIONS.json`, results `docs/PL_FALSIFICATION_RESULTS.json`: LR coverage
  0.939–0.962 on the registered data; Wald misses one-sided where the profile is skewed (N-42);
  the transfer of N-37 to a second design is refuted by the registered rule (N-43).
- `scripts/pl_multistart.py` and `docs/PL_MULTISTART.json`: random multi-start benchmark against
  the profile table (N-45).
- Mutants M39–M43.
- CI: the mutation probe runs as its own job (75 min limit); with 45 mutants it outgrew the 20-minute
  limit of the test matrix, which cancelled the job.

### Fixed -- cross-family review of PR #7 (Codex, N-46)
- Observed-data KL now includes outcomes with zero count (their limit `p`); sparse counts gave a
  wrong likelihood-ratio statistic. Mutant M44.
- Nuisance starts on 0 are projected into the open domain and "no information" is decided on the
  linear scale; a zero start was frozen in log coordinates. Mutant M45 (survives by design: two
  independent defences).
- `scripts/pl_falsification.py` retries unfinished restricted fits and reports unresolved cells;
  corrected part A in `docs/PL_FALSIFICATION_RESULTS_A_CORRECTED.json` (conclusions unchanged).

### Fixed -- R33 inner optimisation (N-44)
- The lowest divergence over the starts is kept (a converged higher local minimum no longer
  replaces it); the Newton decrement ends the search only for undamped steps; a step the line
  search shortened never counts as convergence; continuation runs outwards and back inwards.
  Table figures unchanged; 18 far points now reported as not converged.

### Added -- profile likelihood for practical identifiability (R32: roadmap R29 / P1)
- `likelihood_profile` module (`profile_likelihood`, `profile_report`, `profile_statistic`,
  `ParameterProfile`, `LikelihoodInterval`): profile likelihood after Raue et al. (2009) on the
  *expected* data of N sessions (Asimov data set, Cowan et al. 2011). The per-session statistic
  `2 min KL(p* || p)` is computed once; N sessions scale it by N. Reported per parameter: the
  likelihood-ratio analogue of the Cramér–Rao count and their ratio (`profile_to_crlb`),
  `asymmetry`, the smallest N at which the 95 % interval is bounded within the scan
  (`bounded_sessions`), intervals at given N, flatness and outward monotonicity (second valleys).
  Every figure carries `pl_scope = EXPECTED_DATA_PROFILE_MODEL_INTERNAL_ONE_POINT`; the
  Cramér–Rao figures keep `sessions_scope`. Findings N-36–N-40.
- Inner optimisation: projected Fisher scoring in log coordinates with an active set for domain
  bounds and Levenberg damping, two starts per point; a non-converged point is reported and never
  closes an interval.
- CLI `profile-likelihood --design ... [--param ...] [--sessions N,...]`;
  `reproduce --table profile-likelihood` (not part of `--table all`: about six minutes) and its
  output `docs/PROFILE_LIKELIHOOD.json`.
- Mutants M34–M38 (`scripts/mutation_probe.py`).

### Added -- separate outputs and Pareto list (R31: roadmap R29-02, R29-03)
- `metrics.policy_profile` / `PolicyProfile`: Bayes risk and its three
  ingredients, mutual information `I(X0; D_T)` and `I(theta; D_T)` in bits,
  remaining entropy of the current bit, `P(x_T != x0)`, energy, priced energy,
  expected steps and the distribution of the number of readings -- each on its
  own, exact within the model, with `scope = MODEL_INTERNAL_EXACT_FOR_THIS_POLICY`.
  `value == bayes_risk + energy_cost` is checked on every call. Finding N-32.
- `identifiability.design_pareto` and `PARETO_CRITERIA`: the designs no other
  design beats on verdict, `log det`, Cramér–Rao bound and energy per session.
  Finding N-33.
- Fisher spectrum: `IdentifiabilityReport.normalised_eigenvalues` and
  `.condition_number`; `DesignScore.rank`, `.min_normalised_eigenvalue`,
  `.condition_number`, `.campaign_energy_crlb`. All appended with defaults;
  positional construction unchanged. Finding N-34.
- CLI: `solve --profile`, `designs --length N [--pareto] [--top K]`,
  `reproduce --table profile` and `--table design-pareto` (the latter is not
  part of `--table all`: 486 designs take several seconds).
- Mutants M28–M33 (`scripts/mutation_probe.py`), all killed by assertions.

### Changed
- `energy_per_session` is summed with `math.fsum` (correctly rounded, hence
  independent of the order of the readings). A precaution for the Pareto
  comparison, not a repair: no value changed for the default catalogue.
- `reproduce --table identifiability` rows gain `min_normalised_eigenvalue`
  and `condition_number`.

### Changed -- cross-family review of PR #4 @35aef81 (R30: AIL4-01..04, AIL-01..03)
- `ParameterCalibration` gains `sessions_scope` (`CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY`):
  `crlb_rse` reached Python consumers and the results JSON without the lower-bound
  scope. Also `empirical_rse_mcse` (moment-based Monte-Carlo error of the RSE) and
  `coverage_all_low` / `coverage_all_high` (coverage over all replications, failed
  fits as miss / hit). All appended with defaults; positional construction unchanged.
  Finding N-29, N-30.
- `estimation.sd_mcse` (public). Regression fixed before release: with 2–3
  converged fits the new error estimate would have aborted the report.
- `scripts/estimator_calibration.py` writes `provenance` (git HEAD, dirty flag,
  SHA-256 of protocol, runner and estimator) into every result.

### Added -- pre-registration chain and post-hoc analysis (R30)
- `scripts/verify_preregistration.py`: checks C1–C5 of the chain protocol →
  results from git history; negative controls on fabricated histories. CI runs it
  with full history (`fetch-depth: 0`). Finding N-31.
- `scripts/estimator_posthoc.py` and `docs/ESTIMATOR_POSTHOC.json`: exact
  reproduction of every registered field, then RSE-ratio intervals, Wilson
  coverage intervals, all-replication coverage bounds and a family-wise
  coverage band -- labelled post hoc, not confirmatory. Finding N-30.
- Registered results reproduced bit-identically from the registration commit
  `1097c2d` and from `35aef81` (Python 3.13). `docs/ESTIMATOR_CALIBRATION_RESULTS.json`
  is left untouched (pinned by SHA-256 in the tests).
- Mutants M24–M27.

### Changed -- Cramér–Rao scope names its condition (cross-family R29, AIL-R29-01)
- `SESSIONS_SCOPE` is now `CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY` (was
  `CRLB_NECESSARY_LOWER_BOUND_ONLY`): the count is a lower bound for unbiased
  regular estimators only; a biased estimator can have a smaller MSE.
  `SESSIONS_SCOPE_TEXT` says so. Token change before the first release; JSON
  keys unchanged. Finding N-25.
- `Discrimination` gains `sessions_scope` (`CHERNOFF_SUFFICIENT_KNOWN_PAIR_ONLY`)
  and `notes`, appended with defaults (positional construction unchanged):
  the sufficient Chernoff count had no machine-readable scope in the Python
  API. Finding N-26.

### Added -- estimator calibration (R29 P1)
- `estimation.py`: `sample_counts` (exact multinomial), `fit_mle` (Fisher
  scoring, interior only, fail-closed), `estimator_calibration` (bias, RSE,
  RMSE, Wald coverage with Monte-Carlo errors; failed fits counted).
- Pre-registered protocol `docs/ESTIMATOR_PROTOCOL.md`, runner
  `scripts/estimator_calibration.py`, results JSON; `scripts/chernoff_check.py`.
  Findings N-27, N-28. Mutants M19–M23.

### Changed -- session counts say "lower bound" to every consumer (cross-family R28 Q2)
- `SESSIONS_SCOPE_TEXT`: the scope of every Cramér–Rao session count in words
  (lower bound; necessary, not sufficient). `IdentifiabilityReport.notes`
  carries it whenever a count exists, `reproduce --table identifiability` rows
  now pass the report's `notes` through (new key `notes`), and
  `identify --help` states it for `--target-rse`. JSON key names unchanged.
- README/FINDINGS: the Chernoff count of `discrimination` is a sufficient
  count ("after at most 68 sessions"), not a necessary one. Finding N-24.

### Changed -- identifiability scope (cross-family R25, AIL-R25-01)
- `IdentifiabilityReport.scope` and `DesignScore.scope` (`SCOPE[verdict]`), also in
  the CLI JSON and `reproduce --table identifiability`: `GO` is a local statement
  and is never emitted without its scope. Finding N-22.

### Added -- experimental identifiability (roadmap R13/P0)
- `identifiability.py`: `Calibration` (planning model plus `prep_error`,
  `bias_prior`), `Design` (fixed sequence, optional known writes),
  `outcome_distribution`, `fisher_information` (exact forward-mode
  derivatives), `identifiability` (rank, null space, structural witnesses,
  Cramér–Rao session counts, verdict GO / NO-GO / AMBIGUOUS),
  `discrimination` (KL, Chernoff information), `design_search`.
- CLI `identify`, `reproduce --table identifiability`; findings N-16..N-21.
- Limits: local (Rothenberg) statements need interior points; full rank is
  numerical (tolerance `RANK_TOL`), witnesses exact; one parameter point.

### Added -- retention limit (roadmap R13/P0, formula F29)
- `atomic_information_lab.retention`: planning under `P(x != x0) <= delta`,
  `solve_retention(problem, delta, horizon, mode)` with two readings:
  `pathwise` (every STOP has posterior `P(x != x0 | data) <= delta`; exact DP
  with inadmissible states) and `expected` (`P(x_T != x0) <= delta`; exact
  Lagrangian breakpoint search -- the multiplier shifts the R12 retention
  weight --, mixture of at most two deterministic policies, dual bound and
  duality gap). Also: best deterministic policy under the expected limit
  (exact Pareto-front DP, `best_deterministic_retention`), best fixed
  sequence under the limit (`best_open_loop_retention`),
  `retention_profile` (value, `P(x_T != x0)`, probability of stopping above
  `delta`) for any policy, `pathwise_policy`, `lagrangian_policy`. Every
  result reports what the unconstrained optimum would have done (value,
  retention risk, probability of violating the limit). Infeasible starts and
  bad inputs fail closed (`ValueError`); a broken multiplier search raises.
- CLI: `solve --retention-limit DELTA [--retention-mode pathwise|expected]`,
  `reproduce --table retention`.
- `tests/test_retention.py`: exhaustive search over all 7 204 deterministic
  policies in rational arithmetic (`h = 3`, six cases, nine limits), lower
  convex hull for the randomised optimum, independent exact DP on the
  oracle's messages at `h = 4`; mixture, certificate, ordering, monotonicity,
  zero-limit and static-model controls, fail-closed tests, README table
  pinned to the code. Mutation probe gains M9-M13: loosened pathwise limit,
  wrong mixture weight and too coarse Pareto pruning are killed by the
  tests' assertions; multiplier ignored at STOP and a collapsed Pareto front
  are refused loudly by the solver's own guards (bracket failure,
  "infeasible") instead of returning a wrong number.

Integrates the cumulative patch of an external cross-family review
(rounds R2–R9, patch SHA-256 `f47af938…b7c01a4`). Scope is unchanged:
**synthetic classical two-state models only, no worst-case guarantee, no
hardware evidence.** Cross-checks were run on Windows (CPython 3.12/3.13)
locally; hosted CI on this change is still open.

### Added
- `atomic_information_lab.audit.self_audit` and `solve --self-audit`:
  internal consistency checks (Bellman value vs. explicit policy evaluation,
  cost decomposition, dominance over fixed plans, horizon monotonicity).
  Output is marked `independent_oracle: false`; this is not an independent
  verification. `--self-audit --forced` is rejected with exit code 2.
- `atomic_information_lab.robustness.policy_transfer_report`: evaluates a
  frozen nominal policy under each supplied candidate model and reports the
  signed regret against that model's own optimum. Valid only for the supplied
  finite candidate grid; no statement about models outside it.
- `scripts/research_gate.py` (seeded sweep against the independent oracle,
  horizon ≤ 3, absolute tolerance 3e-11) and `scripts/near_tie_gate.py`
  (decision regret near the 1e-12 tie threshold).
- Tests comparing policy-transfer values with the independent rational oracle
  (`tests/_oracle.py`, no import of the package) at horizon 2 and at horizon 3
  (eight seeded cases, seed 2026102009, tolerance 3e-11). The oracle shares
  the model specification and evaluates the package's nominal policy, so it
  confirms the evaluation and the true optimum independently, not the
  specification itself.

- Policy cross-evaluation (`tests/test_policy_cross_evaluation.py`): the
  package's decision rule is replayed inside the independent oracle and
  scored there, against the oracle optimum (exact, `h ≤ 3`; float, `h = 4`)
  and the brute-force minimum over all policies (`h ≤ 3`), with and without
  STOP, on all 13 oracle cases. Before, the tests compared values only and
  scored the package policy with the package itself. Negative controls:
  never-stop, greedy, and a single wrong decision at depth 2.
- `scripts/research_gate.py`: an action name the oracle does not know now
  raises `ArithmeticError` (verdict fail, exit 2) instead of escaping as
  `StopIteration`.
- `scripts/research_gate.py` measures policy optimality as well
  (`policy_comparisons`, `max_abs_policy_gap`) and reports the claim scope
  machine-readably under `claims`: value agreement and policy optimality
  `measured`, specification independence `not_measured`, physical validity
  `not_claimed`. Schema bumped to `ail-independent-oracle-sweep-r4-v2`;
  `certify(policy_fn=...)` takes a mutant for negative controls. A wrong
  policy with correct values (`solve_fn` untouched) makes the gate fail.
- Metamorphic tests (`tests/test_metamorphic.py`), oracle-free, each with a
  control: reading flip, bit relabelling, hypothesis and action permutation,
  split into identical copies, `λ·E` rescaling, monotonicity in `λ`,
  dominated and cheaper action copies.
- Simulation of the generative story (`tests/_simulator.py`,
  `tests/test_generative_simulation.py`): Gillespie Monte Carlo from the
  verbal model, no shared code, compared per reading pattern by binomial
  z-scores; four cases, power controls (swapped rates, reading before the
  evolution, quadrupled bias flip) far above the threshold. It is the first
  check that sees rates swapped consistently everywhere.
- `scripts/mutation_probe.py` (CI step on ubuntu/3.13) and
  `tests/test_mutation_probe.py` (anchor drift guard): eight mutants on a copy
  of `src/`, verdicts classified as `relation`/`guard`/`survives` and
  compared with the expected ones; green baseline and import-path check
  first. Replaces ad-hoc probes in the working tree.
- `oracle_independence(module, allowed=...)`: the allowlist is a parameter
  (default unchanged), so the simulator is checked with the same rule plus
  `random`.
- Belief input contract tests (`tests/test_belief_contract.py`): 20 000
  seeded and several adversarial float-normalised beliefs per size (8, 16,
  32 states) are accepted, both with built-in floats (compensated `sum()`)
  and with plain-loop normalisation of float-subclass entries (the path
  NumPy scalars take); worst deviation `1e-15` at 32 states, bound
  `2·n·2⁻⁵³` (≥ 140× below 1e-12); boundary `5e-13`/`9e-13` accepted, `2e-12` rejected;
  no renormalisation; `±inf`, NaN and `bool` rejected; `-0.0` equals `0.0`;
  posteriors over full planning trees keep the same margin.

### Corrected (within this unreleased change)
- An earlier draft of this entry said six mutants were killed by the
  metamorphic relations. One of them (bias frozen when `x = 1`) produced a
  non-stochastic kernel and was caught by the fail-closed mass guard, not by
  a relation; it is replaced by a stochastic mutant, and the probe now
  reports which mechanism caught each mutant. The quoted 1e-12 margin
  ("about 3000×") was measured with compensated summation only; the
  realistic plain-summation margin is about 1000×.

### Changed
- `validate_belief` documents its contract (reject, never renormalise) and
  the rounding margin; hypothesis weights now use the same `_SUM_TOL`
  constant as beliefs (same value 1e-12, same NaN-safe comparison).
- **Breaking for callers passing their own beliefs:** probability-mass
  tolerance tightened from 1e-9 to 1e-12 (`validate_belief`, outcome masses in
  `branches`). Beliefs that sum to one only within 1e-9 are now rejected.
- `scripts/research_gate.py`: `oracle_independent_of_package` was a constant
  `True`; it is now measured on the oracle module actually used (imports
  only from an allowlist — also inside functions —, no dynamic-import names,
  package name in no non-docstring string, no package object bound in the
  loaded module). A dependent oracle makes `certify` raise and the CLI print
  `verdict: fail` with exit code 2. The output of a passing run is unchanged
  (stdout SHA-256 identical before/after for the default run and for
  `--horizon 3 --cases 4`). A run-time-assembled import inside a function is
  out of reach of this check.

### Fixed
- A NaN or ±inf option in a Bellman minimum now raises `ArithmeticError`
  (before: a NaN alternative was ignored, a NaN first option was returned as
  the minimum, and a -inf alternative won).
- Overflowing action costs and non-finite terminal risks, recursion values,
  policy evaluations and open-loop values fail closed with `ArithmeticError`
  instead of losing silently against STOP.
- Test-only: the R8/R9 negative controls were inline comparisons (forged
  value vs. oracle) and stayed green when the package's `true_optimal_value`
  was off by +1e-10. They now forge the value inside the package (one field at
  a time) and require the shared check `tests/_transfer_check.py`, which the
  positive transfer tests also use, to reject exactly that field; plus a
  within-tolerance (1e-12) acceptance control and a NaN/±inf fail-closed test.
- Test-only: removing the seed of the R8/R9 case grids kept every test green
  (the oracle agrees for any grid). The grids are now pinned by SHA-256, and
  the research-gate test pins `configs_sha256` for seed 20261020 and checks
  that a different seed changes it.

- Test-only: the three CLI subprocess tests in `tests/test_r3_diagnostics.py`
  failed from a plain checkout (`No module named atomic_information_lab`)
  because, unlike `tests/test_cli.py`, they did not set `PYTHONPATH`; they
  passed only with the package installed.

### Removed
- `tests/test_ail_r2_external_replay.py` from the R9 delivery: it was a
  byte-identical copy of `tests/test_ail_hardening_v2.py` (git blob
  `cbdefc3`) and counted 18 tests twice. The suite counts 298, not 316.

### Packaging
- `MANIFEST.in`: the sdist now ships `tests/` (incl. `_oracle.py`,
  `_transfer_check.py`), `scripts/`, `docs/`, `CHANGELOG.md` and
  `CITATION.cff`. Before, its tests did not collect (7 collection errors,
  exit code 2); now pytest inside the unpacked sdist collects and passes 325
  tests (Windows, CPython 3.12). The wheel is unchanged (package modules
  only). `tests/test_packaging_manifest.py` checks the rule on every run.

## Consolidation base of 0.1.0 (R10–R12 core, never tagged on its own)

### Added
- `Problem` (one or several static model hypotheses), `Action`, `Hypothesis`,
  `Objective`.
- Exact planner with optional STOP (`solve`), forced-horizon variant,
  arbitrary-policy evaluation, greedy rule, best fixed sequence, nominal
  (misspecified) policy.
- `python -m atomic_information_lab {solve,reproduce}`.
- Independent test oracle (rational arithmetic, brute force over all
  policies), golden values, controls, fail-closed and CLI tests.

### Fixed (relative to the historic R10–R12 code)
- Overflowing `rate × back-action` is rejected instead of silently dropping
  both observation branches.
- Branch-mass check is NaN-safe.
