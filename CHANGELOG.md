# Changelog

All notable changes to this project are documented here. Format: Keep a
Changelog; versioning: SemVer.

## [Unreleased]

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
  dominated and cheaper action copies. Six injected model mutants (reference
  reads `x`, cost not `λ·E`, one kernel for all hypotheses, model-posterior
  slice, `k10` ignored, frozen bias) were all killed.
- Belief input contract tests (`tests/test_belief_contract.py`): 20 000
  seeded and several adversarial float-normalised beliefs per size (8, 16,
  32 states) are accepted, with a measured worst deviation `≤ 2·n·2⁻⁵³`
  (≥ 100× below 1e-12); boundary `5e-13`/`9e-13` accepted, `2e-12` rejected;
  no renormalisation; `±inf`, NaN and `bool` rejected; `-0.0` equals `0.0`;
  posteriors over full planning trees keep the same margin.

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

## [0.1.0] - not released

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
