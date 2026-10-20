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

### Changed
- **Breaking for callers passing their own beliefs:** probability-mass
  tolerance tightened from 1e-9 to 1e-12 (`validate_belief`, outcome masses in
  `branches`). Beliefs that sum to one only within 1e-9 are now rejected.

### Fixed
- A NaN or ±inf option in a Bellman minimum now raises `ArithmeticError`
  (before: a NaN alternative was ignored, a NaN first option was returned as
  the minimum, and a -inf alternative won).
- Overflowing action costs and non-finite terminal risks, recursion values,
  policy evaluations and open-loop values fail closed with `ArithmeticError`
  instead of losing silently against STOP.

### Removed
- `tests/test_ail_r2_external_replay.py` from the R9 delivery: it was a
  byte-identical copy of `tests/test_ail_hardening_v2.py` (git blob
  `cbdefc3`) and counted 18 tests twice. The suite counts 298, not 316.

### Known limitations
- The sdist does not ship `tests/_oracle.py` or `scripts/`, so its tests do
  not collect (already true for the oracle before this change).

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
