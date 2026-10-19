# Changelog

All notable changes to this project are documented here. Format: Keep a
Changelog; versioning: SemVer.

## [Unreleased]

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
