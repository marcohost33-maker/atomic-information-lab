# History of the research line

This repository consolidates an exploratory research line ("Atomic Information
Lab", rounds R1–R12, October 2026). Only the exactly solvable discrete core is
kept; earlier rounds are documented here for provenance. All rounds used
synthetic classical models only.

| Round | Content | Status in this repository |
|---|---|---|
| R1–R2 | Shannon entropy, binary symmetric channel, repeated reads, ideal area per bit (dimension arithmetic) | not included (no code needed) |
| R3–R7 | symmetric two-state CTMC, back-action, greedy information choice, 8-state sensor-bias filter, short exact planning | not included (bundles not audited) |
| R8 | continuous Gaussian AR(1) drift, reference slots, calibration; 15 tests | not included — Monte-Carlo line without exact oracle |
| R9 | calibrated transition rates, actions with energy, online myopic selection; 19 tests; no statistically supported advantage over a fixed reference plan | not included (same reason) |
| R10 | Brier-aligned one-step selection; exact finite-horizon oracle for an 8-state discrete model, **without** a STOP action; 17 tests | discrete oracle superseded by `planner.solve(allow_stop=False)` |
| R11 | exact Bellman recursion with STOP; best fixed sequence, greedy rule; 12 tests | core of `planner` |
| R12 | finite set of static model hypotheses, retention penalty, reference back-action variants; 12 tests | merged into `Problem` (one code path for one or several hypotheses) |
| 0.1.0 | consolidation, independent oracle, fail-closed fixes (rate overflow, NaN-safe mass guard), mutation probes | this repository |
| R13 (P0, part) | hard retention limit `P(x_t != x0) <= delta` (pathwise and expected, F29) | `retention.py` (unreleased) |

## Source bundles (SHA-256 of the archived ZIP files)

| Round | SHA-256 |
|---|---|
| R8 | `1c3c81b5d7f66d54eed6af0e1db37603ac7e8cea8a5e2ebfaf4418cc4d08a79e` |
| R9 | `322bc70a6eb68e2f2e2735db7f57369e19f91d55d56bc0d412816d863c19ed21` |
| R10 | `c939fd7ee4494d3698042376ed57fb41650b12a389aae98f8f9a9002ca78adfe` |
| R11 | `e8891a3cbd05b6382234a884c0b0b2ad46c8b73bd992e7d262fe2bb1090b6234` |
| R12 | `02bc038f68a5b6cbbeb172fb56217378834dddb7f3b0bf79f32b30348780e532` |

The bundles carry unsigned SHA-256 manifests; all 58 entries matched their
files at the 2026-10-19 audit. A hash shows integrity, not authorship.

## What changed in the consolidation

* One model class for single-model (R10/R11) and mixture (R12) problems.
* Rates whose product with back-action overflows are rejected; previously the
  NaN posterior made both observation branches disappear and the planner
  reported only the energy cost as risk.
* Branch masses are checked with a NaN-safe comparison.
* Deterministic tie rule (tolerance 1e-12, STOP first, then catalogue order).
* Test oracle that shares no code with the package; posterior marginals are
  compared, not only values.
