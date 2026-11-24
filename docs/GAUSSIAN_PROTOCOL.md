# R34 pre-registered grid study: Gaussian line (roadmap R13/P0)

**Status:** registered before any grid run. This file, `scripts/gaussian_grid.py` and
`src/atomic_information_lab/gaussian.py` are committed together; the results file
`docs/GAUSSIAN_GRID_RESULTS.json` is produced afterwards by the unchanged runner. Changes after
the run go under *Deviations* below, never into the rules.

## What was seen before registration (exploratory, one point)

At the default point (`prior_var = bias_prior_var = 0.25`, `drift_var = 0.02`,
`bias_drift_var = 0.01`, `energy_lambda = 0.012`, actions fast/precise/reference) the table
`gaussian_table()` for `h = 0..10` was computed once while writing the module:

* optimum 0.105198 from `h = 3` on, plan `fast, reference, fast`;
* greedy (depth 1) reaches the optimum; the two-step look-ahead stops at 0.110536
  (`reference, fast, fast`), 5.1 % above it;
* best forced plan and the fixed reference plan grow with `h` (no stopping).

Nothing else has been computed. The grid below excludes no point.

## Grid

27 points, all other parameters at the defaults, horizon `h = 6`:

| parameter | values |
|---|---|
| `bias_prior_var` | 0.05, 0.25, 1.0 |
| `drift_var` | 0.005, 0.02, 0.08 |
| `energy_lambda` | 0.004, 0.012, 0.036 |

Per point: optimum (`solve_gaussian`), two-step look-ahead (`lookahead_plan(depth=2)`),
greedy (`depth=1`), fixed reference plan `reference_plan(6)` measured to the end.
Relative excess `r(policy) = (value - optimum) / optimum`. "Strictly worse" means a
difference above `1e-9`.

## Predictions and decision rules

| ID | Prediction | Rule |
|---|---|---|
| G1 | The optimum is at most every heuristic at every point (structural; a failure is a bug) | holds iff `optimum <= value + 1e-12` for all 27 x 3 |
| G2 | The two-step look-ahead is strictly worse than greedy at at least 2 points **other than** the default point | confirmed iff count >= 2; otherwise N-48 stays a single-point example |
| G3 | Averaged over the grid, the two-step look-ahead is closer to the optimum than greedy | confirmed iff `mean r(two-step) <= mean r(greedy)` |
| G4 | The two-step look-ahead is never more than 10 % above the optimum | confirmed iff `max r(two-step) <= 0.10` |
| G5 | The fixed reference plan is at least 5 % above the optimum at >= 20 of 27 points | confirmed iff count >= 20 |
| G6 | Where the offset prior is not small (`bias_prior_var >= 0.25`) and the optimal plan reads the target, it also reads the reference | confirmed iff no counter-example among those points |

Every prediction is reported as CONFIRMED or REFUTED; a refuted prediction is a result, not a
reason to change the grid. MC error: none (all figures are exact within the model).

## Deviations

(none yet)

## Outcome (appended after the run; rules above unchanged)

Run at `f1eb353` on a clean tree, results in `docs/GAUSSIAN_GRID_RESULTS.json`:
G1 CONFIRMED; G2 CONFIRMED (10 points besides the default); G3 CONFIRMED (mean excess two-step
3.08 %, greedy 4.04 %); **G4 REFUTED** (max excess of the two-step rule 11.85 % at
`bias_prior_var = 0.25, drift_var = 0.02, energy_lambda = 0.004`); G5 CONFIRMED (26 / 27); G6
CONFIRMED (0 counter-examples). One defect of the runner (operator precedence in the G6 filter)
was found and fixed before the registration commit; the run used the committed runner unchanged.
