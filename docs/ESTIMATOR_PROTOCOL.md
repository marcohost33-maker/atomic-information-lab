# Pre-registered protocol: does the MLE reach the Cramér–Rao session counts? (AIL-R29-01, R29 P1)

Written and committed **before** the main run; the results file is produced by
`scripts/estimator_calibration.py` in a later commit. Any change of this
protocol after the run must be listed under "Deviations" below with its reason.

## Question

The identifiability table reports, for the 6-step design with known writes and
the fast back-action as anchor, a Cramér–Rao count of **373 950 sessions** for a
10 % relative standard error on `prep_error` (the worst parameter). The bound
holds for unbiased regular estimators only. Does the maximum-likelihood
estimator, which is what one would actually use, deliver that precision at that
number of sessions — and where does the asymptotic picture break down?

## Fixed setting

* Truth: `Calibration(prep_error=0.02)`, all other parameters at their defaults.
* Design: `fast, reference, precise, reference, fast, precise`, `known_write=True`.
* Free: all ten parameters except `backaction:fast` (the anchor).
* Sessions: N ∈ {23 372, 93 488, 373 950, 1 495 800} (= ⌈373 950 · m⌉ for m = 1/16, 1/4, 1, 4).
* Replications: R = 1000 per N. Seed: `20261104 + index of N` (0..3).
* Estimator: `fit_mle` (Fisher scoring, step halving, interior only),
  started at `true · 1.2` for every free parameter — never at the truth.
* Interval: Wald, `estimate ± 1.959963984540054 · SE`, SE from the expected
  Fisher information at the estimate.

## Endpoints (decided before the run)

Primary, at N = 373 950 for `prep_error`:

* **P1** empirical relative SE ≤ 0.11 (within 10 % of the 0.10 target) →
  "the CRLB count is attained by the MLE at this point, within the model".
  Otherwise: "the CRLB count understates the sessions the MLE needs".

Secondary, at every N, for every parameter:

* **S1** failure rate (non-converged fits, reported by reason);
* **S2** relative bias, flagged when |bias| > 3 MCSE and |bias| > 0.02;
* **S3** coverage of the nominal 95 % interval, flagged outside [0.93, 0.97]
  (≈ 3 MCSE at R = 1000);
* **S4** `rse_ratio` = empirical RSE / CRLB RSE.

## What a result does and does not mean

Model-internal: the data come from the model that is fitted, so model error is
excluded by construction; one parameter point, one design, one estimator.
A pass of P1 does not make the count sufficient elsewhere, for other
estimators, or on an instrument; a fail does not refute the bound (it only
bounds unbiased estimators).

## Deviations

None so far.
