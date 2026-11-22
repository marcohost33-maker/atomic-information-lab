# Pre-registered protocol: does the expected-data profile hold up on sampled data? (R33)

Written and committed **before** any run of `scripts/pl_falsification.py`. The
predictions in `docs/PL_TRANSFER_PREDICTIONS.json` are committed in the same
commit. Any change after the run goes under "Deviations" with its reason.

## Why

R32 (findings N-36–N-40) profiles the *expected* data only. Two claims are open:

* the 95 % profile (likelihood-ratio) intervals rest on Wilks' asymptotic
  threshold, whose finite-N coverage was never measured;
* N-37 -- "the profile names the parameters whose MLE leaves the local
  picture" -- is post hoc, on one design, with two parameters.

An in-sample calibration on the registered run (before this protocol, reported
as finding N-41) already showed: the profile predicts the **direction** of the
MLE bias of the two back-action factors and underestimates its size by about
40 % (within 3 Monte-Carlo SE); it does **not** predict the variance inflation
(predicted RSE ratio 1.02, measured 1.14 and 1.13). The hypotheses below are
chosen accordingly; B4 and B5 are registered although B5 is expected to fail.

## Fixed setting (both parts)

* Truth: `Calibration(prep_error=0.02)`, all other parameters at their defaults.
* Free: all ten parameters except `backaction:fast` (the anchor).
* Estimator: `fit_mle`, start `true * 1.2` for every free parameter (as registered).
* Replications R = 1000; failed fits are counted, never dropped.
* Flags as in `docs/ESTIMATOR_PROTOCOL.md`: S2 bias flag = |bias| > 3 MCSE and
  |bias| > 0.02; S3 coverage screen = [0.93, 0.97].

## Part A -- likelihood-ratio interval coverage on the registered data

Design `fast, reference, precise, reference, fast, precise`, known writes.
N = 23 372 (seed 20261104) and N = 373 950 (seed 20261106): the seeds of the
registered run, so the outcome counts are identical to it.

Per replication: the full MLE, then for each parameter a restricted MLE with
that parameter fixed at its true value (start: the full estimates);
`Lambda = 2 (loglik_full - loglik_restricted)`. The 95 % likelihood-ratio
interval covers the truth iff `Lambda <= 3.841458820694124`. A restricted fit
above the full one (the full fit was not global) counts `Lambda = 0` and is
reported.

* **A0 (provenance).** The full fits reproduce the registered relative bias,
  empirical RSE and Wald coverage of every parameter to 1e-12. Otherwise the
  data are not the registered data and Part A is void.
* **A1 (coverage).** LR coverage over converged fits within [0.93, 0.97] for
  all ten parameters at both N. Pass / fail per cell.
* **A2.** Coverage over all replications (failed fits as miss / hit) reported.
* **A3 (tail balance, directional).** For `backaction:precise` and
  `backaction:reference` at N = 23 372: |misses below - misses above| of the
  LR interval <= that of the Wald interval. Low power (about 25 misses per
  tail); reported per parameter.

## Part B -- prospective transfer to a design never used in a Monte-Carlo run

Design `fast, precise, reference, precise`, known writes (the D-optimal
4-step design of the identifiability table). Two session counts:

* N = 417 942 = ceil(6 687 058 / 16), the rule of the registered protocol
  (count / 16 of the worst parameter), seed 20261122;
* N = 30 894 = ceil(494 299 / 16), count / 16 of `backaction:reference`, the
  regime in which N-37 was observed, seed 20261123.

Predictions (`docs/PL_TRANSFER_PREDICTIONS.json`, from `scripts/pl_predict.py`,
expected data only): `predicted_bias` = midpoint shift of the `Delta = 1`
profile interval, `predicted_rse_ratio` = its half-width over the Cramér–Rao
standard error. Key values: at N = 30 894, `backaction:precise` +9.4 %,
`backaction:reference` +9.2 %, `error:reference` -3.7 %, `prep_error`
interval open below; at N = 417 942 every |predicted bias| < 0.6 %.

* **B1 (sign).** Every S2-flagged parameter (both N) has the sign of its
  predicted bias. Vacuous if nothing is flagged (reported as such).
* **B2 (effect present).** At N = 30 894, `backaction:precise` and
  `backaction:reference` both have a positive bias above 3 MCSE.
* **B3 (effect absent).** At N = 417 942 no parameter is S2-flagged.
* **B4 (size, uncertain).** |measured - predicted bias| <= 3 MCSE, per
  parameter with a prediction; count reported.
* **B5 (spread, expected to fail).** |rse_ratio - predicted_rse_ratio| <= 3
  MCSE of the ratio, per parameter; count reported.
* **B6 (practical non-identifiability).** More failed fits at N = 30 894
  (where the `prep_error` interval is open) than at N = 417 942.

**Decision rule.** N-37 is called *transferred* iff B1, B2 and B3 pass;
otherwise it is reported as *not transferred* and the README wording is
reduced to the registered design.

## What a result does and does not mean

Model-internal throughout: the data come from the fitted model, one
parameter point, two designs, one estimator. A pass says the expected-data
profile is a useful predictor *in this model*, not on an instrument.

## Deviations

None. The run used exactly the setting above (R = 1000, the four seeds above, runner committed in
`01048a6` before the run; provenance in the results: head `01048a6`, bound files clean).

## Outcome (added after the run)

* A0 pass (both N reproduce the registered run bit for bit, deviation 0.0); A1 pass (LR coverage
  0.939–0.962 in all 20 cells); A3 pass for both parameters (Wald 0 / 36 and 0 / 45, LR 27 / 19 and
  29 / 19 below / above).
* B1 **fail** (`k01`, `k10` biased −8.9 % and −9.1 % at N = 30 894 against predicted +0.2 % and
  +0.7 %); B2 pass (+18.4 %, +18.5 %); B3 pass; B4 14 / 19; B5 16 / 19; B6 pass (365 against 9
  failed fits). **Verdict by the decision rule: N-37 not transferred** (FINDINGS N-43).
