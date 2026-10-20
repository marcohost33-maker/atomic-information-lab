# Atomic Information Lab R3: self-audit scope

Research-only synthetic classical two-state measurement model. Source baseline:
`1afcbe1285deac39ad2db8e6f567c0d3a6c1e16f` + cumulative R2/R3 patch.

New optional CLI command:

```
python -m atomic_information_lab solve --horizon 2 --self-audit
```

The output contains explicit `independent_oracle: false` and
`claim_scope: synthetic_classical_two_state_only` markers. `self_audit()`
checks policy/Bellman internal consistency, cost decomposition, horizon
monotonicity and dominance against fixed plans. It does **not** certify the
mathematics independently, calibrate an instrument, or prove quantum claims.
Use the existing separate rational oracle and adversarial tests for independent
comparisons.

The tolerance is computed from values in each comparison. It deliberately
**does not** use the largest value across forced and optional plans, which
would allow a huge forced-plan cost to hide a serious STOP-policy discrepancy.
`--forced --self-audit` fails rather than silently auditing a different mode.

Open: external calibrated measurements; numerical interval error control;
robustness to model misspecification and observation nonstationarity; audit of
near ties and claims at h=5 on dedicated hardware.
