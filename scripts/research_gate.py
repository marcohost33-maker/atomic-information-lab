"""Independent, reproducible finite-horizon oracle sweep (research-only).

The independent implementation is tests/_oracle.py, which shares no model
source with the package. Its CTMC kernel is produced by uniformisation (float
series), then the conditional policy DP uses rational Fractions in exact mode.
It does NOT constitute an instrument, calibration, or hardware verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import _oracle as oracle
from atomic_information_lab import Action, Hypothesis, Objective, Problem, solve

SCHEMA = 'ail-independent-oracle-sweep-r4-v1'
ABS_TOL = 3e-11


def generated_case(rng: random.Random) -> dict:
    # Finite rates comfortably inside the series convergence domain. Include
    # near-zero energy prices and 0.5 sensor-flip stress cases deliberately.
    two = rng.randrange(3) == 0
    count = 2 if two else 1
    rates = []
    for t in range(count):
        rates.append((round(rng.uniform(0.0, 0.30), 4),
                      round(rng.uniform(0.0, 0.30), 4),
                      rng.choice((0.0, 0.025, 0.10, 0.5))))
    weights = (0.5, 0.5) if two else (1.0,)
    hyps = tuple((w, *r) for w, r in zip(weights, rates, strict=True))
    actions = (
        ('quick', round(rng.uniform(0.1, 1.5),4), round(rng.uniform(0.01,0.35),4), 1.0, False),
        ('ref', round(rng.uniform(0.1, 1.5),4), round(rng.uniform(0.01,0.35),4), 0.8, True),
    )
    return {'hyps':hyps,'actions':actions,
            'lam': rng.choice((0.0, 0.000001, 0.005, 0.05)),
            'alpha':rng.choice((0.0,0.5,1.0)),
            'beta':rng.choice((0.0, 0.15))}


def certify(cases: int=8, horizon: int=2, seed: int=20261020, *, solve_fn=solve) -> dict:
    if type(cases) is not int or not 1 <= cases <= 32:
        raise ValueError('cases must be integer in 1..32')
    if type(horizon) is not int or not 0 <= horizon <= 3:
        raise ValueError('horizon must be integer in 0..3')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    rng=random.Random(seed)
    max_residual=0.0
    tested=0
    configs=[]
    for case_idx in range(cases):
        spec=generated_case(rng)
        pkg=Problem(
            tuple(Hypothesis(k01,k10,flip) for _,k01,k10,flip in spec['hyps']),
            tuple(w for w,*_ in spec['hyps']),
            tuple(Action(*act) for act in spec['actions']),
            Objective(spec['alpha'],spec['lam'],spec['beta']))
        # Fraction arithmetic in the oracle (except numerical uniformisation).
        oracle_model=oracle.Model(spec['hyps'], spec['actions'],spec['lam'],spec['alpha'],spec['beta'], exact=True)
        for forced in (False, True):
            for h in range(horizon+1):
                reference=float(oracle_model.value(oracle_model.root(), h, allow_stop=not forced))
                observed=solve_fn(pkg,horizon=h,allow_stop=not forced).value
                if not math.isfinite(reference) or not math.isfinite(observed):
                    raise ArithmeticError(f'non-finite oracle or model: case={case_idx} h={h}')
                residual=abs(observed-reference)
                max_residual=max(max_residual,residual)
                tested+=1
                if residual>ABS_TOL:
                    raise ArithmeticError(f'oracle disagreement: case={case_idx} h={h} '
                                          f'forced={forced} observed={observed} oracle={reference} residual={residual}')
        configs.append(spec)
    canonical=json.dumps(configs,sort_keys=True,separators=(',',':')).encode()
    return {'schema':SCHEMA,'claim_scope':'synthetic_classical_two_state_only',
            'oracle_independent_of_package':True,'kernel_oracle_method':'uniformisation_float_then_rational_DP',
            'seed':seed,'cases':cases,'horizon':horizon,'comparisons':tested,
            'configs_sha256':hashlib.sha256(canonical).hexdigest(),
            'max_abs_residual':max_residual,'abs_tolerance':ABS_TOL,'verdict':'pass'}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=8)
    parser.add_argument('--horizon',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261020)
    args=parser.parse_args()
    try:
        result=certify(args.cases,args.horizon,args.seed)
    except (ValueError, ArithmeticError) as exc:
        print(json.dumps({'schema':SCHEMA,'verdict':'fail','error':str(exc)},sort_keys=True))
        return 2
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
