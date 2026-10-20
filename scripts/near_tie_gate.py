"""Deterministic near-tie decision audit: distinguishes strict optimum from tolerance policy.

The library's tie policy intentionally treats improvements <=1e-12 as ties.
An independent Fraction-DP oracle assesses decision regret in that regime.
Synthetic classical two-state model only; no instrument inference.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
import _oracle
from atomic_information_lab import Problem, Objective, solve
from atomic_information_lab.planner import TIE_TOL, _action_value

SCHEMA='ail-near-tie-gate-r5-v1'
HORIZON=2


def evaluate_lam(lam: float):
    problem=Problem(objective=Objective(0.5,lam,0.0))
    belief=problem.initial_belief()
    stop=problem.terminal_loss(belief)
    act=min(_action_value(problem,belief,a,HORIZON,True) for a in problem.actions)
    return problem,act-stop


def run(*,solve_fn=solve):
    low,high=0.05,0.10
    for _ in range(60):
        mid=(low+high)/2.0
        _, delta=evaluate_lam(mid)
        if delta < 0:low=mid
        else:high=mid
    lam0=(low+high)/2.0
    shifts=(-5e-12,-2e-13,0,2e-13,5e-12)
    rows=[]
    for shift in shifts:
        lam=lam0+shift
        p,gap=evaluate_lam(lam)
        d=solve_fn(p,horizon=HORIZON,allow_stop=True)
        oracle=_oracle.Model(((1.0,0.07,0.045,0.025),),
            tuple((a.name,a.backaction,a.error,a.energy,a.reads_reference) for a in p.actions),
            lam,0.5,0.0,exact=True)
        ref=float(oracle.value(oracle.root(),HORIZON,True))
        if d.action not in ('stop',*(a.name for a in p.actions)):
            raise AssertionError('unknown action')
        if not (math.isfinite(d.value) and math.isfinite(ref)):
            raise AssertionError('non-finite result')
        residual=d.value-ref
        # tolerance-aware regret budget at the root, not a proof of all actions
        if residual < -3e-11 or residual > 1.1e-12:
            raise AssertionError(f'near-tie regret exceeded tolerance at shift={shift}: {residual}')
        if abs(gap) > 2*TIE_TOL and (gap < 0 and d.action=='stop' or gap > 0 and d.action!='stop'):
            raise AssertionError(f'non-tie action disagreement shift={shift}')
        rows.append({'delta_lambda':shift,'decision':d.action,'action_vs_stop_gap':gap,
                     'observed':d.value,'independent_oracle':ref,'regret':residual,
                     'tie_sensitive':abs(gap)<=TIE_TOL})
    return {'schema':SCHEMA,'scope':'synthetic_classical_two_state_only',
            'lambda_threshold':lam0,'tie_tolerance':TIE_TOL,
            'root_cases':len(rows),'results':rows,'verdict':'pass'}


def main():
    print(json.dumps(run(),indent=2,sort_keys=True))
if __name__=='__main__':main()
