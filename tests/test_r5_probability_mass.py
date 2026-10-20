"""Adversarial mass conservation probes, including R4 negative controls."""
import math
import pytest
from atomic_information_lab import Problem, DEFAULT_ACTIONS

def test_small_substochastic_leak_fails_closed():
    p=Problem()
    b=p.initial_belief()
    key=(0, DEFAULT_ACTIONS[0].name)
    rows=list(p._kernel[key]); i=next(i for i,x in enumerate(b) if x>0)
    row=list(rows[i]); row[0] += 2e-10; rows[i]=tuple(row)
    p._kernel[key]=tuple(rows)
    with pytest.raises(ArithmeticError,match='sum to one'):
        p.branches(b,DEFAULT_ACTIONS[0])

def test_normalized_kernel_passes():
    p=Problem(); b=p.initial_belief()
    branches=p.branches(b,DEFAULT_ACTIONS[0]); assert abs(sum(x[1] for x in branches)-1.0)<1e-14

def test_nonunit_belief_rejected():
    p=Problem(); b=list(p.initial_belief()); b[0]+=2e-10
    with pytest.raises(ValueError,match='sum to one'):p.validate_belief(b)
