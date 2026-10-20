"""Self-audit covers diagnostics only; independent oracle remains separate."""
import json
import subprocess
import sys
import math
import pytest
from atomic_information_lab import Problem, Hypothesis, Objective
from atomic_information_lab.audit import self_audit

@pytest.mark.parametrize("h",[0,1,2,3])
def test_audit_consistency(h):
    x=self_audit(Problem(),h)
    assert x["verdict"]=="internal_consistency_pass"
    assert x["independent_oracle"] is False
    assert x["horizon"]==h
    assert math.isfinite(x["max_residual"])
    assert x["max_residual"]<=x["tolerance"]

@pytest.mark.parametrize("hyp",[Hypothesis(.1,.0,0.0),Hypothesis(.12,.14,.2)])
def test_model_diagnostics_are_nonphysical(hyp):
    x=self_audit(Problem(hypotheses=(hyp,)),2)
    assert x["claim_scope"]=="synthetic_classical_two_state_only"


def test_cli_self_audit_json():
    p=subprocess.run([sys.executable,"-m","atomic_information_lab","solve","--horizon","2","--self-audit"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads(p.stdout)
    assert data["horizon"]==2 and data["self_audit"]["independent_oracle"] is False


def test_cli_without_self_audit_preserves_old_schema():
    p=subprocess.run([sys.executable,"-m","atomic_information_lab","solve","--horizon","1"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert "self_audit" not in json.loads(p.stdout)


def test_diagnostic_detects_wrong_bellman_value(monkeypatch):
    import atomic_information_lab.audit as a
    old=a.solve
    from atomic_information_lab.planner import Decision
    monkeypatch.setattr(a,"solve",lambda *args,**kwargs: Decision(42.0,"stop"))
    with pytest.raises(ArithmeticError):
        a.self_audit(Problem(),2)
    monkeypatch.setattr(a,"solve",old)


def test_self_audit_forced_mode_fails_explicitly():
    p=subprocess.run([sys.executable,"-m","atomic_information_lab","solve","--horizon","2","--self-audit","--forced"],capture_output=True,text=True)
    assert p.returncode==2
    assert "optional-stop" in p.stderr


def test_huge_forced_loss_cannot_hide_wrong_stop_policy(monkeypatch):
    import atomic_information_lab.audit as a
    from atomic_information_lab.planner import PolicyStats
    problem=Problem(objective=Objective(energy_lambda=1e200))
    real=a.evaluate_policy
    def corrupt(*args,**kwargs):
        st=real(*args,**kwargs)
        return PolicyStats(st.value+0.1,st.expected_steps,st.expected_energy,st.expected_terminal_loss+0.1)
    monkeypatch.setattr(a,"evaluate_policy",corrupt)
    with pytest.raises(ArithmeticError,match="Bellman / explicit policy"):
        a.self_audit(problem,1)
