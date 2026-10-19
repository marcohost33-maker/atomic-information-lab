"""atomic-information-lab: exact value-of-information and optional stopping in a
synthetic discrete two-state measurement model.

Research reference only. No hardware access, no instrument calibration, no
quantum or atomic-physics validation claim.
"""

from .model import (
    DEFAULT_ACTIONS,
    DEFAULT_HYPOTHESIS,
    STOP,
    TWO_HYPOTHESES,
    Action,
    Hypothesis,
    Objective,
    Problem,
)
from .planner import (
    MAX_HORIZON,
    TIE_TOL,
    Decision,
    PolicyStats,
    best_open_loop,
    evaluate_policy,
    greedy_policy,
    nominal_policy,
    open_loop_value,
    optimal_policy,
    solve,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Hypothesis",
    "Objective",
    "Problem",
    "DEFAULT_ACTIONS",
    "DEFAULT_HYPOTHESIS",
    "TWO_HYPOTHESES",
    "STOP",
    "MAX_HORIZON",
    "TIE_TOL",
    "Decision",
    "PolicyStats",
    "solve",
    "evaluate_policy",
    "optimal_policy",
    "greedy_policy",
    "open_loop_value",
    "best_open_loop",
    "nominal_policy",
    "__version__",
]
