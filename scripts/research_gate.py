"""Independent, reproducible finite-horizon oracle sweep (research-only).

The independent implementation is tests/_oracle.py, which shares no model
source with the package. Its CTMC kernel is produced by uniformisation (float
series), then the conditional policy DP uses rational Fractions in exact mode.
It does NOT constitute an instrument, calibration, or hardware verification.

Two separate measurements, reported separately in ``claims``:

* value agreement -- the package's optimal value equals the oracle's optimal
  value (with and without STOP);
* policy optimality -- the package's decision rule (its own belief filter plus
  its own Bellman choice) is replayed inside the ORACLE's dynamics and scored
  by the oracle; its score must equal the oracle's optimum (with STOP; the
  forced variant is covered in tests/test_policy_cross_evaluation.py). Equal values alone
  would not show this: a wrong decision with a correctly computed value
  elsewhere stays invisible to a value comparison.

Not measured: independence of the model *specification* (the oracle is a
second transcription of the same equations) and any physical validity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Imports below need the sys.path entries above (script run from a checkout).
import _oracle as oracle  # noqa: E402

from atomic_information_lab import STOP, Action, Hypothesis, Objective, Problem, optimal_policy, solve  # noqa: E402

SCHEMA = "ail-independent-oracle-sweep-r4-v2"  # v2: adds claims, policy_comparisons, max_abs_policy_gap
CLAIMS = {
    "value_agreement": "measured",
    "policy_optimality_on_grid": "measured",
    "specification_independence": "not_measured",
    "physical_validity": "not_claimed",
}
ABS_TOL = 3e-11
PACKAGE = "atomic_information_lab"
# Allowlist, not a denylist: anything else the oracle imports counts as a
# possible route back into the package (or into shared code).
ORACLE_ALLOWED_IMPORTS = frozenset({"__future__", "math", "fractions", "itertools"})
_DYNAMIC_IMPORT_NAMES = frozenset({"__import__", "importlib", "exec", "eval", "compile", "__builtins__"})


def oracle_independence(module) -> tuple[bool, str]:
    """MEASURE whether ``module`` (the oracle actually used) is independent of the package.

    Two checks, both must pass:
    1. its source file imports only ``ORACLE_ALLOWED_IMPORTS`` (anywhere, also
       inside functions), uses no dynamic-import names, and names the package
       in no string other than a docstring;
    2. nothing bound in the loaded module object comes from the package
       (catches attributes injected after import).
    Guards against accidental coupling; a deliberately obfuscated import
    (string assembled at run time) is out of reach of a static check.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                docstrings.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {"." * node.level + (node.module or "").split(".")[0]}
        else:
            roots = set()
        if roots - ORACLE_ALLOWED_IMPORTS:
            return False, f"import outside allowlist at line {node.lineno}: {sorted(roots - ORACLE_ALLOWED_IMPORTS)}"
        if isinstance(node, ast.Name | ast.Attribute):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in _DYNAMIC_IMPORT_NAMES:
                return False, f"dynamic import/exec name at line {node.lineno}: {name}"
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and PACKAGE in node.value
            and id(node) not in docstrings
        ):
            return False, f"package name in a non-docstring string at line {node.lineno}"
    for name, value in vars(module).items():
        origin = value.__name__ if inspect.ismodule(value) else getattr(value, "__module__", None)
        if isinstance(origin, str) and origin.split(".")[0] == PACKAGE:
            return False, f"module attribute {name!r} comes from {origin}"
    return True, "imports within allowlist; no package object bound"


def generated_case(rng: random.Random) -> dict:
    # Finite rates comfortably inside the series convergence domain. Include
    # near-zero energy prices and 0.5 sensor-flip stress cases deliberately.
    two = rng.randrange(3) == 0
    count = 2 if two else 1
    rates = []
    for _ in range(count):
        rates.append(
            (round(rng.uniform(0.0, 0.30), 4), round(rng.uniform(0.0, 0.30), 4), rng.choice((0.0, 0.025, 0.10, 0.5)))
        )
    weights = (0.5, 0.5) if two else (1.0,)
    hyps = tuple((w, *r) for w, r in zip(weights, rates, strict=True))
    actions = (
        ("quick", round(rng.uniform(0.1, 1.5), 4), round(rng.uniform(0.01, 0.35), 4), 1.0, False),
        ("ref", round(rng.uniform(0.1, 1.5), 4), round(rng.uniform(0.01, 0.35), 4), 0.8, True),
    )
    return {
        "hyps": hyps,
        "actions": actions,
        "lam": rng.choice((0.0, 0.000001, 0.005, 0.05)),
        "alpha": rng.choice((0.0, 0.5, 1.0)),
        "beta": rng.choice((0.0, 0.15)),
    }


def oracle_side_policy(pkg: Problem, oracle_model, choose):
    """Wrap the package decision rule ``choose(belief, history, h)`` for ``oracle_model.policy_value``.

    The package belief is rebuilt from the observation history with the
    package's own filter, so the oracle scores the complete package pipeline.
    An observation the oracle reaches but the package calls impossible fails
    closed.
    """

    def policy(_message, history, h):
        belief = pkg.initial_belief()
        for name, y in history:
            match = [post for yy, _, post in pkg.branches(belief, pkg.action(name)) if yy == y]
            if not match:
                raise ArithmeticError(f"package calls an oracle-reachable observation impossible: {history}")
            belief = match[0]
        name = choose(belief, history, h)
        return None if name == STOP else next(a for a in oracle_model.actions if a[0] == name)

    return policy


def certify(
    cases: int = 8,
    horizon: int = 2,
    seed: int = 20261020,
    *,
    solve_fn=solve,
    policy_fn=optimal_policy,
    oracle_module=oracle,
) -> dict:
    if type(cases) is not int or not 1 <= cases <= 32:
        raise ValueError("cases must be integer in 1..32")
    if type(horizon) is not int or not 0 <= horizon <= 3:
        raise ValueError("horizon must be integer in 0..3")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    independent, why = oracle_independence(oracle_module)
    if not independent:
        # Fail closed: agreement with a dependent oracle certifies nothing.
        raise RuntimeError(f"oracle not independent of package: {why}")
    rng = random.Random(seed)
    max_residual = 0.0
    max_policy_gap = 0.0
    tested = 0
    policy_tested = 0
    configs = []
    for case_idx in range(cases):
        spec = generated_case(rng)
        pkg = Problem(
            tuple(Hypothesis(k01, k10, flip) for _, k01, k10, flip in spec["hyps"]),
            tuple(w for w, *_ in spec["hyps"]),
            tuple(Action(*act) for act in spec["actions"]),
            Objective(spec["alpha"], spec["lam"], spec["beta"]),
        )
        # Fraction arithmetic in the oracle (except numerical uniformisation).
        oracle_model = oracle_module.Model(
            spec["hyps"], spec["actions"], spec["lam"], spec["alpha"], spec["beta"], exact=True
        )
        for forced in (False, True):
            for h in range(horizon + 1):
                reference = float(oracle_model.value(oracle_model.root(), h, allow_stop=not forced))
                observed = solve_fn(pkg, horizon=h, allow_stop=not forced).value
                if not math.isfinite(reference) or not math.isfinite(observed):
                    raise ArithmeticError(f"non-finite oracle or model: case={case_idx} h={h}")
                residual = abs(observed - reference)
                max_residual = max(max_residual, residual)
                tested += 1
                if residual > ABS_TOL:
                    raise ArithmeticError(
                        f"oracle disagreement: case={case_idx} h={h} "
                        f"forced={forced} observed={observed} oracle={reference} residual={residual}"
                    )
        pkg_policy = oracle_side_policy(pkg, oracle_model, policy_fn(pkg))
        for h in range(1, horizon + 1):
            optimum = oracle_model.value(oracle_model.root(), h)
            scored, _ = oracle_model.policy_value(oracle_model.root(), h, pkg_policy)
            gap = float(scored - optimum)  # >= 0 up to the tie tolerance if the policy is optimal
            if not math.isfinite(gap) or abs(gap) > ABS_TOL:
                raise ArithmeticError(f"policy not optimal under the oracle: case={case_idx} h={h} gap={gap}")
            max_policy_gap = max(max_policy_gap, abs(gap))
            policy_tested += 1
        configs.append(spec)
    canonical = json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": SCHEMA,
        "claim_scope": "synthetic_classical_two_state_only",
        "claims": dict(CLAIMS),
        "oracle_independent_of_package": independent,  # measured above, fail-closed if False
        "kernel_oracle_method": "uniformisation_float_then_rational_DP",
        "seed": seed,
        "cases": cases,
        "horizon": horizon,
        "comparisons": tested,
        "configs_sha256": hashlib.sha256(canonical).hexdigest(),
        "max_abs_residual": max_residual,
        "policy_comparisons": policy_tested,
        "max_abs_policy_gap": max_policy_gap,
        "abs_tolerance": ABS_TOL,
        "verdict": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    try:
        result = certify(args.cases, args.horizon, args.seed)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({"schema": SCHEMA, "verdict": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
