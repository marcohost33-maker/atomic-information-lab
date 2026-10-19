"""Command line: python -m atomic_information_lab {reproduce,solve} ..."""
from __future__ import annotations

import argparse
import json
import math
import sys

from .model import DEFAULT_ACTIONS, Hypothesis, Objective, Problem
from .planner import MAX_HORIZON, solve
from .reproduce import single_model_table, two_model_sensitivity, two_model_table


def _finite_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"must be finite: {text!r}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="atomic_information_lab",
                                 description="Exact optional-stop planning in a synthetic two-state model.")
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("reproduce", help="print the reference tables as JSON")
    rp.add_argument("--table", choices=("single", "two-model", "sensitivity", "all"), default="all")
    rp.add_argument("--max-horizon", type=int, default=None)
    sp = sub.add_parser("solve", help="solve one single-model instance")
    sp.add_argument("--horizon", type=int, required=True)
    sp.add_argument("--k01", type=_finite_float, default=0.07)
    sp.add_argument("--k10", type=_finite_float, default=0.045)
    sp.add_argument("--bias-flip", type=_finite_float, default=0.025)
    sp.add_argument("--alpha", type=_finite_float, default=0.5)
    sp.add_argument("--energy-lambda", type=_finite_float, default=0.012)
    sp.add_argument("--retention-weight", type=_finite_float, default=0.0)
    sp.add_argument("--forced", action="store_true", help="no STOP before the horizon")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "reproduce":
            out = {}
            single_h = 5 if args.max_horizon is None else args.max_horizon
            two_h = 4 if args.max_horizon is None else args.max_horizon
            if args.table in ("single", "all"):
                out["single_model"] = single_model_table(single_h)
            if args.table in ("two-model", "all"):
                out["two_model"] = two_model_table(two_h)
            if args.table in ("sensitivity", "all"):
                out["two_model_sensitivity"] = two_model_sensitivity(two_h)
        else:
            pr = Problem((Hypothesis(args.k01, args.k10, args.bias_flip),), actions=DEFAULT_ACTIONS,
                         objective=Objective(args.alpha, args.energy_lambda, args.retention_weight))
            d = solve(pr, horizon=args.horizon, allow_stop=not args.forced)
            out = {"value": d.value, "first_action": d.action, "horizon": args.horizon,
                   "forced": args.forced}
    except (ValueError, ArithmeticError) as exc:
        print(f"error: {exc} (horizon range 0..{MAX_HORIZON})", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
