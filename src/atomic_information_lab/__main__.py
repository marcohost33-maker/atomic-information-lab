"""Command line: python -m atomic_information_lab {reproduce,solve,identify} ..."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys

from .model import DEFAULT_ACTIONS, Hypothesis, Objective, Problem
from .planner import MAX_HORIZON, solve
from .reproduce import (
    identifiability_table,
    retention_table,
    single_model_table,
    two_model_sensitivity,
    two_model_table,
)
from .retention import RETENTION_MODES, solve_retention


def _finite_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"must be finite: {text!r}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="atomic_information_lab", description="Exact optional-stop planning in a synthetic two-state model."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("reproduce", help="print the reference tables as JSON")
    rp.add_argument(
        "--table", choices=("single", "two-model", "sensitivity", "retention", "identifiability", "all"), default="all"
    )
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
    sp.add_argument(
        "--self-audit", action="store_true", help="report internal consistency checks (NOT an independent oracle)"
    )
    sp.add_argument(
        "--retention-limit",
        type=_finite_float,
        default=None,
        metavar="DELTA",
        help="require P(x != x0) <= DELTA (roadmap R13); see --retention-mode",
    )
    sp.add_argument("--retention-mode", choices=RETENTION_MODES, default="pathwise")
    ip = sub.add_parser("identify", help="which parameters calibration sessions can separate (roadmap R13)")
    ip.add_argument("--design", required=True, help="comma-separated action names, e.g. fast,reference,precise")
    ip.add_argument("--known-write", action="store_true", help="the written bit x0 is recorded per session")
    ip.add_argument(
        "--free",
        default="anchored",
        help="'all', 'anchored' (all but backaction:fast) or comma-separated parameter names",
    )
    ip.add_argument("--k01", type=_finite_float, default=0.07)
    ip.add_argument("--k10", type=_finite_float, default=0.045)
    ip.add_argument("--bias-flip", type=_finite_float, default=0.025)
    ip.add_argument("--prep-error", type=_finite_float, default=0.02)
    ip.add_argument("--bias-prior", type=_finite_float, default=0.5)
    ip.add_argument("--target-rse", type=_finite_float, default=0.1)
    return ap


def _identify(args) -> dict:
    from .identifiability import Calibration, Design, identifiability, parameter_names

    cal = Calibration(Hypothesis(args.k01, args.k10, args.bias_flip), DEFAULT_ACTIONS, args.prep_error, args.bias_prior)
    names = parameter_names(cal)
    if args.free == "all":
        free = names
    elif args.free == "anchored":
        free = tuple(n for n in names if n != "backaction:fast")
    else:
        free = tuple(f.strip() for f in args.free.split(","))
    design = Design(tuple(a.strip() for a in args.design.split(",")), args.known_write)
    r = identifiability(cal, design, free, args.target_rse)
    return {"identifiability": dataclasses.asdict(r)}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "reproduce":
            if args.max_horizon is not None and not 0 <= args.max_horizon <= MAX_HORIZON:
                raise ValueError("--max-horizon out of range")
            out = {}
            single_h = 5 if args.max_horizon is None else args.max_horizon
            two_h = 4 if args.max_horizon is None else args.max_horizon
            if args.table in ("single", "all"):
                out["single_model"] = single_model_table(single_h)
            if args.table in ("two-model", "all"):
                out["two_model"] = two_model_table(two_h)
            if args.table in ("sensitivity", "all"):
                out["two_model_sensitivity"] = two_model_sensitivity(two_h)
            if args.table in ("retention", "all"):
                out["retention"] = retention_table(4 if args.max_horizon is None else args.max_horizon)
            if args.table in ("identifiability", "all"):
                out["identifiability"] = identifiability_table()
        elif args.command == "identify":
            out = _identify(args)
        else:
            pr = Problem(
                (Hypothesis(args.k01, args.k10, args.bias_flip),),
                actions=DEFAULT_ACTIONS,
                objective=Objective(args.alpha, args.energy_lambda, args.retention_weight),
            )
            if args.retention_limit is not None:
                if args.forced or args.self_audit:
                    raise ValueError("--retention-limit cannot be combined with --forced or --self-audit")
                r = solve_retention(pr, args.retention_limit, args.horizon, args.retention_mode)
                out = {"horizon": args.horizon, "retention": dataclasses.asdict(r)}
                print(json.dumps(out, indent=2))
                return 0
            d = solve(pr, horizon=args.horizon, allow_stop=not args.forced)
            out = {"value": d.value, "first_action": d.action, "horizon": args.horizon, "forced": args.forced}
            if args.self_audit:
                if args.forced:
                    raise ValueError("--self-audit is defined for optional-stop only; omit --forced")
                from .audit import self_audit

                out["self_audit"] = self_audit(pr, args.horizon)
    except (ValueError, ArithmeticError) as exc:
        hint = "" if args.command == "identify" else f" (horizon range 0..{MAX_HORIZON})"
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
