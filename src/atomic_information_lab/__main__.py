"""Command line: python -m atomic_information_lab <command> ... (see --help for the list of commands)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys

from .identifiability import SESSIONS_SCOPE, SESSIONS_SCOPE_TEXT
from .model import DEFAULT_ACTIONS, Hypothesis, Objective, Problem
from .planner import MAX_HORIZON, optimal_policy, solve
from .reproduce import (
    design_pareto_table,
    design_score_row,
    identifiability_table,
    profile_likelihood_table,
    profile_row,
    profile_table,
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
        "--table",
        choices=(
            "single",
            "two-model",
            "sensitivity",
            "retention",
            "identifiability",
            "profile",
            "design-pareto",
            "profile-likelihood",
            "gaussian",
            "all",
        ),
        default="all",
    )
    rp.add_argument("--max-horizon", type=int, default=None)
    gp = sub.add_parser(
        "gaussian", help="linear-Gaussian line: exact plan with STOP vs. look-ahead and reference plan (R13/P0)"
    )
    gp.add_argument("--horizon", type=int, required=True)
    gp.add_argument("--depth", type=int, default=2, help="look-ahead depth of the receding-horizon rule")
    gp.add_argument("--bias-prior-var", type=_finite_float, default=0.25)
    gp.add_argument("--drift-var", type=_finite_float, default=0.02)
    gp.add_argument("--energy-lambda", type=_finite_float, default=0.012)
    gp.add_argument("--retention-limit", type=_finite_float, default=None, help="limit on E[(x - x0)^2]")
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
        "--profile",
        action="store_true",
        help="also report Bayes risk, information about the written bit (bits), P(x != x0), energy and steps "
        "of the returned policy, each on its own (roadmap R29-02); model-internal figures",
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
    ip.add_argument(
        "--target-rse",
        type=_finite_float,
        default=0.1,
        help=f"target relative standard error (default 0.1); {SESSIONS_SCOPE_TEXT}",
    )
    dp = sub.add_parser(
        "designs",
        help="score every fixed calibration design of one length; --pareto keeps the non-dominated ones "
        "(roadmap R29-03)",
    )
    dp.add_argument("--length", type=int, required=True, help="number of readings per session")
    dp.add_argument("--known-write", choices=("both", "yes", "no"), default="both")
    dp.add_argument(
        "--free",
        default="anchored",
        help="'all', 'anchored' (all but backaction:fast) or comma-separated parameter names",
    )
    dp.add_argument(
        "--pareto",
        action="store_true",
        help="only designs that no other design beats on verdict, log det, Cramer-Rao bound and energy at once",
    )
    dp.add_argument("--top", type=int, default=10, help="number of designs printed (0 = all); ignored with --pareto")
    dp.add_argument("--k01", type=_finite_float, default=0.07)
    dp.add_argument("--k10", type=_finite_float, default=0.045)
    dp.add_argument("--bias-flip", type=_finite_float, default=0.025)
    dp.add_argument("--prep-error", type=_finite_float, default=0.02)
    dp.add_argument("--bias-prior", type=_finite_float, default=0.5)
    dp.add_argument(
        "--target-rse",
        type=_finite_float,
        default=0.1,
        help=f"target relative standard error (default 0.1); {SESSIONS_SCOPE_TEXT}",
    )
    lp = sub.add_parser(
        "profile-likelihood",
        help="profile likelihood on the expected data of N sessions: practical identifiability beyond the "
        "local Cramer-Rao picture (roadmap R29 / P1, Raue et al. 2009)",
    )
    lp.add_argument("--design", required=True, help="comma-separated action names, e.g. fast,reference,precise")
    lp.add_argument("--known-write", action="store_true", help="the written bit x0 is recorded per session")
    lp.add_argument(
        "--free",
        default="anchored",
        help="'all', 'anchored' (all but backaction:fast) or comma-separated parameter names",
    )
    lp.add_argument(
        "--param",
        default=None,
        help="comma-separated free parameters to profile (default: all free ones; about 10-80 s each)",
    )
    lp.add_argument(
        "--sessions",
        default="",
        help="comma-separated session counts N at which the likelihood-ratio interval is reported",
    )
    lp.add_argument("--level", type=_finite_float, default=0.95, help="confidence level of the intervals")
    lp.add_argument("--k01", type=_finite_float, default=0.07)
    lp.add_argument("--k10", type=_finite_float, default=0.045)
    lp.add_argument("--bias-flip", type=_finite_float, default=0.025)
    lp.add_argument("--prep-error", type=_finite_float, default=0.02)
    lp.add_argument("--bias-prior", type=_finite_float, default=0.5)
    lp.add_argument(
        "--target-rse",
        type=_finite_float,
        default=0.1,
        help=f"target relative standard error (default 0.1); {SESSIONS_SCOPE_TEXT}",
    )
    ci = sub.add_parser(
        "lr-interval",
        help="likelihood-ratio (profile) interval from OBSERVED session counts, next to the Wald interval (R33)",
    )
    ci.add_argument("--design", required=True, help="comma-separated action names, e.g. fast,reference,precise")
    ci.add_argument("--known-write", action="store_true", help="the written bit x0 is recorded per session")
    ci.add_argument(
        "--free",
        default="anchored",
        help="'all', 'anchored' (all but backaction:fast) or comma-separated parameter names",
    )
    ci.add_argument("--param", required=True, help="comma-separated free parameters to report")
    ci.add_argument(
        "--counts",
        required=True,
        help='JSON file mapping outcomes to counts, e.g. {"1,0,1": 12, ...} (x0 first with --known-write)',
    )
    ci.add_argument("--level", type=_finite_float, default=0.95)
    ci.add_argument("--k01", type=_finite_float, default=0.07, help="start value of the fit")
    ci.add_argument("--k10", type=_finite_float, default=0.045, help="start value of the fit")
    ci.add_argument("--bias-flip", type=_finite_float, default=0.025, help="start value of the fit")
    ci.add_argument("--prep-error", type=_finite_float, default=0.02, help="start value of the fit")
    ci.add_argument("--bias-prior", type=_finite_float, default=0.5, help="start value of the fit")
    return ap


def _read_counts(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("--counts must hold a non-empty JSON object outcome -> count")
    out = {}
    for key, n in raw.items():
        try:
            outcome = tuple(int(x) for x in str(key).split(","))
        except ValueError as exc:
            raise ValueError(f"bad outcome key {key!r}: comma-separated 0/1") from exc
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"count of {key!r} must be an int >= 0")
        out[outcome] = n
    return out


def _lr_interval(args) -> dict:
    from .identifiability import Design
    from .likelihood_profile import DATA_PL_SCOPE, DATA_PL_SCOPE_TEXT, likelihood_ratio_interval

    cal, free = _calibration_and_free(args)
    design = Design(tuple(a.strip() for a in args.design.split(",")), args.known_write)
    counts = _read_counts(args.counts)
    names = tuple(n.strip() for n in args.param.split(","))
    rows = [dataclasses.asdict(likelihood_ratio_interval(cal, design, free, counts, n, args.level)) for n in names]
    return {"lr_intervals": rows, "pl_scope": DATA_PL_SCOPE, "notes": [DATA_PL_SCOPE_TEXT]}


def _profile_likelihood(args) -> dict:
    from .identifiability import Design
    from .likelihood_profile import PL_SCOPE, PL_SCOPE_TEXT, profile_report

    cal, free = _calibration_and_free(args)
    design = Design(tuple(a.strip() for a in args.design.split(",")), args.known_write)
    names = None if args.param is None else tuple(n.strip() for n in args.param.split(","))
    try:
        sessions = tuple(int(n) for n in args.sessions.split(",") if n.strip())
    except ValueError as exc:
        raise ValueError("--sessions must be comma-separated integers") from exc
    rows = profile_report(cal, design, free, names, args.target_rse, sessions, args.level)
    return {
        "profile_likelihood": [profile_row(r) for r in rows],
        "pl_scope": PL_SCOPE,
        "sessions_scope": SESSIONS_SCOPE,
        "notes": [PL_SCOPE_TEXT, SESSIONS_SCOPE_TEXT],
    }


def _calibration_and_free(args):
    from .identifiability import Calibration, parameter_names

    cal = Calibration(Hypothesis(args.k01, args.k10, args.bias_flip), DEFAULT_ACTIONS, args.prep_error, args.bias_prior)
    names = parameter_names(cal)
    if args.free == "all":
        free = names
    elif args.free == "anchored":
        free = tuple(n for n in names if n != "backaction:fast")
    else:
        free = tuple(f.strip() for f in args.free.split(","))
    return cal, free


def _designs(args) -> dict:
    from .identifiability import PARETO_CRITERIA, SESSIONS_SCOPE, design_pareto, design_search

    if args.top < 0:
        raise ValueError("--top must be >= 0")
    cal, free = _calibration_and_free(args)
    modes = {"both": (False, True), "yes": (True,), "no": (False,)}[args.known_write]
    scores = design_search(cal, free, args.length, modes, args.target_rse)
    shown = design_pareto(scores) if args.pareto else (scores if args.top == 0 else scores[: args.top])
    return {
        "designs": {
            "length": args.length,
            "scored": len(scores),
            "full_rank": sum(s.log_det is not None for s in scores),
            "pareto": args.pareto,
            "criteria": list(PARETO_CRITERIA) if args.pareto else None,
            "target_rse": args.target_rse,
            "sessions_scope": SESSIONS_SCOPE,
            "notes": [SESSIONS_SCOPE_TEXT],
            "items": [design_score_row(s) for s in shown],
        }
    }


def _identify(args) -> dict:
    from .identifiability import Design, identifiability

    cal, free = _calibration_and_free(args)
    design = Design(tuple(a.strip() for a in args.design.split(",")), args.known_write)
    r = identifiability(cal, design, free, args.target_rse)
    return {"identifiability": dataclasses.asdict(r)}


def _gaussian(args) -> dict:
    from .gaussian import (
        DEFAULT_GAUSSIAN_MODEL,
        GaussianObjective,
        fixed_plan,
        lookahead_plan,
        reference_plan,
        solve_gaussian,
    )

    m = dataclasses.replace(
        DEFAULT_GAUSSIAN_MODEL,
        bias_prior_var=args.bias_prior_var,
        drift_var=args.drift_var,
        objective=GaussianObjective(energy_lambda=args.energy_lambda),
    )
    opt = solve_gaussian(m, args.horizon, retention_limit=args.retention_limit)
    look = lookahead_plan(m, args.horizon, args.depth, retention_limit=args.retention_limit)
    out = {"horizon": args.horizon, "optimal": dataclasses.asdict(opt), "lookahead": dataclasses.asdict(look)}
    out["lookahead"]["depth"] = args.depth
    if args.retention_limit is None:
        out["reference_plan"] = dataclasses.asdict(fixed_plan(m, reference_plan(args.horizon)))
    return out


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
            if args.table in ("profile", "all"):
                out["profile"] = profile_table(single_h)
            if args.table == "design-pareto":  # not part of "all": 486 designs take several seconds
                out["design_pareto"] = design_pareto_table()
            if args.table == "profile-likelihood":  # not part of "all": ten profiles take several minutes
                out["profile_likelihood"] = profile_likelihood_table()
            if args.table == "gaussian":  # not part of "all": its own model class (R34)
                from .gaussian import MAX_GAUSSIAN_HORIZON, gaussian_table

                gh = 8 if args.max_horizon is None else args.max_horizon
                if not 0 <= gh <= MAX_GAUSSIAN_HORIZON:
                    raise ValueError("--max-horizon out of range")
                out["gaussian"] = gaussian_table(max_horizon=gh)
        elif args.command == "identify":
            out = _identify(args)
        elif args.command == "designs":
            out = _designs(args)
        elif args.command == "profile-likelihood":
            out = _profile_likelihood(args)
        elif args.command == "lr-interval":
            out = _lr_interval(args)
        elif args.command == "gaussian":
            out = _gaussian(args)
        else:
            pr = Problem(
                (Hypothesis(args.k01, args.k10, args.bias_flip),),
                actions=DEFAULT_ACTIONS,
                objective=Objective(args.alpha, args.energy_lambda, args.retention_weight),
            )
            if args.retention_limit is not None:
                if args.forced or args.self_audit or args.profile:
                    raise ValueError("--retention-limit cannot be combined with --forced, --self-audit or --profile")
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
            if args.profile:
                from .metrics import policy_profile

                prof = policy_profile(pr, optimal_policy(pr, allow_stop=not args.forced), args.horizon)
                out["profile"] = dataclasses.asdict(prof)
    except (ValueError, ArithmeticError, OSError) as exc:
        calibration_commands = ("identify", "designs", "profile-likelihood", "lr-interval", "gaussian")
        hint = "" if args.command in calibration_commands else f" (horizon range 0..{MAX_HORIZON})"
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
