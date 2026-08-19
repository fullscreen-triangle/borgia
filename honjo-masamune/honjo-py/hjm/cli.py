"""Command-line runner.

    python -m hjm run script.hj              # honjo
    python -m hjm run plan.ms                # masamune (by extension)
    python -m hjm run -e 'floor 1.0
    C := cut 6' --lang honjo
    python -m hjm translate --format smiles 'c1ccccc1'
    python -m hjm caps

Everything writes JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .honjo import run_honjo
from .masamune import capability, run_plan
from .masamune.capability import CAPABILITY
from .masamune.translate import translate

EXT_LANG = {
    ".hj": "honjo",
    ".honjo": "honjo",
    ".ms": "masamune",
    ".masamune": "masamune",
    ".plan": "masamune",
}


def _detect(path: str) -> str:
    return EXT_LANG.get(os.path.splitext(path)[1].lower(), "honjo")


def _emit(obj, pretty: bool) -> None:
    json.dump(obj, sys.stdout, indent=2 if pretty else None, sort_keys=False)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hjm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a honjo or masamune script")
    run.add_argument("path", nargs="?", help="script file")
    run.add_argument("-e", "--eval", dest="src", help="inline source")
    run.add_argument("--lang", choices=["honjo", "masamune"],
                     help="override language detection")
    run.add_argument("--eps", type=float, default=1e-9,
                     help="target numeric resolution (honjo resolution gate)")
    run.add_argument("--compact", action="store_true", help="single-line JSON")

    tr = sub.add_parser("translate", help="translate one record")
    tr.add_argument("text", help="the record, e.g. a SMILES string")
    tr.add_argument("--format", default="smiles", choices=sorted(CAPABILITY))
    tr.add_argument("--require", default="element,connectivity",
                    help="comma-separated feature requirement")
    tr.add_argument("--floor", type=float, default=1.0)
    tr.add_argument("--compact", action="store_true")

    caps = sub.add_parser("caps", help="print per-format capability sets")
    caps.add_argument("--compact", action="store_true")

    args = ap.parse_args(argv)
    pretty = not getattr(args, "compact", False)

    if args.cmd == "caps":
        _emit({f: sorted(c) for f, c in sorted(CAPABILITY.items())}, pretty)
        return 0

    if args.cmd == "translate":
        req = {s.strip() for s in args.require.split(",") if s.strip()}
        v = translate(args.format, args.text, required=req, floor=args.floor)
        _emit(v.to_dict(), pretty)
        return 0 if v.ok else 1

    # run
    if args.src is None and args.path is None:
        ap.error("give a script path or -e SOURCE")
    if args.src is not None:
        src, base, lang = args.src, os.getcwd(), args.lang or "honjo"
    else:
        with open(args.path, "r", encoding="utf-8") as fh:
            src = fh.read()
        base = os.path.dirname(os.path.abspath(args.path)) or "."
        lang = args.lang or _detect(args.path)

    if lang == "masamune":
        out = run_plan(src, base_dir=base)
        ok = out.get("status") == "ok"
    else:
        out = run_honjo(src, base_dir=base, eps_target=args.eps)
        ok = out.get("status") == "ok"

    out["language"] = lang
    _emit(out, pretty)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
