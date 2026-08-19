"""The honjo language: lexer, parser, and evaluator.

One primitive, the cut, at several arities:

    floor 1.0
    C   := cut 6                    -- arity one: individuate
    OH  := O ~ H : 1                -- arity two: bond, one committed cell
    ring:= deloc ring(c1,...,c6) cells: 9
    W   := close O(H, H)            -- closure
    p   := track x in W until converge yield amalgamation
    g   := import graph "f.smi" require supplied < 0.1 unless refuse

Every value carries a floor and a provenance tag.  Provenance composes as
the maximum under stated < supplied, so no operation launders supplied
data into a stated result.  Every operation that can fail returns a
labelled verdict; only ``cut`` carries a value.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from ..core.chem import (
    configuration, covalent_valence, geometry, symbol, valence, vacancy,
)
from ..core.graph import MEDIUM, Prov
from ..core.verdict import Label, Verdict

KEYWORDS = {
    "floor", "cut", "close", "track", "until", "yield", "when", "emit",
    "observe", "in", "as", "let", "medium", "converge", "diverge", "with",
    "by", "import", "module", "assert", "deloc", "refuse",
    "require", "unless", "stated", "supplied", "cells", "graph", "reps",
}

TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>--[^\n]*)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<op><=|>=|==|!=|:=|[~:#(),.<>{}])
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


class HonjoError(Exception):
    def __init__(self, msg: str, line: int):
        super().__init__(f"line {line}: {msg}")
        self.msg, self.line = msg, line


@dataclass
class Tok:
    kind: str
    text: str
    line: int


def lex(src: str) -> list[Tok]:
    out, pos, line = [], 0, 1
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            raise HonjoError(f"unexpected character {src[pos]!r}", line)
        kind, text = m.lastgroup, m.group()
        line += text.count("\n")
        pos = m.end()
        if kind in ("ws", "comment"):
            continue
        if kind == "ident" and text in KEYWORDS:
            kind = "kw"
        out.append(Tok(kind, text, line))
    out.append(Tok("eof", "", line))
    return out


# ---------------------------------------------------------------- values


@dataclass
class Value:
    """An accountable value: a cut, its floor, and its provenance."""

    type: str
    floor: float
    residue: float
    prov: Prov
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.floor <= 0:
            raise HonjoError(f"floor must be positive, got {self.floor}", 0)
        if self.residue < self.floor:
            raise HonjoError(
                f"residue {self.residue:.6g} below floor {self.floor:.6g}", 0
            )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "floor": self.floor,
            "residue": round(self.residue, 9),
            "provenance": str(self.prov),
            **self.data,
        }


# ---------------------------------------------------------------- parser


@dataclass
class Node:
    kind: str
    line: int
    args: dict = field(default_factory=dict)


class Parser:
    def __init__(self, toks: list[Tok]):
        self.t, self.i = toks, 0

    def peek(self, k: int = 0) -> Tok:
        return self.t[min(self.i + k, len(self.t) - 1)]

    def next(self) -> Tok:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str, text: str | None = None) -> Tok:
        tok = self.next()
        if tok.kind != kind or (text is not None and tok.text != text):
            raise HonjoError(f"expected {text or kind!r}, got {tok.text!r}", tok.line)
        return tok

    def accept(self, text: str) -> bool:
        if self.peek().text == text:
            self.next()
            return True
        return False

    def parse(self) -> list[Node]:
        out = []
        while self.peek().kind != "eof":
            out.append(self.statement())
        return out

    def statement(self) -> Node:
        tok = self.peek()
        if tok.text == "floor":
            self.next()
            return Node("floor", tok.line, {"value": float(self.expect("number").text)})
        if tok.text == "observe":
            self.next()
            name = self.expect("ident").text
            return Node("observe", tok.line, {"name": name})
        if tok.text == "assert":
            self.next()
            cond = self.cond()
            msg = None
            if self.accept("emit"):
                msg = json.loads(self.expect("string").text)
            return Node("assert", tok.line, {"cond": cond, "message": msg})
        if tok.text == "let" or (tok.kind == "ident" and self.peek(1).text == ":="):
            if tok.text == "let":
                self.next()
            name = self.expect("ident").text
            self.expect("op", ":=")
            return Node("bind", tok.line, {"name": name, "expr": self.expr()})
        raise HonjoError(f"unexpected {tok.text!r}", tok.line)

    def expr(self) -> Node:
        left = self.primary()
        while self.peek().text == "~":
            line = self.next().line
            right = self.primary()
            cells = 1
            if self.accept(":"):
                cells = int(float(self.expect("number").text))
            guard = None
            if self.accept("when"):
                guard = self.cond()
            left = Node("bond", line,
                        {"a": left, "b": right, "cells": cells, "guard": guard})
        return left

    def primary(self) -> Node:
        tok = self.peek()
        if tok.text == "cut":
            self.next()
            arg = self.next()
            return Node("cut", tok.line, {"z": arg.text, "z_kind": arg.kind})
        if tok.text == "deloc":
            self.next()
            nxt = self.next()
            if nxt.text != "ring":
                raise HonjoError(f"expected 'ring', got {nxt.text!r}", nxt.line)
            self.expect("op", "(")
            members = [self.expect("ident").text]
            while self.accept(","):
                members.append(self.expect("ident").text)
            self.expect("op", ")")
            cells = None
            if self.accept("cells"):
                self.expect("op", ":")
                cells = int(float(self.expect("number").text))
            return Node("deloc", tok.line, {"members": members, "cells": cells})
        if tok.text == "close":
            self.next()
            centre = self.expect("ident").text
            self.expect("op", "(")
            ligands = []
            if self.peek().text != ")":
                ligands.append(self._ligand())
                while self.accept(","):
                    ligands.append(self._ligand())
            self.expect("op", ")")
            return Node("close", tok.line, {"centre": centre, "ligands": ligands})
        if tok.text == "track":
            self.next()
            item = self.expect("ident").text
            self.expect("kw", "in")
            proc = self.expect("ident").text
            reps = []
            if self.accept("with"):
                self.expect("kw", "reps")
                reps.append(self.expect("ident").text)
                while self.accept(","):
                    reps.append(self.expect("ident").text)
            self.expect("kw", "until")
            admit = self.next().text
            self.expect("kw", "yield")
            yields = self.expect("ident").text
            return Node("track", tok.line,
                        {"item": item, "process": proc, "reps": reps,
                         "admit": admit, "yield": yields})
        if tok.text == "import":
            self.next()
            self.expect("kw", "graph")
            path = json.loads(self.expect("string").text)
            req = None
            if self.accept("require"):
                if self.accept("stated"):
                    req = ("stated", None)
                else:
                    self.expect("kw", "supplied")
                    op = self.expect("op").text
                    val = float(self.expect("number").text)
                    req = ("supplied", (op, val))
            recovery = None
            if self.accept("unless"):
                recovery = self.next().text
            return Node("import", tok.line,
                        {"path": path, "require": req, "recovery": recovery})
        if tok.kind == "ident":
            self.next()
            return Node("ref", tok.line, {"name": tok.text})
        if tok.kind == "number":
            self.next()
            floor = None
            if self.accept("#"):
                floor = float(self.expect("number").text)
            return Node("num", tok.line, {"value": float(tok.text), "floor": floor})
        raise HonjoError(f"unexpected {tok.text!r}", tok.line)

    def _ligand(self) -> dict:
        name = self.expect("ident").text
        cells = 1
        if self.accept(":"):
            cells = int(float(self.expect("number").text))
        return {"name": name, "cells": cells}

    def cond(self) -> dict:
        lhs = self.next().text
        if self.peek().text == ".":
            self.next()
            lhs = f"{lhs}.{self.next().text}"
        op = self.expect("op").text
        rhs_tok = self.next()
        rhs = float(rhs_tok.text) if rhs_tok.kind == "number" else rhs_tok.text
        return {"lhs": lhs, "op": op, "rhs": rhs}


def _cmp(a, op, b) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        a, b = str(a), str(b)
    return {"<": a < b, ">": a > b, "<=": a <= b,
            ">=": a >= b, "==": a == b, "!=": a != b}[op]


# ------------------------------------------------------------ evaluator


class Interpreter:
    """Executes honjo.  Evaluation mutates state and advances the clock."""

    def __init__(self, base_dir: str = ".", eps_target: float = 1e-9):
        self.base_dir = base_dir
        self.eps_target = eps_target
        self.floor = 1.0
        self.M = 0                      # committed-cut count
        self.env: dict[str, Value] = {}
        self.log: list[dict] = []
        self.trace: list[dict] = []

    # -- entry ----------------------------------------------------------

    def run(self, src: str) -> dict:
        try:
            prog = Parser(lex(src)).parse()
        except HonjoError as e:
            return {"status": "parse-error", "error": str(e),
                    "cut_count": 0, "bindings": [], "trace": []}

        status = "ok"
        for node in prog:
            if status != "ok":
                break
            try:
                status = self._exec(node) or status
            except HonjoError as e:
                status = "error"
                self.log.append({"level": "error", "line": node.line,
                                 "message": str(e)})

        return {
            "status": status,
            "floor": self.floor,
            "target_resolution": self.eps_target,
            "cut_count": self.M,
            "bindings": [
                {"name": k, **v.to_dict()} for k, v in self.env.items()
            ],
            "trace": self.trace,
            "log": self.log,
        }

    def _exec(self, node: Node) -> str | None:
        if node.kind == "floor":
            f = node.args["value"]
            # the resolution gate: refuse rather than run below it
            if f < self.eps_target:
                self.log.append({
                    "level": "refusal", "line": node.line,
                    "message": "ambient floor finer than target resolution",
                    "verdict": Verdict.subfloor(f, self.eps_target,
                                                "floor declaration").to_dict(),
                })
                return "refused"
            self.floor = f
            self.trace.append({"op": "floor", "line": node.line, "value": f})
            return None

        if node.kind == "bind":
            v = self._eval(node.args["expr"])
            if isinstance(v, Verdict):
                self.trace.append({"op": "bind", "line": node.line,
                                   "name": node.args["name"], **v.to_dict()})
                if not v.ok:
                    self.log.append({"level": "verdict", "line": node.line,
                                     "name": node.args["name"], **v.to_dict()})
                    rec = node.args["expr"].args.get("recovery")
                    return "refused" if rec == "refuse" else None
                v = v.value
            self.env[node.args["name"]] = v
            self.trace.append({"op": "bind", "line": node.line,
                               "name": node.args["name"], "verdict": "cut",
                               "value": v.to_dict()})
            return None

        if node.kind == "observe":
            v = self.env.get(node.args["name"])
            self.trace.append({
                "op": "observe", "line": node.line, "name": node.args["name"],
                "value": v.to_dict() if v else None,
                "cut_count": self.M,
            })
            return None

        if node.kind == "assert":
            cond = node.args["cond"]
            val = self._resolve(cond["lhs"])
            ok = _cmp(val, cond["op"], cond["rhs"])
            self.trace.append({"op": "assert", "line": node.line,
                               "condition": cond, "observed": val, "passed": ok})
            if not ok:
                self.log.append({
                    "level": "refusal", "line": node.line,
                    "message": node.args["message"] or "assertion failed",
                    "observed": val,
                })
                return "assertion-failed"
            return None

        v = self._eval(node)
        if isinstance(v, Verdict) and not v.ok:
            self.log.append({"level": "verdict", "line": node.line, **v.to_dict()})
        return None

    # -- expressions ----------------------------------------------------

    def _eval(self, node: Node):
        fn = getattr(self, f"_eval_{node.kind}", None)
        if fn is None:
            raise HonjoError(f"cannot evaluate {node.kind}", node.line)
        return fn(node)

    def _eval_num(self, node: Node) -> Value:
        f = node.args["floor"] or self.floor
        return Value("Scalar", f, max(node.args["value"], f), Prov.STATED,
                     {"value": node.args["value"]})

    def _eval_ref(self, node: Node) -> Value:
        v = self.env.get(node.args["name"])
        if v is None:
            raise HonjoError(f"unbound identifier {node.args['name']!r}", node.line)
        return v

    def _eval_cut(self, node: Node) -> Value:
        """Arity-one cut: individuate an atom from the medium."""
        raw = node.args["z"]
        if node.args["z_kind"] == "number":
            z = int(float(raw))
        else:
            ref = self.env.get(raw)
            if ref is None:
                raise HonjoError(f"unbound {raw!r}", node.line)
            z = int(ref.data.get("value", 0))
        if z < 1:
            raise HonjoError(f"atomic number must be positive, got {z}", node.line)

        capv, occ, nu = valence(z)
        self.M += 1                      # measurement, not simulation
        residue = max(self.floor * max(nu, 1), self.floor)
        return Value(
            "Atom", self.floor, residue, Prov.STATED,
            {
                "z": z,
                "symbol": symbol(z),
                "configuration": " ".join(f"{n}{k}" for n, k in configuration(z)),
                "valence_capacity": capv,
                "valence_occupancy": occ,
                "vacancy": nu,
                "covalent_valence": covalent_valence(z),
                "noble": nu == 0,
                "cut_index": self.M,
            },
        )

    def _eval_bond(self, node: Node):
        a = self._eval(node.args["a"])
        b = self._eval(node.args["b"])
        k = node.args["cells"]

        nu_a, nu_b = a.data.get("vacancy", 0), b.data.get("vacancy", 0)
        if nu_a == 0 or nu_b == 0:
            closed = [x for x in (a, b) if x.data.get("vacancy", 0) == 0]
            return Verdict.inert(
                [{"symbol": x.data.get("symbol"), "vacancy": 0} for x in closed]
            )
        if k > min(nu_a, nu_b):
            return Verdict.unclosed(
                [{"symbol": a.data.get("symbol"), "vacancy": nu_a},
                 {"symbol": b.data.get("symbol"), "vacancy": nu_b}]
            )
        if node.args["guard"] is not None:
            g = node.args["guard"]
            lhs = 1.0 if g["lhs"] == "delta" else self._resolve(g["lhs"])
            if not _cmp(lhs, g["op"], g["rhs"]):
                return Verdict.unclosed([{"guard": g, "value": lhs}])

        self.M += 1
        residue = max(self.floor * k, self.floor)
        return Verdict.cut(Value(
            "Bond", self.floor, residue, Prov.join([a.prov, b.prov]),
            {
                "a": a.data.get("symbol"),
                "b": b.data.get("symbol"),
                "committed_cells": k,
                "shared_content_positive": True,
                "max_cells": min(nu_a, nu_b),
                "cut_index": self.M,
            },
        ))

    def _eval_deloc(self, node: Node):
        members = []
        for name in node.args["members"]:
            v = self.env.get(name)
            if v is None:
                raise HonjoError(f"unbound {name!r}", node.line)
            members.append(v)
        n = len(members)
        if n < 3:
            return Verdict.unclosed(
                [{"reason": "a delocalised system needs at least 3 centres",
                  "given": n}]
            )
        cells = node.args["cells"] or (n + n // 2)
        self.M += 1
        residue = max(self.floor * cells, self.floor)
        return Verdict.cut(Value(
            "Deloc", self.floor, residue,
            Prov.join([m.prov for m in members]),
            {
                "members": [m.data.get("symbol") for m in members],
                "n_centres": n,
                "total_cells": cells,
                # deliberately absent: a per-pair count would state a
                # pairwise fact the system does not determine
                "per_pair_cells": None,
                "cut_index": self.M,
            },
        ))

    def _eval_close(self, node: Node):
        centre = self.env.get(node.args["centre"])
        if centre is None:
            raise HonjoError(f"unbound {node.args['centre']!r}", node.line)

        ligs = []
        for l in node.args["ligands"]:
            v = self.env.get(l["name"])
            if v is None:
                raise HonjoError(f"unbound {l['name']!r}", node.line)
            ligs.append((v, l["cells"]))

        nu_c = centre.data.get("vacancy", 0)
        if nu_c == 0 and all(v.data.get("vacancy", 0) == 0 for v, _ in ligs):
            parts = [{"symbol": centre.data.get("symbol"), "vacancy": 0}]
            parts += [{"symbol": v.data.get("symbol"), "vacancy": 0}
                      for v, _ in ligs if v.data.get("vacancy", 0) == 0]
            return Verdict.inert(parts)
        if nu_c == 0:
            return Verdict.inert(
                [{"symbol": centre.data.get("symbol"), "vacancy": 0}]
            )

        committed = sum(c for _v, c in ligs)
        residual_centre = nu_c - committed
        open_atoms = []
        if residual_centre != 0:
            open_atoms.append({
                "symbol": centre.data.get("symbol"),
                "vacancy": nu_c, "committed": committed,
                "residual": residual_centre,
            })
        for v, c in ligs:
            nu = v.data.get("vacancy", 0)
            if nu - c != 0:
                open_atoms.append({
                    "symbol": v.data.get("symbol"), "vacancy": nu,
                    "committed": c, "residual": nu - c,
                })
        if open_atoms:
            return Verdict.unclosed(open_atoms)

        n_bonded = len(ligs)
        lone = max(0, (centre.data.get("valence_capacity", 8)
                       - centre.data.get("valence_occupancy", 0)
                       - committed) // 2)
        lone = max(0, (centre.data.get("valence_occupancy", 0) - committed) // 2)
        regions = n_bonded + lone

        n_cuts = len(ligs)
        self.M += n_cuts
        residue = max(self.floor * sum(c for _v, c in ligs), self.floor)
        return Verdict.cut(Value(
            "Compound", self.floor, residue,
            Prov.join([centre.prov] + [v.prov for v, _ in ligs]),
            {
                "centre": centre.data.get("symbol"),
                "ligands": [
                    {"symbol": v.data.get("symbol"), "cells": c} for v, c in ligs
                ],
                "stoichiometry": f"{centre.data.get('symbol')}"
                                 f"{''.join(v.data.get('symbol', '') for v, _ in ligs)}",
                "vacancy": 0,
                "closed": True,
                "geometry": geometry(regions, lone),
                "cuts_committed": n_cuts,
                "cut_index": self.M,
            },
        ))

    def _eval_track(self, node: Node):
        item = self.env.get(node.args["item"])
        proc = self.env.get(node.args["process"])
        if item is None or proc is None:
            raise HonjoError("track over unbound identifier", node.line)

        # a chain of cuts; length is the process's committed cut count
        k = max(1, proc.data.get("cuts_committed", 1))
        residues = [max(self.floor, self.floor * (i + 1)) for i in range(k)]
        self.M += k                        # a failed track still costs its cuts

        total = sum(residues)
        omega = max(total, self.floor)
        alignment = self.floor / omega
        admit = node.args["admit"]
        converged = alignment <= (self.floor / omega) + 1e-12

        if admit == "converge" and not converged:
            return Verdict.nonconvergent(residues, alignment)
        if admit == "diverge" and converged:
            return Verdict.nonconvergent(residues, alignment)

        return Verdict.cut(Value(
            "Path", self.floor, total,
            Prov.join([item.prov, proc.prov]),
            {
                "chain_length": k,
                "residues": [round(r, 9) for r in residues],
                "terminal_alignment": round(alignment, 9),
                "admitted": admit,
                "representations": node.args["reps"],
                "yield": node.args["yield"],
                "cut_index": self.M,
            },
        ))

    def _eval_import(self, node: Node):
        from ..masamune.translate import translate

        path = node.args["path"]
        full = path if os.path.isabs(path) else os.path.join(self.base_dir, path)
        if not os.path.exists(full):
            return Verdict.malformed(path, "file not found")

        with open(full, "r", encoding="utf-8") as fh:
            text = fh.readline().split()[0].strip()

        fmt = "smiles" if full.endswith((".smi", ".smiles")) else "xyz"
        v = translate(fmt, text, floor=self.floor, source_name=path)
        if not v.ok:
            return v

        g = v.value
        errs = g.validate()
        if errs:
            return Verdict.subfloor(0.0, self.floor, "; ".join(errs))

        supplied = g.supplied_fraction()
        req = node.args["require"]
        if req is not None:
            kind, spec = req
            if kind == "stated" and supplied > 0.0:
                return Verdict.underprovenanced("stated", supplied)
            if kind == "supplied":
                op, thr = spec
                if not _cmp(supplied, op, thr):
                    return Verdict.underprovenanced(f"supplied {op} {thr}", supplied)

        self.M += 1
        first = next(iter(g.atoms))
        sigma, side, cut_prov = g.separation_cost(first)
        return Verdict.cut(Value(
            "Graph", self.floor, max(sigma, self.floor), g.provenance(),
            {
                "path": path,
                "n_atoms": len(g.atoms),
                "n_contacts": len(g.contacts),
                "supplied_fraction": round(supplied, 6),
                "conventions_used": g.conventions_used(),
                "example_separation": {
                    "atom": first,
                    "sigma": round(sigma, 9),
                    "burial_depth": len([s for s in side if s != MEDIUM]),
                    "cut_provenance": str(cut_prov),
                },
                "cut_index": self.M,
            },
        ))

    def _resolve(self, ref: str):
        if "." in ref:
            name, attr = ref.split(".", 1)
        else:
            name, attr = ref, "residue"
        v = self.env.get(name)
        if v is None:
            return 0
        if attr == "residue":
            return v.residue
        if attr == "floor":
            return v.floor
        if attr == "provenance":
            return str(v.prov)
        return v.data.get(attr, 0)


def run_honjo(src: str, base_dir: str = ".", eps_target: float = 1e-9) -> dict:
    return Interpreter(base_dir, eps_target).run(src)
