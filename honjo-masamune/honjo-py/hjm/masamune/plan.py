"""The Masamune plan language.

    plan name {
      source lib : smiles at "file.smi"
      budget 5000 records

      let raw   := read lib
      let mols  := translate raw require element, connectivity, cellcount
                     expect supplied < 0.25
                     else report
      let core  := select mols where supplied == 0.0
      assert core.count > 2  emit "too few fully-stated structures"
      emit core with provenance
    }

Declaration order is execution order.  No dependency inference.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from ..core.verdict import Label, Verdict
from .capability import capability as _cap_of
from .capability import known_format as _known, missing as _missing
from .translate import translate


class _Cap:
    missing = staticmethod(_missing)
    capability = staticmethod(_cap_of)
    known_format = staticmethod(_known)


cap = _Cap()

KEYWORDS = {
    "plan", "source", "budget", "records", "let", "read", "translate",
    "require", "expect", "supplied", "else", "refuse", "report", "select",
    "where", "emit", "with", "provenance", "assert", "at", "count",
}

TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>--[^\n]*)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<op><=|>=|==|!=|:=|[<>{},:.])
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


class PlanError(Exception):
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
            raise PlanError(f"unexpected character {src[pos]!r}", line)
        kind = m.lastgroup
        text = m.group()
        line += text.count("\n")
        pos = m.end()
        if kind in ("ws", "comment"):
            continue
        if kind == "ident" and text in KEYWORDS:
            kind = "kw"
        out.append(Tok(kind, text, line))
    out.append(Tok("eof", "", line))
    return out


@dataclass
class Step:
    op: str
    target: str | None = None
    args: dict = field(default_factory=dict)
    line: int = 0


@dataclass
class Plan:
    name: str
    sources: dict = field(default_factory=dict)
    budget: int | None = None
    steps: list[Step] = field(default_factory=list)


class Parser:
    def __init__(self, toks: list[Tok]):
        self.t, self.i = toks, 0

    def peek(self) -> Tok:
        return self.t[self.i]

    def next(self) -> Tok:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str, text: str | None = None) -> Tok:
        tok = self.next()
        if tok.kind != kind or (text is not None and tok.text != text):
            want = text or kind
            raise PlanError(f"expected {want!r}, got {tok.text!r}", tok.line)
        return tok

    def accept(self, text: str) -> bool:
        if self.peek().text == text:
            self.next()
            return True
        return False

    def parse(self) -> Plan:
        self.expect("kw", "plan")
        name = self.expect("ident").text
        self.expect("op", "{")
        plan = Plan(name=name)
        while self.peek().text != "}":
            self._decl_or_step(plan)
        self.expect("op", "}")
        return plan

    def _decl_or_step(self, plan: Plan) -> None:
        tok = self.peek()
        if tok.text == "source":
            self.next()
            ident = self.expect("ident").text
            self.expect("op", ":")
            fmt = self.expect("ident").text
            path = None
            if self.accept("at"):
                path = json.loads(self.expect("string").text)
            plan.sources[ident] = {"format": fmt, "path": path}
            return
        if tok.text == "budget":
            self.next()
            n = int(float(self.expect("number").text))
            self.accept("records")
            plan.budget = n
            return
        if tok.text == "let":
            self.next()
            target = self.expect("ident").text
            self.expect("op", ":=")
            plan.steps.append(self._rhs(target, tok.line))
            return
        if tok.text == "assert":
            self.next()
            cond = self._cond()
            msg = None
            if self.accept("emit"):
                msg = json.loads(self.expect("string").text)
            plan.steps.append(
                Step("assert", None, {"cond": cond, "message": msg}, tok.line)
            )
            return
        if tok.text == "emit":
            self.next()
            ident = self.expect("ident").text
            with_prov = False
            if self.accept("with"):
                self.expect("kw", "provenance")
                with_prov = True
            plan.steps.append(
                Step("emit", None, {"name": ident, "provenance": with_prov}, tok.line)
            )
            return
        raise PlanError(f"unexpected {tok.text!r}", tok.line)

    def _rhs(self, target: str, line: int) -> Step:
        tok = self.next()
        if tok.text == "read":
            src = self.expect("ident").text
            return Step("read", target, {"source": src}, line)
        if tok.text == "translate":
            src = self.expect("ident").text
            args: dict = {"input": src, "require": [], "expect": None, "else": None}
            if self.accept("require"):
                args["require"].append(self.expect("ident").text)
                while self.accept(","):
                    args["require"].append(self.expect("ident").text)
            if self.accept("expect"):
                self.expect("kw", "supplied")
                op = self.expect("op").text
                val = float(self.expect("number").text)
                args["expect"] = (op, val)
            if self.accept("else"):
                args["else"] = self.next().text
            return Step("translate", target, args, line)
        if tok.text == "select":
            src = self.expect("ident").text
            self.expect("kw", "where")
            return Step("select", target, {"input": src, "cond": self._cond()}, line)
        raise PlanError(f"unknown operation {tok.text!r}", line)

    def _cond(self) -> dict:
        lhs = self.next().text
        if self.peek().text == ".":
            self.next()
            lhs = f"{lhs}.{self.next().text}"
        op = self.expect("op").text
        rhs_tok = self.next()
        rhs = float(rhs_tok.text) if rhs_tok.kind == "number" else rhs_tok.text
        return {"lhs": lhs, "op": op, "rhs": rhs}


def _cmp(a, op: str, b) -> bool:
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        a, b = str(a), str(b)
    return {
        "<": a < b, ">": a > b, "<=": a <= b,
        ">=": a >= b, "==": a == b, "!=": a != b,
    }[op]


class PlanRunner:
    """Executes a plan.  Steps run in source order; no reordering."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def run(self, src: str) -> dict:
        try:
            plan = Parser(lex(src)).parse()
        except PlanError as e:
            return {
                "plan": None,
                "status": "parse-error",
                "error": str(e),
                "steps": [],
            }

        env: dict = {}
        log: list[dict] = []
        out_steps: list[dict] = []
        status = "ok"
        requests = 0

        # static capability check, before any record is read
        for st in plan.steps:
            if st.op != "translate":
                continue
            src_name = self._source_of(plan, st, env)
            if src_name is None:
                continue
            fmt = plan.sources.get(src_name, {}).get("format", "?")
            req = set(st.args["require"])
            miss = cap.missing(fmt, req) if cap.known_format(fmt) else req
            if miss:
                return {
                    "plan": plan.name,
                    "status": "refused",
                    "refusal": {
                        "reason": "capability",
                        "step_line": st.line,
                        "source": src_name,
                        "format": fmt,
                        "missing_features": sorted(miss),
                        "source_capability": sorted(cap.capability(fmt)),
                    },
                    "records_read": 0,
                    "steps": [],
                }

        for st in plan.steps:
            if status != "ok":
                break
            if st.op == "read":
                recs, err = self._read(plan, st)
                if err:
                    status = "error"
                    out_steps.append({"step": "read", "line": st.line, "error": err})
                    break
                if plan.budget is not None and len(recs) > plan.budget:
                    log.append(
                        {
                            "level": "report",
                            "step_line": st.line,
                            "message": f"budget {plan.budget} < {len(recs)} available; truncated",
                        }
                    )
                    recs = recs[: plan.budget]
                requests += len(recs)
                env[st.target] = {"kind": "records", "items": recs,
                                  "source": st.args["source"]}
                out_steps.append(
                    {"step": "read", "line": st.line, "target": st.target,
                     "count": len(recs)}
                )

            elif st.op == "translate":
                rs = env.get(st.args["input"])
                if rs is None or rs["kind"] != "records":
                    status = "error"
                    out_steps.append(
                        {"step": "translate", "line": st.line,
                         "error": f"{st.args['input']} is not a record set"}
                    )
                    break
                fmt = plan.sources[rs["source"]]["format"]
                req = set(st.args["require"]) or {"element", "connectivity"}
                results = []
                for rec in rs["items"]:
                    v = translate(fmt, rec["text"], required=req,
                                  source_name=rec["name"])
                    v = self._apply_expect(v, st.args.get("expect"))
                    results.append({"record": rec["name"], "verdict": v})
                env[st.target] = {"kind": "verdicts", "items": results}
                tally = _tally(results)
                out_steps.append(
                    {"step": "translate", "line": st.line, "target": st.target,
                     "require": sorted(req), "tally": tally}
                )
                recov = st.args.get("else")
                n_bad = sum(v for k, v in tally.items() if k != "translated")
                if n_bad and recov == "refuse":
                    status = "refused"
                    log.append(
                        {"level": "refusal", "step_line": st.line,
                         "message": f"{n_bad} record(s) did not translate; else refuse"}
                    )
                elif n_bad and recov == "report":
                    log.append(
                        {"level": "report", "step_line": st.line,
                         "message": f"{n_bad} record(s) did not translate"}
                    )

            elif st.op == "select":
                vs = env.get(st.args["input"])
                if vs is None or vs["kind"] != "verdicts":
                    status = "error"
                    out_steps.append(
                        {"step": "select", "line": st.line,
                         "error": f"{st.args['input']} is not a verdict set"}
                    )
                    break
                kept = [r for r in vs["items"] if self._match(r, st.args["cond"])]
                env[st.target] = {"kind": "verdicts", "items": kept}
                out_steps.append(
                    {"step": "select", "line": st.line, "target": st.target,
                     "kept": len(kept), "dropped": len(vs["items"]) - len(kept),
                     "condition": st.args["cond"]}
                )

            elif st.op == "assert":
                val = self._resolve(env, st.args["cond"]["lhs"])
                ok = _cmp(val, st.args["cond"]["op"], st.args["cond"]["rhs"])
                out_steps.append(
                    {"step": "assert", "line": st.line, "condition": st.args["cond"],
                     "observed": val, "passed": ok}
                )
                if not ok:
                    status = "assertion-failed"
                    log.append(
                        {"level": "refusal", "step_line": st.line,
                         "message": st.args["message"] or "assertion failed",
                         "observed": val}
                    )

            elif st.op == "emit":
                vs = env.get(st.args["name"])
                out_steps.append(
                    {"step": "emit", "line": st.line, "name": st.args["name"],
                     "emitted": self._emit(vs, st.args["provenance"])}
                )

        return {
            "plan": plan.name,
            "status": status,
            "budget": plan.budget,
            "records_read": requests,
            "sources": plan.sources,
            "steps": out_steps,
            "log": log,
        }

    # -- helpers --------------------------------------------------------

    def _source_of(self, plan: Plan, st: Step, env: dict) -> str | None:
        for prior in plan.steps:
            if prior.op == "read" and prior.target == st.args.get("input"):
                return prior.args["source"]
        return None

    def _read(self, plan: Plan, st: Step):
        name = st.args["source"]
        decl = plan.sources.get(name)
        if decl is None:
            return [], f"undeclared source {name!r}"
        path = decl.get("path")
        if not path:
            return [], f"source {name!r} has no path"
        full = path if os.path.isabs(path) else os.path.join(self.base_dir, path)
        if not os.path.exists(full):
            return [], f"file not found: {full}"
        recs = []
        with open(full, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                text = parts[0]
                label = parts[1].strip() if len(parts) > 1 else f"{path}:{n}"
                recs.append({"name": label, "text": text})
        return recs, None

    def _apply_expect(self, v: Verdict, expect) -> Verdict:
        if expect is None or v.label is not Label.TRANSLATED:
            return v
        op, thr = expect
        measured = v.payload.get("supplied_fraction", 0.0)
        if not _cmp(measured, op, thr):
            return Verdict.incomplete(["supplied-threshold"], measured)
        return v

    def _match(self, rec: dict, cond: dict) -> bool:
        v: Verdict = rec["verdict"]
        lhs = cond["lhs"]
        if lhs in ("supplied", "supplied_fraction"):
            val = v.payload.get("supplied_fraction", 1.0)
        elif lhs == "verdict":
            return _cmp(str(v.label), cond["op"], cond["rhs"])
        else:
            val = v.payload.get(lhs, 0.0)
        return _cmp(val, cond["op"], cond["rhs"])

    def _resolve(self, env: dict, ref: str):
        if "." in ref:
            name, attr = ref.split(".", 1)
        else:
            name, attr = ref, "count"
        v = env.get(name)
        if v is None:
            return 0
        if attr == "count":
            return len(v.get("items", []))
        return 0

    def _emit(self, vs, with_prov: bool):
        if vs is None:
            return []
        out = []
        for r in vs.get("items", []):
            v: Verdict = r["verdict"]
            item = {"record": r["record"], **v.to_dict()}
            if not with_prov and item.get("value"):
                item["value"].pop("conventions_used", None)
            out.append(item)
        return out


def _tally(results: list[dict]) -> dict:
    t: dict = {}
    for r in results:
        k = str(r["verdict"].label)
        t[k] = t.get(k, 0) + 1
    return dict(sorted(t.items()))


def run_plan(src: str, base_dir: str = ".") -> dict:
    return PlanRunner(base_dir).run(src)
