#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parametric-fidelity audit: do ported problems honor custom M and n?

PlatEMO problems have a default relationship between objectives (M) and decision
variables (D), e.g. DTLZ1 uses D = M + 9 (M=3 -> D=12, M=5 -> D=14). When a user
sets M but not D, D must follow that relationship; when a user sets D explicitly,
it must be honored; and fixed-M problems (e.g. ZDT, M=2) must stay fixed.

This audit, per PlatEMO problem:
  A) M-tracking: instantiate the project problem at two M values (no n_var) and
     check n_var follows PlatEMO's D(M) at each.
  B) custom-n: instantiate with two different explicit n_var values and check the
     resulting n_var responds (problem does not ignore the override).
It only varies M for problems PlatEMO marks customizable (``isempty(obj.M)``).

Outputs specs/problem_parametric_audit.md + .csv. Read-only on source.

Usage: python tools/audit_problem_parametric.py [family ...]
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = Path(
    "/Users/proftheagos/devSuport/METISBr/platemoMETISBr/PlatEMO/"
    "Problems/Multi-objective optimization"
)
SPECS = HERE / "specs"


def _setting_block(text: str) -> str:
    m = re.search(r"function\s+Setting\s*\(\s*obj\s*\)(.*?)\n\s*end", text, re.S)
    return m.group(1) if m else ""


def _default_M(block: str) -> int | None:
    m = re.search(r"obj\.M\s*=\s*(\d+)\s*;", block)
    return int(m.group(1)) if m else None


def _M_customizable(block: str) -> bool:
    return "isempty(obj.M)" in block


def _D_exprs(block: str) -> list[str]:
    return [e.strip() for e in re.findall(r"obj\.D\s*=\s*([^;]+);", block)]


def _eval_D(exprs: list[str], M: int) -> int | None:
    K = M - 1
    cur: int | None = None
    for raw in exprs:
        e = raw.replace("obj.M", str(M)).replace("obj.K", str(K))
        if cur is not None:
            e = e.replace("obj.D", str(cur))
        e = e.replace("ceil", "math.ceil").replace("floor", "math.floor")
        if "obj.D" in e or not re.fullmatch(r"[0-9+\-*/() .a-z_]+", e):
            return cur
        try:
            cur = int(eval(e, {"__builtins__": {}, "math": math}))  # noqa: S307
        except Exception:  # noqa: BLE001
            return cur
    return cur


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _nvar(spec, **cfg):
    prob = spec.factory(cfg)
    return int(getattr(prob, "n_var", -1)), int(getattr(prob, "n_obj", -1))


def main(argv: list[str]) -> int:
    families = argv or None
    SPECS.mkdir(exist_ok=True)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import PymooLab as P

    specs = P.discover_problem_specs(HERE, [])
    by_norm = {}
    for spec in specs.values():
        by_norm.setdefault(_norm(spec.name), spec)

    rows = []
    for fam in sorted(os.listdir(SRC)):
        if families and fam not in families:
            continue
        fdir = SRC / fam
        if not fdir.is_dir():
            continue
        for mf in sorted(fdir.glob("*.m")):
            name = mf.stem
            if not re.match(r"^[A-Z]", name):
                continue
            text = mf.read_text(encoding="utf-8", errors="ignore")
            if "classdef" not in text[:200]:
                continue
            spec = by_norm.get(_norm(name))
            if spec is None:
                continue
            block = _setting_block(text)
            exprs = _D_exprs(block)
            dM = _default_M(block)
            customizable = _M_customizable(block)
            M1 = dM if dM is not None else 3

            rec = {"family": fam, "problem": name, "M_customizable": customizable,
                   "m_tracking": "", "custom_n": "", "status": "OK"}
            issues = []

            # A) M-tracking
            try:
                if customizable:
                    M2 = M1 + 2
                    nv1, _ = _nvar(spec, n_obj=M1)
                    nv2, _ = _nvar(spec, n_obj=M2)
                    e1, e2 = _eval_D(exprs, M1), _eval_D(exprs, M2)
                    ok = True
                    if e1 is not None and nv1 != e1:
                        ok = False; issues.append(f"n(M={M1})={nv1}!={e1}")
                    if e2 is not None and nv2 != e2:
                        ok = False; issues.append(f"n(M={M2})={nv2}!={e2}")
                    rec["m_tracking"] = "ok" if ok else "FAIL"
                else:
                    nv1, no1 = _nvar(spec)
                    rec["m_tracking"] = "fixed-M" if no1 == M1 else f"FAIL(M={no1}!={M1})"
                    if no1 != M1:
                        issues.append(rec["m_tracking"])
            except Exception as exc:  # noqa: BLE001
                rec["m_tracking"] = f"err:{type(exc).__name__}"

            # B) custom n_var honored (two distinct values -> distinct n_var)
            try:
                base = _eval_D(exprs, M1) or (M1 + 9)
                vA, vB = base + 2, base + 6
                kw = {"n_obj": M1} if customizable else {}
                nA, _ = _nvar(spec, n_var=vA, **kw)
                nB, _ = _nvar(spec, n_var=vB, **kw)
                if nA == nB:
                    rec["custom_n"] = "IGNORED"
                    issues.append("custom n_var ignored")
                else:
                    rec["custom_n"] = "honored"
            except Exception as exc:  # noqa: BLE001
                rec["custom_n"] = f"err:{type(exc).__name__}"

            if issues:
                rec["status"] = "ISSUE: " + "; ".join(issues)
            rows.append(rec)

    csv_path = SPECS / "problem_parametric_audit.csv"
    cols = ["family", "problem", "M_customizable", "m_tracking", "custom_n", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    issues = [r for r in rows if r["status"].startswith("ISSUE")]
    track_fail = [r for r in rows if "FAIL" in r["m_tracking"]]
    ignored = [r for r in rows if r["custom_n"] == "IGNORED"]
    errs = [r for r in rows if "err:" in r["m_tracking"] or r["custom_n"].startswith("err")]
    print(f"checked={total} issues={len(issues)} m_tracking_FAIL={len(track_fail)} "
          f"custom_n_IGNORED={len(ignored)} instantiation_errors={len(errs)}")
    for r in issues[:50]:
        print(f"  {r['family']}/{r['problem']}: {r['status']}")
    print(f"report -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
