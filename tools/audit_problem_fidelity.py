#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fidelity audit: ported problem defaults vs PlatEMO MATLAB definitions.

For every PlatEMO multi-objective problem, parse its ``Setting()`` method for the
default number of objectives (M), the default number of variables (D, possibly a
function of M), and the lower/upper bound pattern. Then instantiate the matching
project problem (via its discovered spec factory) at the PlatEMO default M and
compare n_obj, n_var, and bounds. Mismatches are likely silent fidelity bugs.

Cannot run MATLAB, so this compares *definitions* (structural fidelity), not
sampled objective values. Outputs ``specs/problem_fidelity_audit.md`` + ``.csv``.

Usage: python tools/audit_problem_fidelity.py [family ...]
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

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


def _D_expr(block: str) -> str | None:
    exprs = re.findall(r"obj\.D\s*=\s*([^;]+);", block)
    return " ; ".join(e.strip() for e in exprs) if exprs else None


def _eval_D(expr: str | None, M: int) -> int | None:
    """Evaluate the sequence of obj.D assignments (PlatEMO often reassigns D,
    e.g. D=10 then D=ceil((D-1)/2)*2+1)."""
    import math
    if expr is None:
        return None
    K = M - 1  # WFG position params: K = M-1 unless ParameterSet overrides
    cur: int | None = None
    for raw in expr.split(" ; "):
        e = raw.replace("obj.M", str(M)).replace("obj.K", str(K))
        if cur is not None:
            e = e.replace("obj.D", str(cur))
        e = e.replace("ceil", "math.ceil").replace("floor", "math.floor")
        if "obj.D" in e or not re.fullmatch(r"[0-9+\-*/() .a-z_]+", e):
            return cur  # unevaluable step; keep last known
        try:
            cur = int(eval(e, {"__builtins__": {}, "math": math}))  # noqa: S307
        except Exception:  # noqa: BLE001
            return cur
    return cur


def _bounds_kind(block: str) -> str:
    low = re.search(r"obj\.lower\s*=\s*([^;]+);", block)
    up = re.search(r"obj\.upper\s*=\s*([^;]+);", block)
    lk = "zeros" if low and "zeros" in low.group(1) else (low.group(1).strip() if low else "?")
    uk = "ones" if up and "ones" in up.group(1) else (up.group(1).strip() if up else "?")
    return f"{lk} | {uk}"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main(argv: list[str]) -> int:
    families = argv or None
    SPECS.mkdir(exist_ok=True)

    # Discover project problems and index by normalized name.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import PymooLab as P
    specs = P.discover_problem_specs(HERE, [])
    by_norm: dict[str, object] = {}
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
            if "classdef" not in text.split("\n", 1)[0] and "classdef" not in text[:200]:
                continue
            block = _setting_block(text)
            M = _default_M(block)
            Mtest = M if M is not None else 3  # PlatEMO default when isempty
            Dexp = _D_expr(block)
            Dexpect = _eval_D(Dexp, Mtest)
            bounds = _bounds_kind(block)

            spec = by_norm.get(_norm(name))
            rec = {
                "family": fam, "problem": name, "matched": bool(spec),
                "plat_M": M if M is not None else f"isempty->{Mtest}",
                "plat_D_expr": Dexp or "", "plat_D@M": Dexpect,
                "plat_bounds": bounds,
                "proj_n_obj": "", "proj_n_var": "", "proj_bounds": "",
                "status": "no-match" if not spec else "?",
            }
            if spec is not None:
                try:
                    cfg = {"n_obj": Mtest} if M is None else {}
                    prob = spec.factory(cfg)
                    n_obj = int(getattr(prob, "n_obj", -1))
                    n_var = int(getattr(prob, "n_var", -1))
                    xl = np.asarray(getattr(prob, "xl", []), dtype=float)
                    xu = np.asarray(getattr(prob, "xu", []), dtype=float)
                    rec["proj_n_obj"] = n_obj
                    rec["proj_n_var"] = n_var
                    pb = (
                        ("zeros" if xl.size and np.allclose(xl, 0) else "nonzero")
                        + " | "
                        + ("ones" if xu.size and np.allclose(xu, 1) else "nonunit")
                    )
                    rec["proj_bounds"] = pb
                    issues = []
                    if Dexpect is not None and n_var != Dexpect:
                        issues.append(f"D {n_var}!={Dexpect}")
                    if M is not None and n_obj != M:
                        issues.append(f"M {n_obj}!={M}")
                    rec["status"] = "OK" if not issues else "MISMATCH: " + "; ".join(issues)
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = f"instantiate-error: {type(exc).__name__}"
            rows.append(rec)

    csv_path = SPECS / "problem_fidelity_audit.csv"
    cols = ["family", "problem", "matched", "plat_M", "plat_D_expr", "plat_D@M",
            "plat_bounds", "proj_n_obj", "proj_n_var", "proj_bounds", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    mism = [r for r in rows if r["status"].startswith("MISMATCH")]
    errs = [r for r in rows if "error" in r["status"]]
    okc = sum(1 for r in rows if r["status"] == "OK")
    print(f"problems_checked={total} OK={okc} MISMATCH={len(mism)} errors={len(errs)}")
    for r in mism[:40]:
        print(f"  MISMATCH {r['family']}/{r['problem']}: {r['status']} "
              f"(plat M={r['plat_M']} D@M={r['plat_D@M']}; proj M={r['proj_n_obj']} D={r['proj_n_var']})")
    print(f"report -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
