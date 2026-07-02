#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static audit of the algorithms/ plugin catalog.

Step 1 of MLX_FULL_NATIVE_AND_AUDIT_PLAN.md. Scans every ``algorithms/<name>/``
folder via AST (no imports, no side effects) and reports, per folder:

- base class family (core.algorithm.Algorithm / pymoo subclass / other),
- whether an ``__init__.py`` export is present (class or ``create_algorithm``),
- whether ``ALGORITHM_FLAGS`` is declared (and its values),
- operator usage: imports native pymoo operators vs defines its own operators,
- static parse health (syntax-level).

Outputs ``specs/algorithms_audit.md`` (summary + nonconformance lists) and
``specs/algorithms_audit.csv`` (full per-folder detail). Read-only on source.
"""
from __future__ import annotations

import ast
import csv
import os
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO_ROOT = os.path.join(HERE, "algorithms")
SPECS_DIR = os.path.join(HERE, "specs")

# pymoo algorithm base classes commonly subclassed by local plugins.
PYMOO_ALGO_BASES = {
    "NSGA2", "NSGA3", "RNSGA2", "RNSGA3", "UNSGA3", "MOEAD", "CTAEA", "AGEMOEA",
    "AGEMOEA2", "SMSEMOA", "RVEA", "SPEA2", "GA", "DE", "PSO", "CMAES", "ES",
    "BRKGA", "Algorithm", "GeneticAlgorithm",
}
PYMOO_OPERATOR_BASES = {
    "Crossover", "Mutation", "Selection", "Sampling", "Survival", "Repair",
}


def _safe_read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _analyze_folder(name: str) -> dict:
    folder = os.path.join(ALGO_ROOT, name)
    rec = {
        "name": name,
        "has_init": False,
        "init_exports": False,
        "has_factory": False,
        "base_family": "unknown",   # resolved later (chain-aware)
        "direct_base": "",          # immediate base family
        "base_detail": "",
        "local_base_folder": "",    # folder of the local base, if any
        "has_flags": False,
        "flags": "",
        "operators": "none",
        "parse_ok": True,
        "parse_errors": "",
        "notes": "",
    }
    py_files = [e for e in sorted(os.listdir(folder)) if e.endswith(".py")]
    rec["has_init"] = "__init__.py" in py_files

    import_native_ops = False
    defines_custom_ops = False
    bases_found: set[str] = set()
    imported_from: dict[str, str] = {}  # imported name -> source module
    flags_values: list[str] = []
    notes: list[str] = []

    for fname in py_files:
        src = _safe_read(os.path.join(folder, fname))
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            rec["parse_ok"] = False
            notes.append(f"{fname}: SyntaxError {exc.lineno}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for a in node.names:
                    imported_from[a.asname or a.name] = mod
                if mod.startswith("pymoo.operators"):
                    import_native_ops = True
            elif isinstance(node, ast.ClassDef):
                for b in node.bases:
                    bn = _base_name(b)
                    if bn:
                        bases_found.add(bn)
                    if bn in PYMOO_OPERATOR_BASES:
                        defines_custom_ops = True
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "ALGORITHM_FLAGS":
                        rec["has_flags"] = True
                        if isinstance(node.value, ast.Dict):
                            for v in node.value.values:
                                for sub in ast.walk(v):
                                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                                        flags_values.append(sub.value)
            elif isinstance(node, ast.FunctionDef) and node.name == "create_algorithm":
                rec["has_factory"] = True

    # __init__.py export check
    init_src = _safe_read(os.path.join(folder, "__init__.py"))
    if init_src.strip():
        try:
            itree = ast.parse(init_src)
            for node in ast.walk(itree):
                if isinstance(node, ast.ImportFrom) and node.level >= 1:
                    rec["init_exports"] = True
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    rec["init_exports"] = True
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "__all__":
                            rec["init_exports"] = True
        except SyntaxError as exc:
            rec["parse_ok"] = False
            notes.append(f"__init__.py: SyntaxError {exc.lineno}")

    # Direct base family (chain resolved in a second pass).
    direct = "none"
    local_folder = ""
    for b in bases_found:
        src = imported_from.get(b, "")
        if b == "Algorithm" and "core.algorithm" in src:
            direct = "core.algorithm"
            break
        if "pymoo" in src:
            direct = "pymoo_subclass"
            break
        if src.startswith("algorithms.") or src.startswith("algorithms"):
            direct = "local_subclass"
            parts = src.split(".")
            if len(parts) >= 2:
                local_folder = parts[1]
            break
    if direct == "none" and bases_found:
        direct = "other"
    rec["direct_base"] = direct
    rec["local_base_folder"] = local_folder
    rec["base_detail"] = ",".join(sorted(b for b in bases_found if b))

    if import_native_ops and defines_custom_ops:
        rec["operators"] = "native+custom"
    elif import_native_ops:
        rec["operators"] = "native"
    elif defines_custom_ops:
        rec["operators"] = "custom"
    else:
        rec["operators"] = "none"

    rec["flags"] = ",".join(sorted(set(flags_values)))
    rec["parse_errors"] = "; ".join(notes)
    rec["notes"] = "; ".join(notes)
    return rec


def _resolve_root_family(name: str, by_name: dict[str, dict],
                         _seen: set[str] | None = None) -> str:
    """Follow local_subclass chains to the ultimate base family."""
    _seen = _seen or set()
    rec = by_name.get(name)
    if rec is None or name in _seen:
        return "other"
    _seen.add(name)
    direct = rec["direct_base"]
    if direct == "local_subclass":
        parent = rec.get("local_base_folder", "")
        if parent and parent in by_name:
            return _resolve_root_family(parent, by_name, _seen)
        return "local_subclass"
    return direct


def main() -> int:
    os.makedirs(SPECS_DIR, exist_ok=True)
    folders = sorted(
        n for n in os.listdir(ALGO_ROOT)
        if n != "__pycache__" and not n.startswith(".")
        and os.path.isdir(os.path.join(ALGO_ROOT, n))
    )
    records = [_analyze_folder(n) for n in folders]
    by_name = {r["name"]: r for r in records}

    # Resolve chain-aware base family and plugin classification.
    for r in records:
        r["base_family"] = _resolve_root_family(r["name"], by_name)
        is_plugin = (
            r["base_family"] in {"core.algorithm", "pymoo_subclass", "local_subclass"}
            or r["has_flags"]
        )
        r["kind"] = "plugin" if is_plugin else "non-plugin"

    # CSV (full detail)
    csv_path = os.path.join(SPECS_DIR, "algorithms_audit.csv")
    cols = ["name", "kind", "base_family", "direct_base", "local_base_folder",
            "base_detail", "has_init", "init_exports", "has_factory",
            "has_flags", "flags", "operators", "parse_ok", "parse_errors"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Aggregates
    total = len(records)
    plugins = [r for r in records if r["kind"] == "plugin"]
    non_plugins = [r["name"] for r in records if r["kind"] == "non-plugin"]
    no_init = [r["name"] for r in plugins if not r["has_init"]]
    no_export = [r["name"] for r in plugins if not r["init_exports"]]
    no_flags = [r["name"] for r in plugins if not r["has_flags"]]
    parse_bad = [r["name"] for r in records if not r["parse_ok"]]
    by_base: dict[str, int] = {}
    by_ops: dict[str, int] = {}
    for r in plugins:
        by_base[r["base_family"]] = by_base.get(r["base_family"], 0) + 1
        by_ops[r["operators"]] = by_ops.get(r["operators"], 0) + 1

    def _lst(items: list[str]) -> str:
        return "none" if not items else ", ".join(f"`{x}`" for x in items)

    md = []
    md.append("# Algorithms Catalog Audit\n")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M} (static AST scan, "
              "read-only).\n")
    md.append(f"Total folders under `algorithms/`: **{total}** "
              f"(plugins: **{len(plugins)}**, non-plugin/infra: "
              f"**{len(non_plugins)}**)\n")
    md.append("## Summary (plugins only)\n")
    md.append("Base family resolves local-subclass chains to their ultimate root.\n")
    md.append("| Base family (root) | Count |\n|---|---|")
    for k in sorted(by_base):
        md.append(f"| {k} | {by_base[k]} |")
    md.append("\n| Operator usage | Count |\n|---|---|")
    for k in sorted(by_ops):
        md.append(f"| {k} | {by_ops[k]} |")
    md.append("\n## Nonconformance (plugins to fix in Steps 6-7)\n")
    md.append(f"- Missing `__init__.py`: {_lst(no_init)}")
    md.append(f"- `__init__.py` without a visible export: {_lst(no_export)}")
    md.append(f"- Missing `ALGORITHM_FLAGS`: {_lst(no_flags)}")
    md.append(f"- Static parse errors: {_lst(parse_bad)}")
    md.append("\n## Non-plugin / infrastructure folders (excluded from the above)\n")
    md.append("These folders are not standalone algorithm plugins (shared "
              "helpers or pymoo re-export shims) and are not expected to declare "
              "`ALGORITHM_FLAGS`:")
    md.append(_lst(non_plugins))
    md.append("\n## Operator-usage legend\n")
    md.append("- `native`: imports `pymoo.operators.*` (reuses native operators).")
    md.append("- `custom`: defines its own operator class (Crossover/Mutation/…).")
    md.append("- `native+custom`: both. `none`: relies on the base algorithm's "
              "default operators inherited from its pymoo/core root (typical and "
              "correct for NSGA2/MOEAD subclasses).")
    md.append("\n## Base-family legend\n")
    md.append("- `core.algorithm`: inherits `core.algorithm.Algorithm` "
              "(backend/MLX-aware base).")
    md.append("- `pymoo_subclass`: subclasses a native pymoo algorithm "
              "(NSGA2/NSGA3/MOEAD/…).")
    md.append("- `local_subclass`: subclasses another local algorithm whose root "
              "could not be resolved to pymoo/core (review for Step 7).")
    md.append("\nFull per-folder detail: `specs/algorithms_audit.csv`.\n")

    md_path = os.path.join(SPECS_DIR, "algorithms_audit.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"OK -> {md_path}")
    print(f"OK -> {csv_path}")
    print(f"folders={total} no_flags={len(no_flags)} no_export={len(no_export)} "
          f"parse_bad={len(parse_bad)}")
    print("base_family:", by_base)
    print("operators:", by_ops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
