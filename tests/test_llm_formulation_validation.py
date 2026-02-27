from __future__ import annotations

from pymoolab_core.llm.formulation import LLMFormulationService as S


def test_llm_metric_hv_montecarlo_canonical_validates():
    bundle = S.generate_artifact_bundle(
        "criar uma versao do indicator hypervolume usando metodo monte-carlo para calculo aproximado focado em maops com m>3",
        artifact_type="metric",
        base_name="HV_LLM_MONTECARLO",
        n_var=30,
        n_obj=5,
        provider="local_template",
    )
    rep = bundle.get("_validation_report", {})
    assert rep.get("ok") is True
    runtime = rep.get("runtime", {})
    assert runtime.get("metric_kind_hint") == "hv_mc"
    assert len(runtime.get("validation_cases", [])) >= 2


def test_llm_metric_known_bad_hv_is_rejected():
    bad_code = (
        "import numpy as np\n"
        "def create_metric(context):\n"
        "    def metric(front):\n"
        "        return 1.2345\n"
        "    return metric\n"
    )
    bundle = {
        "artifact_type": "metric",
        "base_name": "HV_LLM_MONTECARLO_BAD",
        "cpu_code": bad_code,
        "jax_code": bad_code,
        "cpu_file": "HV_BAD.py",
        "jax_file": "HV_BAD_JAX.py",
        "n_var": 30,
        "n_obj": 3,
    }
    rep = S.validate_artifact_bundle_detailed(bundle)
    assert rep.get("ok") is False
    assert any("HV Monte Carlo" in msg for msg in rep.get("issues", []))


def test_llm_metric_variant_skips_oracle_but_keeps_runtime_checks():
    bundle = S.generate_artifact_bundle(
        "Create an IGD-like metric variant for pymoolab",
        artifact_type="metric",
        base_name="IGD_LIKE_PROMPT",
        n_var=30,
        n_obj=3,
        provider="local_template",
    )
    rep = bundle.get("_validation_report", {})
    runtime = rep.get("runtime", {})
    assert rep.get("ok") is True
    assert runtime.get("oracle_skipped_reason") == "variant_or_novel_metric_request"
    assert "cpu_jax_rel_err" in runtime


def test_llm_problem_template_reports_pf_mode_and_probe():
    bundle = S.generate_artifact_bundle(
        "Create a simple bi-objective continuous benchmark problem",
        artifact_type="problem",
        base_name="toy_problem_llm",
        n_var=30,
        n_obj=2,
        provider="local_template",
    )
    rep = bundle.get("_validation_report", {})
    runtime = rep.get("runtime", {})
    assert rep.get("ok") is True
    assert runtime.get("pf_mode") in {"approximate", "exact", "custom", "unavailable"}
    assert "pf_probe_status" in runtime


def test_llm_problem_known_zdt_uses_canonical_problem_fallback():
    bundle = S.generate_artifact_bundle(
        "Create ZDT1 problem for pymoolab with standard defaults",
        artifact_type="problem",
        base_name="zdt1_prompt",
        n_var=30,
        n_obj=2,
        provider="local_template",
    )
    cpu_code = str(bundle.get("cpu_code", ""))
    jax_code = str(bundle.get("jax_code", ""))
    assert "from problems.multi.zdt import ZDT1 as _CanonicalProblem" in cpu_code
    assert "from problems.multi.zdt import ZDT1_JAX as _CanonicalProblem" in jax_code
    assert bundle.get("_validation_report", {}).get("ok") is True


def test_llm_spec_first_local_template_embeds_spec_report():
    bundle = S.generate_artifact_bundle(
        "Create a custom front-spacing metric variant",
        artifact_type="metric",
        base_name="spacing_prompt_variant",
        n_var=30,
        n_obj=3,
        provider="local_template",
        spec_first=True,
    )
    assert bundle.get("_spec_first") is True
    spec = bundle.get("_spec_report", {})
    assert isinstance(spec, dict) and spec
    assert spec.get("mode") in {"local_template_spec", "spec_first_failed"}


def test_llm_ast_hardening_blocks_risky_calls():
    code = (
        "def create_metric(context):\n"
        "    def metric(front):\n"
        "        open('x.txt','w').write('x')\n"
        "        return 0.0\n"
        "    return metric\n"
    )
    ok, issues = S.validate_metric_code(code)
    assert ok is False
    assert any("Unsupported call: open(...)" in msg for msg in issues)


def test_validate_key_access_uses_cache(monkeypatch):
    key = "test-key"
    cache_key = str(hash(key))
    S._KEY_VALIDATION_CACHE[cache_key] = {
        "has_key": True,
        "auth_ok": True,
        "route_ok": True,
        "message": "cached",
        "model": "claude-sonnet-4-6",
        "_ts": __import__("time").time(),
    }
    out = S.validate_anthropic_key_access(key, timeout_s=0.1)
    assert out.get("cached") is True
    assert out.get("route_ok") is True
    assert out.get("message") == "cached"
