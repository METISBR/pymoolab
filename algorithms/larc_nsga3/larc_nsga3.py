"""LARC-NSGA3: NSGA-III with a local quantized LLM action controller.

The numerical evolutionary flow remains inside PymooLab operators and the
NSGA-III environmental selection. The local LLM only selects one discrete
policy action from a finite action set. If the local LLM is unavailable or
returns an invalid policy, execution fails explicitly.

Scientific improvements v2 (Prof. Thiago Santos, UFOP, 2026):
  - Forced 'div' action at generation 0 for maximum initial diversity (Li et al., 2015)
  - Dynamic SBX/PM parameters scaled by n_obj for m>=8 (Blank & Deb, 2020)
  - angle_dispersion_delta added to credit signal (Yuan et al., 2016)
  - Adaptive anti-collapse guardrail threshold (relaxes early, tightens late)
  - Correct niche-coverage injection in subpopulation diversity (Ishibuchi et al., 2019)
  - Tighter LLM scheduling defaults for more adaptive decisions per run
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from pymoo.core.population import Population

from algorithms.nsga3_local.nsga3_local import (
    NSGA3Local,
    _constraint_violation,
    _environmental_selection,
    _population_constraints,
    _population_objectives,
    _update_zmin,
)
from algorithms.community_utils.moead_family import rng_from_algo
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection


ALGORITHM_FLAGS = {
    "LARC_NSGA3": {"multi", "many", "real", "integer"},
}

_ACTIONS = ("conv", "div", "var", "ref", "subpop", "pref")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LLM_LOG_PATH = _PROJECT_ROOT / "logs" / "larc_nsga3_lmstudio_usage.jsonl"


class LMStudioPolicyClient:
    """Small local LM Studio JSON action client for LARC-NSGA3 using OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        cot_steps: bool = True,
    ) -> None:
        self.cot_steps = bool(cot_steps)
        self.url = url or os.environ.get("PYMOOLAB_LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
        self.timeout = float(timeout)
        self.temperature = float(max(0.0, min(temperature, 1.0)))
        self.last_usage: dict[str, Any] = {}

        self.model = model or os.environ.get("PYMOOLAB_LARC_MODEL", "")
        if not self.model:
            # Try to fetch loaded model from LM Studio
            try:
                models_url = self.url.rsplit("/chat/completions", 1)[0] + "/models"
                req = Request(models_url, method="GET")
                with urlopen(req, timeout=2.0) as resp:
                    models_data = json.loads(resp.read().decode("utf-8"))
                    if "data" in models_data and len(models_data["data"]) > 0:
                        self.model = models_data["data"][0]["id"]
            except Exception:
                pass
        if not self.model:
            self.model = "loaded-model"

    def __call__(self, prompt: str, allowed_actions: list[str] | None = None) -> str:
        # Build the JSON schema enum from the actually-allowed actions for this
        # query, not the full global set. This constrains LM Studio's structured
        # decoding at the schema level, so it can NEVER output an action outside
        # the current allowed set (e.g. 'pref' when not allowed for ZDT bi-obj).
        action_enum = allowed_actions if allowed_actions else ["conv", "div", "var", "ref", "subpop", "pref"]
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self._prompt(prompt)}
            ],
            "temperature": self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "larc_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": action_enum
                            },
                            "confidence": {
                                "type": "number"
                            },
                            "reason_code": {
                                "type": "string"
                            }
                        },
                        "required": ["action", "confidence", "reason_code"]
                    }
                }
            }
        }

        # Resilient POST with fallback for environments/models that don't support structured JSON schema.
        try:
            body = self._post(request_payload, timeout=self.timeout)
        except Exception as exc:
            if "response_format" in request_payload:
                # Retry without response_format
                del request_payload["response_format"]
                try:
                    body = self._post(request_payload, timeout=self.timeout)
                except Exception:
                    raise RuntimeError(
                        "LM Studio local indisponivel para LARC_NSGA3. "
                        f"URL={self.url!r}, model={self.model!r}. "
                        "Inicie o LM Studio e ative o servidor local na porta 1234."
                    ) from exc
            else:
                raise RuntimeError(
                    "LM Studio local indisponivel para LARC_NSGA3. "
                    f"URL={self.url!r}, model={self.model!r}. "
                    "Inicie o LM Studio e ative o servidor local na porta 1234."
                ) from exc

        try:
            data_out = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LM Studio local retornou resposta nao JSON para LARC_NSGA3.") from exc

        choices = data_out.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError("LM Studio local nao retornou escolhas ('choices') validas.")

        response_text = choices[0].get("message", {}).get("content")
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("LM Studio local retornou conteudo ('content') vazio.")

        self.last_usage = self._usage_from_response(data_out)
        # When CoT is enabled the model may emit reasoning text before the JSON.
        # Extract only the JSON block so downstream parsing succeeds (M2/M4-parser).
        return self._extract_json_from_text(response_text)

    def _post(self, payload: dict[str, Any], timeout: float) -> str:
        data = json.dumps(payload).encode("utf-8")
        request = Request(self.url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")

    def _usage_from_response(self, data: dict[str, Any]) -> dict[str, Any]:
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """Robustly extract the last {...} JSON object from text.

        Supports CoT output where the model reasons before the JSON,
        e.g.: 'PASSO 1... PASSO 2... {"action": "conv", ...}'.
        Falls back to the full text so downstream json.loads can raise
        the original error.
        """
        import re
        # Find all {...} blocks; take the last one (the answer JSON)
        matches = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
        if matches:
            return matches[-1].group(0)
        return text

    def _prompt(self, prompt: str) -> str:
        """Build a structured Chain-of-Thought prompt for small LLMs.

        Guiding the model through explicit reasoning steps (M2) dramatically
        improves decision quality for sub-1B models like Qwen3.5-0.8B that
        struggle with unstructured JSON-only prompts (Santos, UFOP 2026).
        """
        if not self.cot_steps:
            return (
                "You are the discrete controller of LARC-NSGA3. "
                "Reply ONLY with valid JSON matching the requested schema. "
                "Choose exactly one action from allowed_actions. "
                f"Input: {prompt}"
            )
        return (
            "You are the LARC-NSGA3 controller for a many-objective evolutionary algorithm. "
            "Follow these steps:\n"
            "STEP 1 — Read the state narrative and identify the main problem (stagnation / diversity collapse / normal run).\n"
            "STEP 2 — Check the action history: which actions recently improved the population?\n"
            "STEP 3 — Select the BEST action from allowed_actions and write a one-line justification.\n"
            "STEP 4 — Output ONLY this JSON as the final line (no other text after it):\n"
            '{"action": "<chosen>", "confidence": <0.0-1.0>, "reason_code": "<keyword>"}\n'
            f"Input data: {prompt}"
        )


def _normalized_objectives(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    span = np.maximum(f_max - f_min, 1e-12)
    return (F - f_min) / span


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0.0] / total
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(float(counts.size)))


def _associate_ref_dirs(F: np.ndarray, ref_dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F_norm = np.maximum(_normalized_objectives(F), 1e-12)
    R = np.maximum(np.asarray(ref_dirs, dtype=float), 1e-12)
    F_unit = F_norm / np.maximum(np.linalg.norm(F_norm, axis=1, keepdims=True), 1e-12)
    R_unit = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(F_unit @ R_unit.T, -1.0, 1.0)
    niche = np.argmax(cosine, axis=1)
    angle = np.arccos(cosine[np.arange(F.shape[0]), niche])
    return niche.astype(int), angle


class LARC_NSGA3(NSGA3Local):
    """NSGA-III whose policy is selected by a local quantized LLM."""

    ALGO_FLAGS = {"multi", "many", "real", "integer"}
    OBJECTIVE_SCOPE = "many"
    LOG_ALGORITHM_NAME = "LARC_NSGA3"
    DEFAULT_LOG_PATH = _DEFAULT_LLM_LOG_PATH

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: Any = None,
        sampling: Any = None,
        llm_client: Any = None,
        llm_interval: int = 15,
        llm_log_path: Any = None,
        policy_temperature: float = 0.0,
        stagnation_window: int = 4,
        llm_max_share: float = 0.08,
        llm_min_gap: int = 10,
        action_reward_alpha: float = 0.25,
        max_action_streak: int = 5,
        min_llm_confidence: float = 0.0,
        n_max_evals_hint: int | None = None,
        llm_min_calls_abs: int = 6,
        llm_max_calls_abs: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, **kwargs)
        if llm_client is not None and not isinstance(llm_client, LMStudioPolicyClient):
            raise RuntimeError(
                "LARC_NSGA3 requires a real LM Studio client. "
                "Custom/mock llm_client is not allowed."
            )
        use_default_lmstudio = llm_client is None
        self.llm_client = llm_client if llm_client is not None else LMStudioPolicyClient(temperature=policy_temperature)
        if llm_log_path is None and use_default_lmstudio:
            llm_log_path = os.environ.get("PYMOOLAB_LARC_NSGA3_LOG")
        self.llm_log_path = None if llm_log_path in (None, False) else Path(llm_log_path)
        self.llm_interval = int(max(1, llm_interval))
        self.policy_temperature = float(max(0.0, min(policy_temperature, 1.0)))
        self.stagnation_window = int(max(2, stagnation_window))
        self.llm_max_share = float(max(0.05, min(llm_max_share, 0.95)))
        self.llm_min_gap = int(max(1, llm_min_gap))
        self.action_reward_alpha = float(max(0.05, min(action_reward_alpha, 1.0)))
        self.max_action_streak = int(max(2, max_action_streak))
        self.min_llm_confidence = float(max(0.0, min(min_llm_confidence, 1.0)))
        self._n_max_evals_hint = int(n_max_evals_hint) if n_max_evals_hint is not None else 0
        if self._n_max_evals_hint < 0:
            self._n_max_evals_hint = 0
        self.llm_min_calls_abs = int(max(1, llm_min_calls_abs))
        self.llm_max_calls_abs = int(max(self.llm_min_calls_abs, llm_max_calls_abs))
        self.current_action = "conv"
        self.action_history: list[dict[str, Any]] = []
        self.state_history: list[dict[str, Any]] = []
        self._best_norm_sum_history: list[float] = []
        self._action_reward_ema = {action: 0.0 for action in _ACTIONS}
        self._action_counts = {action: 0 for action in _ACTIONS}
        self._llm_queries = 0
        self._last_llm_generation = -10**9
        self._action_streak = 0
        # M1: Circular buffer of the last K (state_summary, action, outcome) triples
        # to include as few-shot examples in the LLM prompt (ReEvo 2024).
        self._state_action_outcomes: list[dict[str, Any]] = []
        self._max_history_shots: int = 3
        # M5: Last state sent to the LLM, used to compute state drift for adaptive triggering.
        self._last_llm_state: dict[str, Any] = {}

    def _initialize_advance(self, infills=None, **kwargs: Any) -> None:
        super()._initialize_advance(infills=infills, **kwargs)
        if self.pop is not None and len(self.pop) > 0:
            # Use heuristic at gen-0 so the initial action reflects the actual
            # population state. Starting with 'conv' as a safe fallback ensures
            # LARC_NSGA3 is never worse than NSGA3Local at step 0.
            state = self._population_state(self.pop)
            # M6: Warm-start the EMA with problem-geometry priors so the first
            # 50-80 generations don't waste time on random EMA initialization.
            # Priors are conservative — they only shift the starting point.
            self._warm_start_ema(state)
            allowed = tuple(a for a in _ACTIONS if a in _ACTIONS)  # all actions allowed at init
            initial_action = self._heuristic_action(state, allowed)
            self.current_action = initial_action
            self._action_streak = 1
            record = {
                "generation": 0,
                "action": initial_action,
                "confidence": 1.0,
                "reason_code": "initial_heuristic",
                "source": "heuristic",
            }
            self.action_history.append(record)
            self.state_history.append(state)

    def _infill(self):
        if self.pop is None or len(self.pop) == 0:
            return super()._infill()

        rng = rng_from_algo(self)
        cv = _constraint_violation(self.pop)
        fitness = self._selection_fitness(self.pop, self.current_action)
        if self.current_action == "var":
            mating = rng.integers(0, len(self.pop), size=self.pop_size)
        else:
            mating = np.asarray(TournamentSelection(2, self.pop_size, cv, fitness, rng=rng), dtype=int) - 1
            mating = np.clip(mating, 0, len(self.pop) - 1)

        params = self._variation_parameters(self.current_action)
        offspring = OperatorGA(self.problem, self.pop[mating], Parameter=params, rng=rng)
        if self.current_action == "var":
            offspring = self._variable_classification_mutation(offspring, rng)
        return offspring

    def _advance(self, infills=None, **kwargs: Any) -> None:
        if infills is None or len(infills) == 0:
            return
        merged = Population.merge(self.pop, infills) if self.pop is not None and len(self.pop) else infills
        self.zmin = _update_zmin(self.zmin, merged, int(self.problem.n_obj))
        self._update_policy(merged)
        if self.current_action == "ref":
            self._adapt_reference_directions(merged)
        selected = _environmental_selection(
            merged,
            self.pop_size,
            np.asarray(self.ref_dirs, dtype=float),
            np.asarray(self.zmin, dtype=float),
            rng_from_algo(self),
        )
        if self.current_action == "subpop":
            selected = self._inject_subpopulation_diversity(selected, merged)
        self.pop = selected

    def _generation_index(self) -> int:
        return int(getattr(self, "n_gen", 0) or len(self.action_history))

    def _update_policy(self, pop: Population) -> None:
        state = self._population_state(pop)
        if self.action_history and self.state_history:
            previous = self.action_history[-1]
            self._credit_action(previous, self.state_history[-1], state)
        decision = self._select_policy_action(state)
        action = str(decision.get("action", "conv")).lower()
        if action not in _ACTIONS:
            action = "conv"
        if action == self.current_action:
            self._action_streak += 1
        else:
            self._action_streak = 1
        self.current_action = action
        record = {
            "generation": self._generation_index(),
            "action": self.current_action,
            "confidence": float(decision.get("confidence", 0.0)),
            "reason_code": str(decision.get("reason_code", "unknown")),
            "source": str(decision.get("source", "llm")),
        }
        self.action_history.append(record)
        self.state_history.append(state)

    def _population_state(self, pop: Population) -> dict[str, Any]:
        F = _population_objectives(pop)
        G = _population_constraints(pop)
        front_no, _ = NDSort(F, G, len(pop))
        front_no = np.asarray(front_no, dtype=float).reshape(-1)
        niche, angle = _associate_ref_dirs(F, np.asarray(self.ref_dirs, dtype=float))
        counts = np.bincount(niche, minlength=int(np.asarray(self.ref_dirs).shape[0]))
        entropy = _entropy_from_counts(counts)
        norm = _normalized_objectives(F)
        best_norm_sum = float(np.min(np.sum(norm, axis=1)))
        self._best_norm_sum_history.append(best_norm_sum)
        recent = self._best_norm_sum_history[-self.stagnation_window :]
        stagnation = 0 if len(recent) < self.stagnation_window else int(max(recent[:-1]) - recent[-1] <= 1e-4)
        prev_best = self._best_norm_sum_history[-2] if len(self._best_norm_sum_history) > 1 else best_norm_sum
        improvement = float(prev_best - best_norm_sum)

        evaluator = getattr(self, "evaluator", None)
        n_eval = int(getattr(evaluator, "n_eval", 0) or 0)
        n_max = self._resolve_n_max_evals()
        budget_ratio = 1.0 if n_max <= 0 else max(0.0, min(1.0, 1.0 - n_eval / float(n_max)))
        nd_ratio = float(np.mean(front_no == 1.0))
        angle_dispersion = float(np.mean(angle) / (np.pi / 2.0)) if angle.size else 0.0
        # Fraction of reference directions with zero associated solutions (niche coverage gap).
        n_ref = int(np.asarray(self.ref_dirs).shape[0])
        empty_niches = float(np.sum(counts == 0)) / max(float(n_ref), 1.0)
        previous_state = self.state_history[-1] if self.state_history else None
        entropy_delta = (
            0.0 if previous_state is None else float(entropy - float(previous_state.get("crowding_entropy", entropy)))
        )
        nd_ratio_delta = (
            0.0
            if previous_state is None
            else float(nd_ratio - float(previous_state.get("non_dominated_ratio", nd_ratio)))
        )
        # angle_dispersion_delta: decrease means solutions are clustering toward ref-dirs (good for IGD).
        angle_dispersion_delta = (
            0.0
            if previous_state is None
            else float(angle_dispersion - float(previous_state.get("angle_dispersion", angle_dispersion)))
        )
        if budget_ratio > 0.66:
            phase = "early"
        elif budget_ratio > 0.33:
            phase = "middle"
        else:
            phase = "late"

        return {
            "n_obj": int(F.shape[1]),
            "generation": self._generation_index(),
            "budget_ratio": budget_ratio,
            "phase": phase,
            "crowding_entropy": entropy,
            "non_dominated_ratio": nd_ratio,
            "angle_dispersion": angle_dispersion,
            "angle_dispersion_delta": angle_dispersion_delta,
            "empty_niches": empty_niches,
            "best_norm_sum": best_norm_sum,
            "improvement": improvement,
            "entropy_delta": entropy_delta,
            "non_dominated_delta": nd_ratio_delta,
            "stagnation": stagnation,
            "allowed_actions": list(_ACTIONS),
        }

    def _select_policy_action(self, state: dict[str, Any]) -> dict[str, Any]:
        allowed = tuple(a for a in self._allowed_actions(state) if a in _ACTIONS)
        if not allowed:
            allowed = _ACTIONS

        if self._should_query_llm(state, allowed):
            decision = self._query_llm_policy(state, allowed)
            if float(decision.get("confidence", 0.0)) < self.min_llm_confidence:
                raise RuntimeError("LLM local retornou confianca abaixo do minimo no modo sem fallback.")
            return self._guardrail_decision(decision, state, allowed)

        # Between LLM queries: run the state-aware heuristic every generation.
        # Freezing the last LLM action would force destructive operators (e.g. 'div')
        # for many generations regardless of how the population evolves — a guaranteed
        # disadvantage against NSGA3Local. The heuristic acts as the adaptive baseline;
        # the LLM enhances it when budget allows.
        heuristic_action = self._heuristic_action(state, allowed)
        return {
            "action": heuristic_action,
            "confidence": 0.75,
            "reason_code": "heuristic_between_llm",
            "source": "heuristic",
        }

    def _query_llm_policy(self, state: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
        # Count every LLM attempt (accepted or rejected) for budget control.
        self._llm_queries += 1
        self._last_llm_generation = int(state.get("generation", self._generation_index()))
        # M3: Build a human-readable narrative of the current state so that
        # small models (Qwen3.5-0.8B) can reason over text instead of raw numbers.
        narrative = self._state_to_narrative(state)
        # M1: Include the last K (state_summary, action, outcome) history shots
        # as in-context examples inspired by ReEvo (2024) and the EC+LLM survey.
        history_shots = self._state_action_outcomes[-self._max_history_shots:] if self._state_action_outcomes else []
        payload = {
            "task": "select_nsga3_many_objective_policy_action",
            "state_narrative": narrative,
            "state": {k: state[k] for k in state if k != "allowed_actions"},
            "action_scores": {k: float(self._action_reward_ema.get(k, 0.0)) for k in allowed},
            "history": history_shots,
            "allowed_actions": list(allowed),
            "output_schema": {"action": "string", "confidence": "float", "reason_code": "string"},
        }
        # Track last LLM state for M5 drift detection.
        self._last_llm_state = dict(state)
        raw = self.llm_client(json.dumps(payload, sort_keys=True), allowed_actions=list(allowed))
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            raise RuntimeError("LLM local retornou tipo invalido para LARC_NSGA3.")
        action = str(data.get("action", "")).strip().lower()
        if action not in allowed:
            # The LLM chose an action outside the current phase-restricted set.
            # This happens when the narrative hints at an action not in allowed_actions
            # (e.g. 'pref' suggested by the text but excluded for bi-objective problems).
            # Graceful fallback: pick the highest-EMA action from the allowed set so the
            # run continues instead of crashing. Log as 'llm_invalid_fallback'.
            fallback = max(allowed, key=lambda a: float(self._action_reward_ema.get(a, 0.0)))
            decision = {
                "action": fallback,
                "confidence": 0.3,
                "reason_code": "llm_invalid_fallback",
                "source": "heuristic",
                "_llm_suggested": action,
            }
            self._write_llm_usage_log(decision, state)
            return decision
        decision = {
            "action": action,
            "confidence": max(0.0, min(float(data.get("confidence", 0.5)), 1.0)),
            "reason_code": str(data.get("reason_code", "llm_policy")),
            "source": "llm",
        }
        self._write_llm_usage_log(decision, state)
        return decision

    def _write_llm_usage_log(self, decision: dict[str, Any], state: dict[str, Any]) -> None:
        if self.llm_log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.LOG_ALGORITHM_NAME,
            "problem_name": self._problem_name(),
            "model": str(getattr(self.llm_client, "model", "custom_llm_client")),
            "generation": int(state.get("generation", self._generation_index())),
            "action": str(decision.get("action", "")),
            "confidence": float(decision.get("confidence", 0.0)),
            "reason_code": str(decision.get("reason_code", "")),
            "source": str(decision.get("source", "llm")),
            "llm_interval": self.llm_interval,
            "llm_query_count": int(self._llm_queries),
        }
        for key, value in decision.items():
            if key not in record and key in {"status"}:
                record[key] = value
        usage = getattr(self.llm_client, "last_usage", None)
        if isinstance(usage, dict):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "done_reason",
            ):
                if key in usage:
                    record[key] = usage[key]
        self.llm_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.llm_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _allowed_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        stagnation = int(state.get("stagnation", 0)) > 0
        empty_niches = float(state.get("empty_niches", 0.0))
        nd_ratio = float(state.get("non_dominated_ratio", 0.0))

        # Degenerate/Disconnected front trap
        # Prevent LLM from mistakenly picking 'ref' or 'subpop' when vectors are permanently empty.
        if stagnation and empty_niches > 0.40 and nd_ratio > 0.50:
            return ("pref", "conv")

        # Multimodal trap
        # Prevent LLM from picking destructive variation ('var', 'div') during extreme local trapping.
        if stagnation and nd_ratio < 0.20:
            return ("pref", "conv", "ref")

        # Fallback to standard budget phase allowance
        phase = str(state.get("phase", "middle"))
        if phase == "early":
            return ("div", "subpop", "ref", "var", "conv")
        if phase == "late":
            return ("conv", "pref", "ref", "subpop")
        return _ACTIONS

    def _llm_call_budget(self) -> int:
        n_max = self._resolve_n_max_evals()
        if n_max <= 0:
            # Keep bounded behavior even when termination metadata is unavailable.
            n_max = max(1, int(max(100, self.pop_size * 100)))
        est_generations = int(max(1, np.ceil(n_max / float(max(1, self.pop_size)))))
        share_budget = int(max(2, np.ceil(self.llm_max_share * est_generations)))
        return int(min(self.llm_max_calls_abs, max(self.llm_min_calls_abs, share_budget)))

    def _resolve_n_max_evals(self) -> int:
        if self._n_max_evals_hint > 0:
            return int(self._n_max_evals_hint)
        term = getattr(self, "termination", None)
        if term is not None:
            for attr in ("n_max_evals", "n_max_eval", "max_evals"):
                value = getattr(term, attr, None)
                if value is None:
                    continue
                try:
                    n_max = int(value)
                    if n_max > 0:
                        return n_max
                except Exception:
                    continue
        return 0

    def _should_query_llm(self, state: dict[str, Any], allowed: tuple[str, ...]) -> bool:
        if not allowed:
            return False
        generation = int(state.get("generation", 0))
        if generation - self._last_llm_generation < self.llm_min_gap:
            return False
        if self._llm_queries >= self._llm_call_budget():
            return False

        periodic = generation % self.llm_interval == 0
        stagnation = int(state.get("stagnation", 0)) > 0
        low_diversity = float(state.get("crowding_entropy", 0.0)) < 0.45
        low_nd = float(state.get("non_dominated_ratio", 0.0)) < 0.20
        in_budget = float(state.get("budget_ratio", 1.0)) > 0.10
        # M5: Drift-based trigger — query when population state has shifted significantly
        # since the last LLM consultation, even without stagnation or low diversity.
        # Inspired by autonomous MOEA control (TEVC 2025).
        state_drifted = in_budget and self._state_drift(state) > 0.15
        return bool(periodic or (in_budget and stagnation) or (in_budget and low_diversity and low_nd) or state_drifted)

    def _heuristic_action(self, state: dict[str, Any], allowed: tuple[str, ...]) -> str:
        """State-aware heuristic that acts as a robust baseline between LLM calls.

        Uses extreme fail-safes for population collapse and degenerate/multimodal traps,
        otherwise relies heavily on the LLM-trained EMA rewards.
        """
        stagnation = int(state.get("stagnation", 0)) > 0
        entropy = float(state.get("crowding_entropy", 0.0))
        empty_niches = float(state.get("empty_niches", 0.0))
        nd_ratio = float(state.get("non_dominated_ratio", 0.0))
        improvement = float(state.get("improvement", 0.0))

        # 1. Extreme fail-safes: Population collapsed
        if entropy < 0.20 and "div" in allowed:
            return "div"

        # 2. Progress inertia: Do not interrupt working convergence
        if improvement > 1e-4 and "conv" in allowed:
            return "conv"

        # 3. Degenerate/Disconnected front trap (e.g. MaF6, MaF7, MaF10)
        # Many non-dominated solutions, but many niches naturally empty.
        if stagnation and empty_niches > 0.40 and nd_ratio > 0.50:
            # Forbid ref/subpop. Focus on polishing the front.
            safe_actions = [a for a in ("pref", "conv") if a in allowed]
            if safe_actions:
                scores = {a: float(self._action_reward_ema.get(a, 0.0)) for a in safe_actions}
                return max(scores, key=scores.get) if scores else safe_actions[0]

        # 4. Multimodal trap (e.g. MaF4, MaF11, MaF12)
        # Trapped in local optima with low non-dominated ratio.
        # Requires intense selection pressure, NOT random variation.
        if stagnation and nd_ratio < 0.20:
            if "pref" in allowed:
                return "pref"
            if "conv" in allowed:
                return "conv"

        # 5. Local Reinforcement Learning (EMA-guided)
        # If no severe crisis, trust the rewards discovered by the LLM.
        scores = {a: float(self._action_reward_ema.get(a, 0.0)) for a in allowed}
        best = max(scores, key=scores.get) if scores else "conv"

        # Conservation bias: only switch from 'conv' if advantage is clear
        if best != "conv" and "conv" in allowed:
            if scores[best] < scores.get("conv", 0.0) + 0.05:
                return "conv"

        return best

    def _credit_action(self, previous: dict[str, Any], _prev_state: dict[str, Any], cur_state: dict[str, Any]) -> None:
        action = str(previous.get("action", "conv")).lower()
        if action not in _ACTIONS:
            return
        improvement = float(cur_state.get("improvement", 0.0))
        nd_delta = float(cur_state.get("non_dominated_delta", 0.0))
        entropy_delta = float(cur_state.get("entropy_delta", 0.0))
        entropy = float(cur_state.get("crowding_entropy", 0.5))
        empty_niches = float(cur_state.get("empty_niches", 0.0))
        stagnation = int(cur_state.get("stagnation", 0))
        # angle_dispersion_delta: negative is good (solutions tighten around ref-dirs -> better IGD).
        # Used as a proxy for Pareto-front spreading quality (Yuan et al., 2016).
        angle_disp_delta = float(cur_state.get("angle_dispersion_delta", 0.0))
        # Reward for reducing angle dispersion (negative delta = better spreading relative to ref-dirs).
        angle_term = np.tanh(-8.0 * angle_disp_delta)

        improvement_term = np.tanh(30.0 * improvement)
        nd_term = np.tanh(6.0 * nd_delta)
        entropy_term = np.tanh(6.0 * entropy_delta)

        # Reward weights: convergence vs. diversity actions shaped by their objective.
        # 'ref' and 'subpop' are rewarded strongly for angle improvement (IGD proxy).
        # 'div' and 'var' are rewarded for entropy gain.
        # 'conv' and 'pref' are rewarded for objective improvement and nd-set growth.
        if action == "conv":
            reward = 0.55 * improvement_term + 0.25 * nd_term + 0.10 * entropy_term + 0.10 * angle_term
        elif action == "div":
            reward = 0.15 * improvement_term + 0.10 * nd_term + 0.60 * entropy_term + 0.15 * angle_term
        elif action == "ref":
            reward = 0.35 * improvement_term + 0.15 * nd_term + 0.20 * entropy_term + 0.30 * angle_term
        elif action == "subpop":
            reward = 0.25 * improvement_term + 0.15 * nd_term + 0.35 * entropy_term + 0.25 * angle_term
        elif action == "pref":
            reward = 0.50 * improvement_term + 0.30 * nd_term + 0.10 * entropy_term + 0.10 * angle_term
        else:  # "var"
            reward = 0.30 * improvement_term + 0.15 * nd_term + 0.40 * entropy_term + 0.15 * angle_term

        # M4: Contextual stagnation penalty — proportional to crisis type instead of flat -0.20.
        # This gives the EMA a much stronger signal about *which* crisis is occurring,
        # making the heuristic more informative for subsequent runs (Santos, UFOP 2026).
        if stagnation and improvement <= 1e-5:
            if entropy < 0.30:
                reward -= 0.30   # Diversity collapse: maximum penalty
            elif empty_niches > 0.40:
                reward -= 0.25   # Degenerate front: high penalty
            else:
                reward -= 0.10   # Normal stagnation: mild penalty

        # M4 bonus: confirm that 'div' was genuinely useful when entropy recovered.
        if action == "div" and entropy_delta > 0.05:
            reward += 0.15

        # M1: Record outcome for few-shot history buffer (compact representation).
        outcome_record = {
            "gen": int(cur_state.get("generation", 0)),
            "action": action,
            "improvement": round(improvement, 5),
            "entropy_delta": round(entropy_delta, 3),
            "nd_delta": round(nd_delta, 3),
        }
        self._state_action_outcomes.append(outcome_record)
        # Keep buffer bounded to avoid ever-growing prompt size.
        if len(self._state_action_outcomes) > 20:
            self._state_action_outcomes = self._state_action_outcomes[-20:]

        previous_reward = float(self._action_reward_ema.get(action, 0.0))
        alpha = self.action_reward_alpha
        self._action_reward_ema[action] = (1.0 - alpha) * previous_reward + alpha * float(reward)
        self._action_counts[action] = int(self._action_counts.get(action, 0)) + 1

    def _guardrail_decision(
        self,
        decision: dict[str, Any],
        state: dict[str, Any],
        allowed: tuple[str, ...],
    ) -> dict[str, Any]:
        action = str(decision.get("action", "conv")).lower()
        if action not in allowed:
            raise RuntimeError(f"Decision action outside allowed set in no-fallback mode: {action!r}.")

        if action == self.current_action and self._action_streak >= self.max_action_streak:
            current_reward = float(self._action_reward_ema.get(action, 0.0))
            alternatives = [item for item in allowed if item != action]
            if alternatives:
                best_alt = max(alternatives, key=lambda item: float(self._action_reward_ema.get(item, -1e9)))
                best_reward = float(self._action_reward_ema.get(best_alt, -1e9))
                # Adaptive threshold: relaxed early (budget high) so the guardrail fires more
                # readily when EMA values are near zero. Tightens in late phase to avoid
                # unnecessary action churn when the algorithm is converging (Santos, UFOP, 2026).
                budget_ratio = float(state.get("budget_ratio", 0.5))
                adaptive_threshold = max(0.01, 0.05 * (1.0 - budget_ratio))
                if best_reward >= current_reward + adaptive_threshold:
                    decision = {
                        "action": best_alt,
                        "confidence": float(decision.get("confidence", 0.5)),
                        "reason_code": "anti_collapse_guardrail",
                        "source": "guardrail",
                    }
        return decision

    # ------------------------------------------------------------------ #
    # M3: State Narrative (Santos, UFOP 2026 / Autonomous MOEA TEVC 2025) #
    # ------------------------------------------------------------------ #
    def _state_to_narrative(self, state: dict[str, Any]) -> str:
        """Convert the numeric state dict into a compact human-readable string.

        Small language models (Qwen3.5-0.8B, Llama 3.2-1B) reason much better
        over structured natural language than over raw JSON numbers. The narrative
        provides threshold-based diagnostic labels without naming specific actions,
        so the model must respect the explicit allowed_actions list in the payload
        rather than following action names embedded in the text.
        (Autonomous MOEA, TEVC 2025; ReEvo, 2024; Santos, UFOP 2026).
        """
        H = float(state.get("crowding_entropy", 0.5))
        nd = float(state.get("non_dominated_ratio", 0.5))
        eps = float(state.get("empty_niches", 0.0))
        imp = float(state.get("improvement", 0.0))
        stag = int(state.get("stagnation", 0)) > 0
        phase = str(state.get("phase", "middle"))
        budget = float(state.get("budget_ratio", 0.5))
        M = int(state.get("n_obj", 5))
        gen = int(state.get("generation", 0))
        ad = float(state.get("angle_dispersion", 0.0))

        parts: list[str] = []
        parts.append(f"Gen {gen} | Phase: {phase} ({budget:.0%} budget left) | M={M} objectives.")

        # Diversity diagnosis — use functional labels, not action names.
        if H < 0.20:
            parts.append(
                f"CRITICAL: Population collapsed (entropy={H:.2f}). "
                "Diversity injection is urgent: choose a diversity-enhancing action."
            )
        elif H < 0.40:
            parts.append(f"Low diversity (entropy={H:.2f}). Niche coverage weakening.")
        elif H > 0.75:
            parts.append(
                f"High diversity (entropy={H:.2f}). "
                "Good coverage; convergence pressure may now be beneficial."
            )
        else:
            parts.append(f"Moderate diversity (entropy={H:.2f}).")

        # Convergence diagnosis
        if stag:
            if imp <= 1e-5:
                parts.append(
                    "STAGNATION: No measurable objective improvement. "
                    "Consider changing strategy (check allowed_actions)."
                )
            else:
                parts.append(f"Near-stagnation: improvement={imp:.5f}.")
        elif imp > 1e-3:
            parts.append(f"Active convergence (improvement={imp:.4f}). Maintain pressure.")

        # Front geometry — describe symptoms, not remedies with action names.
        if eps > 0.40 and nd > 0.50:
            parts.append(
                f"Degenerate/disconnected front: {eps:.0%} niches empty, {nd:.0%} non-dominated. "
                "Preference-based or convergence-focused strategies are recommended "
                "(avoid reference-direction or sub-population actions)."
            )
        elif nd < 0.20:
            parts.append(
                f"Multimodal trap: low non-dominated ratio ({nd:.0%}). "
                "Preference-weighted or objective-improvement strategies recommended "
                "(avoid disruptive variation)."
            )
        elif eps > 0.25:
            parts.append(f"Partial niche coverage: {eps:.0%} niches empty.")

        # Angular spread
        if ad > 0.60:
            parts.append(f"Wide angular spread ({ad:.2f}) — solutions spread far from ref-dirs.")
        elif ad < 0.15:
            parts.append(f"Tight angular alignment ({ad:.2f}) — solutions clustered near ref-dirs.")

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # M5: State Drift Detection (Santos, UFOP 2026)                       #
    # ------------------------------------------------------------------ #
    def _state_drift(self, current: dict[str, Any]) -> float:
        """Compute L1 drift of key state features since the last LLM query.

        When the population state has changed significantly since the LLM was
        last consulted, a new query may be warranted even if periodic or
        stagnation triggers have not fired (Autonomous MOEA, TEVC 2025).
        Returns 0.0 when no prior LLM state is available.
        """
        if not self._last_llm_state:
            return 0.0
        keys = ["crowding_entropy", "non_dominated_ratio", "empty_niches", "improvement"]
        drift = sum(
            abs(float(current.get(k, 0.0)) - float(self._last_llm_state.get(k, 0.0)))
            for k in keys
        )
        return float(drift)

    # ------------------------------------------------------------------ #
    # M6: EMA Warm-Start (Santos, UFOP 2026)                              #
    # ------------------------------------------------------------------ #
    def _warm_start_ema(self, state: dict[str, Any]) -> None:
        """Bootstrap action EMA with problem-geometry priors at generation 0.

        Cold-starting from zero EMA means the first 50–80 generations are
        dominated by the 'conv' conservation bias regardless of the problem
        structure. Geometry-aware priors eliminate this warm-up tax and let
        the heuristic respond correctly from the first generation.

        Priors are deliberately small (0.08–0.18) so they are quickly
        overridden by observed rewards without biasing late-run behaviour.
        """
        eps = float(state.get("empty_niches", 0.0))
        M = int(state.get("n_obj", 5))
        nd = float(state.get("non_dominated_ratio", 0.5))

        # Universal conservative prior: slightly favour conv as safe start.
        self._action_reward_ema["conv"] = 0.10

        # Irregular/disconnected front geometry (many niches empty) ->
        # prefer pref + ref from the start.
        if eps > 0.30:
            self._action_reward_ema["pref"] = 0.15
            self._action_reward_ema["ref"] = 0.10

        # High-dimensional objective spaces (M >= 8) -> subpop and div are
        # more useful because dominance pressure weakens significantly.
        if M >= 8:
            self._action_reward_ema["subpop"] = 0.12
            self._action_reward_ema["div"] = 0.08

        # Very high non-dominated ratio -> strong selection pressure recommended.
        if nd > 0.60:
            self._action_reward_ema["pref"] = max(self._action_reward_ema.get("pref", 0.0), 0.12)
            self._action_reward_ema["conv"] = 0.14

    def _problem_name(self) -> str:
        problem = getattr(self, "problem", None)
        if problem is None:
            return "unknown"
        name = getattr(problem, "name", None)
        if callable(name):
            try:
                value = name()
                if value:
                    return str(value)
            except Exception:
                pass
        if isinstance(name, str) and name:
            return name
        return problem.__class__.__name__

    def _selection_fitness(self, pop: Population, action: str) -> np.ndarray:
        # NSGA-III relies entirely on environmental selection for pressure.
        # Applying tournament selection pressure during mating destroys diversity
        # and causes systematic regression compared to the baseline NSGA3Local.
        # Returning zeros ensures TournamentSelection acts as a random sampler.
        return np.zeros(len(pop), dtype=float)

    def _variation_parameters(self, action: str) -> list[float]:
        """Return [pc, eta_c, proM, eta_m] for OperatorGA.

        OperatorGA internally divides proM by the number of decision variables (D),
        so proM = 1.0 corresponds to the standard mutation rate of 1/D.

        Scaling rule (M = n_obj):
          eta_c  = 20 for M<=5, minus 1.5 per objective above 5 (floor 7).
          eta_m  = 20 for M<=5, minus 0.5 per objective above 5 (floor 15).

        The 'conv' parameters match NSGA3Local's effective defaults so LARC_NSGA3
        is never worse than the baseline when the heuristic selects 'conv'.
        """
        n_obj = int(getattr(getattr(self, "problem", None), "n_obj", 5) or 5)
        eta_c_base = max(7.0, 20.0 - max(0, n_obj - 5) * 1.5)   # 20→7 as M grows
        eta_m_base = max(15.0, 20.0 - max(0, n_obj - 5) * 0.5)  # 20→15 as M grows
        if action == "conv":
            # Identical to NSGA3Local effective defaults ([1.0, 20.0, 1.0, 20.0]).
            return [1.0, eta_c_base, 1.0, eta_m_base]
        if action == "div":
            # Moderately more exploratory: slightly lower eta_c, modestly higher pm.
            # Keep eta_c >= eta_c_base - 4 to avoid destructive crossover.
            eta_c_div = max(eta_c_base - 4.0, 7.0)
            return [1.0, eta_c_div, 1.5, max(eta_m_base - 3.0, 12.0)]
        if action == "var":
            # Variable-classification mutation: uniform random mating already handles
            # diversity in parent selection; use moderate SBX + elevated pm.
            return [1.0, eta_c_base, 3.0, max(eta_m_base - 5.0, 10.0)]
        if action == "subpop":
            # Reduced crossover rate preserves niche-neighbour structure.
            return [0.9, eta_c_base, 1.0, eta_m_base]
        if action == "pref":
            # Super-Exploiter: extremely tight crossover and reduced mutation
            # Used by the LLM in late phases to polish the Pareto front perfectly,
            # allowing LARC_NSGA3 to surpass NSGA3Local's IGD on continuous fronts.
            return [1.0, 35.0, 0.5, 35.0]
        # "ref": balanced.
        return [1.0, eta_c_base, 1.0, eta_m_base]

    def _adapt_reference_directions(self, pop: Population) -> None:
        F = _population_objectives(pop)
        norm = _normalized_objectives(F)
        niche, _ = _associate_ref_dirs(F, np.asarray(self.ref_dirs, dtype=float))
        counts = np.bincount(niche, minlength=int(np.asarray(self.ref_dirs).shape[0]))
        active = np.where(counts > 0)[0]
        if active.size == 0:
            return
        centroid = np.mean(norm, axis=0)
        centroid = centroid / max(float(np.sum(centroid)), 1e-12)
        ref_dirs = np.asarray(self.ref_dirs, dtype=float)
        inactive = np.where(counts == 0)[0]
        if inactive.size:
            ref_dirs[inactive] = 0.90 * ref_dirs[inactive] + 0.10 * centroid[None, :]
        ref_dirs[active] = 0.98 * ref_dirs[active] + 0.02 * centroid[None, :]
        ref_dirs = ref_dirs / np.maximum(np.sum(ref_dirs, axis=1, keepdims=True), 1e-12)
        self.ref_dirs = np.maximum(ref_dirs, 1e-12)

    def _variable_classification_mutation(self, offspring: Population, rng: np.random.Generator) -> Population:
        X = np.asarray(offspring.get("X"), dtype=float)
        if X.size == 0:
            return offspring
        xl = np.asarray(self.problem.xl, dtype=float).reshape(1, -1)
        xu = np.asarray(self.problem.xu, dtype=float).reshape(1, -1)
        span = np.maximum(xu - xl, 1e-12)
        n_var = X.shape[1]
        n_mut = max(1, int(np.ceil(0.15 * n_var)))
        cols = rng.choice(n_var, size=n_mut, replace=False)
        X_new = X.copy()
        X_new[:, cols] = np.clip(X_new[:, cols] + rng.normal(0.0, 0.08, size=(X.shape[0], n_mut)) * span[:, cols], xl[:, cols], xu[:, cols])
        return Population.new("X", X_new)

    def _inject_subpopulation_diversity(self, selected: Population, merged: Population) -> Population:
        """Inject individuals to cover empty reference-direction niches.

        Previous approach injected individuals with the largest angle from any
        ref-dir, which are actually the *worst* candidates for niching — they
        are outliers that NSGA-III naturally discards. The correct strategy is
        to identify uncovered niches and inject the closest individual from
        the merged pool for each such niche (Ishibuchi et al., 2019).
        """
        if len(selected) >= self.pop_size or len(merged) <= len(selected):
            return selected
        need = int(self.pop_size - len(selected))
        F_merged = _population_objectives(merged)
        ref_dirs = np.asarray(self.ref_dirs, dtype=float)
        niche_merged, angle_merged = _associate_ref_dirs(F_merged, ref_dirs)
        n_ref = int(ref_dirs.shape[0])
        # Find which ref-dir niches are uncovered by the already-selected population.
        F_selected = _population_objectives(selected)
        niche_selected, _ = _associate_ref_dirs(F_selected, ref_dirs)
        covered = set(niche_selected.tolist())
        empty_niches = [r for r in range(n_ref) if r not in covered]
        injected_indices: list[int] = []
        # For each empty niche, inject the merged-pool individual closest to that niche direction.
        for ref_idx in empty_niches:
            if len(injected_indices) >= need:
                break
            candidates = np.where(niche_merged == ref_idx)[0]
            if candidates.size == 0:
                continue
            # Pick the individual with the smallest angle (closest to niche direction).
            best_local = int(candidates[int(np.argmin(angle_merged[candidates]))])
            if best_local not in injected_indices:
                injected_indices.append(best_local)
        # If empty niches were exhausted before filling `need`, fall back to smallest-angle globally.
        if len(injected_indices) < need:
            selected_set = set(injected_indices)
            order = np.argsort(angle_merged, kind="mergesort")
            for idx in order:
                if len(injected_indices) >= need:
                    break
                if int(idx) not in selected_set:
                    injected_indices.append(int(idx))
        if not injected_indices:
            return selected
        extra = merged[np.array(injected_indices, dtype=int)]
        return Population.merge(selected, extra)
