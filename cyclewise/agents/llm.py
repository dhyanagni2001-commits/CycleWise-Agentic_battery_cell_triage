"""LLM calls with schema validation, retries, a response cache, and a call budget.

Backends (CYCLEWISE_LLM env var, default "auto"):
  anthropic  Claude via the Anthropic SDK (ANTHROPIC_API_KEY or an `ant auth login` profile)
  offline    no model call; the agent's deterministic fallback is used and logged
  auto       anthropic if credentials resolve, else offline

Determinism: current Claude models do not accept a temperature parameter, so
reproducibility comes from the cache. Every response is stored under
cache/llm/<sha256(model, system, prompt, schema)>.json, and replay reads it back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from cyclewise.config import ROOT, load_config

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
DEFAULT_MODEL = "claude-opus-5"
CACHE = ROOT / "cache" / "llm"


@dataclass
class LLMOutcome:
    value: BaseModel
    model: str
    fallback_used: bool
    attempts: int
    cached: bool
    errors: list[str]


class _Budget:
    calls = 0


def backend() -> str:
    b = os.environ.get("CYCLEWISE_LLM", "auto")
    if b != "auto":
        return b
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    return "offline"


def model_name() -> str:
    return os.environ.get("CYCLEWISE_MODEL", DEFAULT_MODEL)


def _extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def _call_anthropic(system: str, prompt: str) -> str:
    import anthropic

    cfg = load_config()["agents"]
    client = anthropic.Anthropic(timeout=float(cfg["llm_timeout_s"]), max_retries=3)
    resp = client.messages.create(
        model=model_name(),
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused")
    return "".join(b.text for b in resp.content if b.type == "text")


def complete(schema: type[T], system: str, prompt: str, *, fallback: Callable[[], T],
             check: Callable[[T], None] | None = None) -> LLMOutcome:
    """Ask for JSON matching `schema`; validate (pydantic + `check`); retry with the
    validation error up to max_json_retries; then use the deterministic fallback."""
    cfg = load_config()["agents"]
    be = backend()
    if be == "offline":
        return LLMOutcome(fallback(), "offline-deterministic", True, 0, False, ["offline backend"])

    schema_txt = json.dumps(schema.model_json_schema(), sort_keys=True)
    full_system = (f"{system}\n\nReply with a single JSON object that validates against this JSON "
                   f"schema, and nothing else:\n{schema_txt}")
    errors: list[str] = []
    msg = prompt
    for attempt in range(cfg["max_json_retries"] + 1):
        # Attempt index is part of the key: an identical retry prompt must reach the model
        # again, not replay the cached failure. Replays still follow the same sequence.
        key = hashlib.sha256(json.dumps([model_name(), full_system, msg, attempt]).encode()).hexdigest()
        cpath = CACHE / f"{key}.json"
        cached = cpath.exists()
        try:
            if cached:
                text = json.loads(cpath.read_text())["text"]
            else:
                if _Budget.calls >= cfg["max_llm_calls_per_run"]:
                    raise RuntimeError("max LLM calls per run reached")
                _Budget.calls += 1
                text = _with_backoff(lambda: _call_anthropic(full_system, msg))
                CACHE.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps({"model": model_name(), "text": text}))
            value = schema.model_validate(_extract_json(text))
            if check:
                check(value)
            return LLMOutcome(value, model_name(), False, attempt + 1, cached, errors)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            errors.append(str(e)[:800])
            msg = (f"{prompt}\n\nYour previous reply failed validation:\n{str(e)[:800]}\n"
                   "Fix it and reply with the corrected JSON object only.")
        except Exception as e:  # network, auth, budget: do not retry the JSON loop
            errors.append(f"{type(e).__name__}: {e}"[:800])
            break
    log.warning("LLM fallback used for %s: %s", schema.__name__, errors[-1] if errors else "")
    return LLMOutcome(fallback(), model_name(), True, len(errors), False, errors)


def _with_backoff(fn: Callable[[], str], tries: int = 3) -> str:
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            name = type(e).__name__
            if i == tries - 1 or name in {"AuthenticationError", "PermissionDeniedError",
                                           "BadRequestError", "NotFoundError"}:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")
