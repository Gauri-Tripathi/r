"""Teacher target import and optional HTTP generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from filelock import FileLock
from pydantic import Field

from .io import atomic_json, digest, read_json
from .schema import Example, Strict, TeacherAnswer
from .supervision import CONTRACT, PROMPT_VERSION, TeacherIdentity, teacher_request
from .supervision import checked as checked
from .supervision import imported as imported


class TeacherConfig(Strict):
    endpoint: str
    model: str = Field(min_length=1, max_length=128)
    revision: str = "unspecified"
    budget_usd: float = Field(gt=0, le=10)
    input_per_million: float = Field(gt=0)
    output_per_million: float = Field(gt=0)
    max_output_tokens: int = Field(ge=32, le=256, default=160)
    retries: int = Field(ge=0, le=1, default=0)
    timeout_s: float = Field(ge=1, le=60, default=30)


def api_targets(rows: list[Example], cfg: TeacherConfig, cache: Path) -> list[Example]:
    parts = urlsplit(cfg.endpoint)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Teacher endpoint must be an HTTPS URL without credentials/query")
    key = os.environ.get("TEACHER_API_KEY")
    if not key:
        raise ValueError("Set TEACHER_API_KEY; no automatic provider or paid fallback")
    if any(r.split == "test" for r in rows):
        raise ValueError(
            "API target building excludes test: use a train/dev-only candidate manifest"
        )
    cache.mkdir(parents=True, exist_ok=True)
    output = []
    # Keep requests serial so a failed batch cannot leave billable work running.
    with httpx.Client(timeout=cfg.timeout_s, follow_redirects=False) as client:
        for row in rows:
            prompt = teacher_request(row)
            prompt["response_schema"] = TeacherAnswer.model_json_schema()
            identity = TeacherIdentity(model=cfg.model, revision=cfg.revision, access="api")
            cache_key = digest(
                {
                    "prompt": prompt,
                    "version": PROMPT_VERSION,
                    "schema": TeacherAnswer.model_json_schema(),
                    "config": cfg.model_dump(),
                }
            )
            dest = cache / f"{cache_key}.json"
            if dest.exists():
                output.append(
                    checked(
                        row,
                        TeacherAnswer.model_validate(read_json(dest)).text,
                        identity,
                        contract=CONTRACT,
                    )
                )
                continue
            text = _request(client, cfg, cache, key, prompt)
            answer = checked(row, text, identity, contract=CONTRACT)
            atomic_json(dest, {"text": text})
            output.append(answer)
    return output


def _request(
    client: httpx.Client, cfg: TeacherConfig, cache: Path, key: str, prompt: dict[str, Any]
) -> str:
    # Worst-case input bound using UTF-8 bytes >= possible token count. Failures are charged.
    import json

    content = json.dumps(prompt)
    reserve = (
        len(content.encode()) * cfg.input_per_million
        + cfg.max_output_tokens * cfg.output_per_million
    ) / 1e6
    for attempt in range(cfg.retries + 1):
        ledger = cache / "api-budget.json"
        with FileLock(str(ledger) + ".lock"):
            used = read_json(ledger).get("reserved_usd", 0) if ledger.exists() else 0
            if used + reserve > cfg.budget_usd:
                raise ValueError("Teacher estimated spend ceiling reached")
            atomic_json(ledger, {"reserved_usd": used + reserve, "ceiling_usd": cfg.budget_usd})
        try:
            response = client.post(
                cfg.endpoint,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": cfg.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": cfg.max_output_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "strict": True,
                            "schema": TeacherAnswer.model_json_schema(),
                        },
                    },
                },
            )
        except httpx.RequestError:
            if attempt == cfg.retries:
                raise ValueError(
                    "Teacher connection failed; reserved request cost retained"
                ) from None
            continue
        if response.status_code != 200:
            if response.status_code in (429, 503) and attempt < cfg.retries:
                continue
            raise ValueError(f"Teacher HTTP {response.status_code}; response body suppressed")
        if len(response.content) > 65536:
            raise ValueError("Teacher response exceeds size limit")
        try:
            return TeacherAnswer.model_validate_json(
                response.json()["choices"][0]["message"]["content"]
            ).text
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError("Teacher returned invalid schema; body suppressed") from None
    raise RuntimeError("Teacher attempts exhausted")
