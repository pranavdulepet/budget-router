from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from .semantic import SemanticRouter
from .serialization import to_jsonable

try:
    from starlette.requests import Request as StarletteRequest
except ImportError:  # pragma: no cover - proxy extra is optional
    StarletteRequest = Any


@dataclass(frozen=True, slots=True)
class OpenAIUpstream:
    model: str
    base_url: str
    api_key_env: str
    upstream_model: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not self.base_url.startswith(("http://", "https://")):
            raise ValueError("upstream needs a model and HTTP(S) base_url")
        if not self.api_key_env or "=" in self.api_key_env:
            raise ValueError("api_key_env must name an environment variable")
        secret_headers = {
            key.lower()
            for key in self.headers
            if key.lower() in {"authorization", "x-api-key", "api-key"}
        }
        if secret_headers:
            raise ValueError("credentials must use api_key_env, not static headers")

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        return (
            f"{base}/chat/completions"
            if base.endswith("/v1")
            else f"{base}/v1/chat/completions"
        )


def load_proxy_config(
    path: Path,
) -> tuple[SemanticRouter, dict[str, OpenAIUpstream], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    artifact_path = Path(config["artifact"])
    if not artifact_path.is_absolute():
        artifact_path = (path.parent / artifact_path).resolve()
    router = SemanticRouter(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )
    upstream_rows = config.get("upstreams", {})
    if set(upstream_rows) != set(router.models):
        raise ValueError("proxy upstreams must exactly match router models")
    upstreams = {
        model: OpenAIUpstream(
            model=model,
            base_url=str(value["base_url"]),
            api_key_env=str(value["api_key_env"]),
            upstream_model=(
                str(value["upstream_model"])
                if value.get("upstream_model") is not None
                else None
            ),
            headers={
                str(key): str(header_value)
                for key, header_value in value.get("headers", {}).items()
            },
        )
        for model, value in upstream_rows.items()
    }
    settings = {
        "mode": str(config.get("mode", "balanced")),
        "router_model_name": str(config.get("router_model_name", "auto")),
        "default_output_tokens": int(config.get("default_output_tokens", 1_024)),
        "allow_ungated_policy": bool(
            config.get("allow_ungated_policy", False)
        ),
    }
    return router, upstreams, settings


def _message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {
                    "text",
                    "input_text",
                }:
                    parts.append(str(item.get("text", "")))
    text = "\n".join(part for part in parts if part.strip())
    if not text:
        raise ValueError("messages contain no routable text")
    return text


def _routing_arguments(
    payload: Mapping[str, Any],
    *,
    default_mode: str,
    default_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    options = payload.get("router", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("router options must be an object")
    text = _message_text(payload.get("messages"))
    input_tokens = int(options.get("input_tokens", max(1, len(text) // 4)))
    output_tokens = int(
        options.get(
            "output_tokens",
            payload.get("max_completion_tokens", payload.get("max_tokens", default_output_tokens)),
        )
    )
    arguments = {
        "mode": str(options.get("mode", default_mode)),
        "allowed_models": options.get("allowed_models"),
        "required_capabilities": options.get("required_capabilities", ()),
        "denied_providers": options.get("denied_providers", ()),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "max_cost_usd": options.get("max_cost_usd"),
        "cost_weight": options.get("cost_weight"),
        "quality_threshold": options.get("quality_threshold"),
        "max_quality_drop": float(options.get("max_quality_drop", 0.05)),
    }
    return text, arguments


def create_app(
    router: SemanticRouter,
    upstreams: Mapping[str, OpenAIUpstream],
    *,
    mode: str = "balanced",
    router_model_name: str = "auto",
    default_output_tokens: int = 1_024,
    allow_ungated_policy: bool = False,
    transport: Any = None,
) -> Any:
    """Create an OpenAI-compatible FastAPI app without making provider choices."""

    try:
        import httpx
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("install budget-router[proxy] to create the proxy app") from exc

    if set(upstreams) != set(router.models):
        raise ValueError("proxy upstreams must exactly match router models")
    app = FastAPI(title="Budget Router OpenAI Proxy", version="1")

    def route_payload(payload: Mapping[str, Any]) -> Any:
        try:
            text, arguments = _routing_arguments(
                payload,
                default_mode=mode,
                default_output_tokens=default_output_tokens,
            )
            requested = payload.get("model", router_model_name)
            if requested not in {router_model_name, "auto", None}:
                arguments["allowed_models"] = [str(requested)]
            arguments["allow_ungated_policy"] = allow_ungated_policy
            return router.route(text, **arguments)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "router_artifact_hash": router.artifact_hash,
            "models": list(router.models),
        }

    @app.post("/v1/router/decision")
    async def decision(request: StarletteRequest) -> dict[str, Any]:
        payload = await request.json()
        return to_jsonable(route_payload(payload))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: StarletteRequest) -> Any:
        payload = await request.json()
        decision_value = route_payload(payload)
        upstream = upstreams[decision_value.selected_model]
        api_key = os.environ.get(upstream.api_key_env)
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "required upstream credential environment variable is unset: "
                    f"{upstream.api_key_env}"
                ),
            )
        forwarded = dict(payload)
        forwarded.pop("router", None)
        forwarded["model"] = upstream.upstream_model or decision_value.selected_model
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            **dict(upstream.headers),
        }
        response_headers = {
            "x-budget-router-model": decision_value.selected_model,
            "x-budget-router-predicted-success": (
                f"{decision_value.predicted_success[decision_value.selected_model]:.6f}"
            ),
            "x-budget-router-artifact": router.artifact_hash,
        }
        client = httpx.AsyncClient(transport=transport, timeout=None)
        if not bool(forwarded.get("stream", False)):
            try:
                upstream_response = await client.post(
                    upstream.chat_completions_url,
                    json=forwarded,
                    headers=headers,
                )
            finally:
                await client.aclose()
            content_type = upstream_response.headers.get(
                "content-type", "application/json"
            )
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                media_type=content_type.split(";", maxsplit=1)[0],
                headers=response_headers,
            )

        request_value = client.build_request(
            "POST",
            upstream.chat_completions_url,
            json=forwarded,
            headers=headers,
        )
        upstream_response = await client.send(request_value, stream=True)

        async def stream_bytes() -> AsyncIterator[bytes]:
            try:
                async for block in upstream_response.aiter_bytes():
                    yield block
            finally:
                await upstream_response.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_bytes(),
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get(
                "content-type", "text/event-stream"
            ).split(";", maxsplit=1)[0],
            headers=response_headers,
        )

    return app


def app_from_config(path: str | Path, *, transport: Any = None) -> Any:
    config_path = Path(path)
    router, upstreams, settings = load_proxy_config(config_path)
    return create_app(router, upstreams, transport=transport, **settings)
