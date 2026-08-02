from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from budget_router.openai_proxy import OpenAIUpstream, create_app
from budget_router.semantic import (
    ModelCard,
    SemanticOutcome,
    SemanticRouter,
    train_semantic_router,
)


def _router() -> SemanticRouter:
    cards = [
        ModelCard("cheap", provider="a", expected_cost_usd=0.01),
        ModelCard("strong", provider="b", expected_cost_usd=0.20),
    ]
    rows = []
    for split, count in (("train", 10), ("calibration", 6)):
        for index in range(count):
            text = "simple" if index % 2 == 0 else "hard"
            request_id = f"{split}-{index}"
            rows.extend(
                [
                    SemanticOutcome(
                        request_id,
                        text,
                        "cheap",
                        index % 2 == 0,
                        split,
                        0.01,
                    ),
                    SemanticOutcome(request_id, text, "strong", True, split, 0.20),
                ]
            )
    return SemanticRouter(train_semantic_router(rows, cards, dimension=128))


def test_proxy_routes_and_forwards_openai_request(monkeypatch) -> None:
    monkeypatch.setenv("CHEAP_TEST_KEY", "secret-a")
    monkeypatch.setenv("STRONG_TEST_KEY", "secret-b")
    seen: list[dict] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Bearer secret-")
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [],
            },
        )

    app = create_app(
        _router(),
        {
            "cheap": OpenAIUpstream(
                "cheap",
                "https://a.example/v1",
                "CHEAP_TEST_KEY",
                "provider/cheap",
            ),
            "strong": OpenAIUpstream(
                "strong",
                "https://b.example/v1",
                "STRONG_TEST_KEY",
                "provider/strong",
            ),
        },
        transport=httpx.MockTransport(upstream),
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "simple"}],
            "router": {"mode": "cost", "max_quality_drop": 0.3},
        },
    )

    assert response.status_code == 200
    assert response.headers["x-budget-router-model"] in {"cheap", "strong"}
    assert seen[0]["model"].startswith("provider/")
    assert "router" not in seen[0]


def test_proxy_decision_endpoint_never_needs_provider_key() -> None:
    app = create_app(
        _router(),
        {
            "cheap": OpenAIUpstream("cheap", "https://a.example", "UNSET_A"),
            "strong": OpenAIUpstream("strong", "https://b.example", "UNSET_B"),
        },
    )
    response = TestClient(app).post(
        "/v1/router/decision",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "hard"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["selected_model"] in {"cheap", "strong"}


def test_proxy_rejects_literal_secret_headers() -> None:
    try:
        OpenAIUpstream(
            "bad",
            "https://example.test",
            "KEY_ENV",
            headers={"Authorization": "Bearer literal"},
        )
    except ValueError as exc:
        assert "api_key_env" in str(exc)
    else:
        raise AssertionError("literal secret header was accepted")
