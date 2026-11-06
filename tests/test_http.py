"""HTTP 客户端复用/重试与 embedding 归一化。"""

from __future__ import annotations

import httpx
import pytest

from rag_mcp.embedding import OpenAICompatEmbedder
from rag_mcp.http import post_json


def test_post_json_retries_transient_status():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = post_json(client, "http://test/v1", headers={}, payload={})
    finally:
        client.close()

    assert response.status_code == 200
    assert calls == 2


def test_post_json_does_not_retry_client_error():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            post_json(client, "http://test/v1", headers={}, payload={})
    finally:
        client.close()

    assert calls == 1


def test_embedding_uses_pooled_client_and_normalizes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [3.0, 4.0]},
                    {"index": 1, "embedding": [0.0, 0.0]},
                ]
            },
        )

    embedder = OpenAICompatEmbedder(base_url="http://test", model="m")
    embedder._client.close()
    embedder._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        vectors = embedder.embed(["a", "b"])
    finally:
        embedder.close()

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == [0.0, 0.0]
