"""HTTP 小工具：复用连接 + 对瞬时错误做有限重试。

embedding / rerank 以前每次请求都 `httpx.post`，等于每次新建 TCP/TLS 连接；
ingest 的 89 次 batch 请求会付出明显的握手成本。同时 429/5xx/超时没有重试，
一次抖动就会把整个 search 或 ingest 打断。
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger("rag_mcp.http")

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
) -> httpx.Response:
    """POST JSON；仅对瞬时失败重试，4xx（除 429）立即抛出。"""
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code in RETRY_STATUS and attempt + 1 < MAX_ATTEMPTS:
                logger.warning(
                    "transient HTTP %s from %s (attempt %d/%d)",
                    response.status_code,
                    url,
                    attempt + 1,
                    MAX_ATTEMPTS,
                )
                time.sleep(0.2 * (2**attempt))
                continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt + 1 >= MAX_ATTEMPTS:
                raise
            logger.warning(
                "transient HTTP error from %s (attempt %d/%d): %s",
                url,
                attempt + 1,
                MAX_ATTEMPTS,
                type(exc).__name__,
            )
            time.sleep(0.2 * (2**attempt))
    raise last_error or RuntimeError("unreachable")
