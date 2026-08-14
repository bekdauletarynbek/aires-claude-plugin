"""The HTTP half of the MCP server.

Kept apart from the tool wiring so it can be tested without the MCP SDK or a
transport: everything interesting — auth, error shape, and how much text comes
back — lives here.

That last one is the design constraint people underestimate. A tool result is
pasted straight into the model's context, and this corpus holds PRDs running
to twenty thousand tokens. Returning one whole would evict the conversation
that asked for it, so long fields are cut with a marker saying what was cut
and how to get the rest.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from aires_mcp.client import AiresClient, AiresError, truncate

BASE = "https://arxiv.example.com"


def _client(**kwargs: object) -> AiresClient:
    return AiresClient(base_url=BASE, **kwargs)  # type: ignore[arg-type]


# ── truncation ─────────────────────────────────────────────────────────


def test_short_text_is_left_alone() -> None:
    assert truncate("тихий текст", limit=100) == "тихий текст"


def test_long_text_says_what_it_cut_and_how_to_get_it() -> None:
    body = "x" * 5000
    out = truncate(body, limit=100, hint="section='prd'")

    assert len(out) < 400
    assert out.startswith("x" * 100)
    assert "5000" in out, "the reader must know how much was withheld"
    assert "section='prd'" in out, "and how to ask for the rest"


def test_truncation_of_none_is_not_the_string_none() -> None:
    """A missing artefact reads as absent, not as the word None."""
    assert truncate(None, limit=10) == ""


# ── auth ───────────────────────────────────────────────────────────────


@respx.mock
async def test_basic_auth_is_sent_when_configured() -> None:
    """Production sits behind nginx Basic Auth; without this every call 401s."""
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"items": [], "query": "x"})
    )

    async with _client(basic_auth="user:secret") as client:
        await client.search("x")

    auth = route.calls[0].request.headers.get("authorization", "")
    assert auth.startswith("Basic ")


@respx.mock
async def test_no_auth_header_when_not_configured() -> None:
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"items": [], "query": "x"})
    )

    async with _client() as client:
        await client.search("x")

    assert "authorization" not in route.calls[0].request.headers


# ── errors ─────────────────────────────────────────────────────────────


@respx.mock
async def test_http_error_becomes_a_readable_message() -> None:
    """The model sees this string. A raw traceback teaches it nothing."""
    respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(502, text="<html>Bad Gateway</html>")
    )

    async with _client() as client:
        with pytest.raises(AiresError) as exc:
            await client.search("x")

    assert "502" in str(exc.value)


@respx.mock
async def test_api_error_envelope_is_unwrapped() -> None:
    """The API answers errors as {"error": {"code", "message"}} — surface the
    code, not the JSON."""
    respx.put(f"{BASE}/articles/9/prd").mock(
        return_value=httpx.Response(
            409,
            json={
                "error": {
                    "code": "ARTICLE_NOT_VECTORIZED",
                    "message": "Часть 2 must run first",
                    "details": {},
                }
            },
        )
    )

    async with _client() as client:
        with pytest.raises(AiresError) as exc:
            await client.submit_prd(9, "# PRD: x")

    assert "ARTICLE_NOT_VECTORIZED" in str(exc.value)


@respx.mock
async def test_connection_failure_names_the_host() -> None:
    """"Connection refused" alone leaves the user guessing which URL is wrong."""
    respx.post(f"{BASE}/search").mock(side_effect=httpx.ConnectError("refused"))

    async with _client() as client:
        with pytest.raises(AiresError) as exc:
            await client.search("x")

    assert BASE in str(exc.value)


# ── the calls ──────────────────────────────────────────────────────────


@respx.mock
async def test_search_passes_query_and_limit() -> None:
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "query": "fraud",
                "items": [
                    {
                        "id": 1,
                        "name_normalized": "2405_1",
                        "title": "Graph fraud",
                        "lang": "en",
                        "status": "completed",
                        "score": 0.91,
                        "matched_chunks": 3,
                        "excerpt": "passage",
                        "section_title": "Method",
                        "description": "описание",
                    }
                ],
            },
        )
    )

    async with _client() as client:
        hits, total = await client.search("fraud", limit=3)

    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent == {"query": "fraud", "limit": 3}
    assert hits[0]["title"] == "Graph fraud"
    assert total == 1, "a missing total falls back to what was returned"


@respx.mock
async def test_article_detail_is_requested_by_id() -> None:
    respx.get(f"{BASE}/articles/42").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": 42,
                "name_normalized": "2405_1",
                "original_title": "Paper",
                "status": "completed",
                "lang": "en",
                "progress_percent": 100,
                "stages": {"vibe_prd": False},
                "descriptions": {"description_ru": "описание", "description_en": None},
                "artifacts": {"prd_text": None, "analyse_text": "SIA"},
                "analyse_json": {"analysis": []},
                "demo_link": None,
                "has_pdf": True,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
    )

    async with _client() as client:
        article = await client.get_article(42)

    assert article["original_title"] == "Paper"


@respx.mock
async def test_company_articles_send_the_relevance_floor() -> None:
    route = respx.get(f"{BASE}/articles").mock(
        return_value=httpx.Response(
            200, json={"items": [], "total": 0, "limit": 10, "offset": 0}
        )
    )

    async with _client() as client:
        await client.company_articles("Freedom Bank", min_relevance=8, limit=10)

    url = route.calls[0].request.url
    assert url.params["company"] == "Freedom Bank"
    assert url.params["min_relevance"] == "8"


@respx.mock
async def test_submit_prd_sends_the_document_in_the_body() -> None:
    """Контракт PUT: документ уходит целиком, под ключом ``prd_markdown``."""
    route = respx.put(f"{BASE}/articles/7/prd").mock(
        return_value=httpx.Response(
            202,
            json={
                "accepted": True,
                "article_id": 7,
                "presentation_task_id": "t",
                "queued_at": "2026-01-01T00:00:00Z",
            },
        )
    )

    async with _client() as client:
        await client.submit_prd(7, "# PRD: Тест\n\nтело")

    import json as _json

    sent = _json.loads(route.calls[0].request.content)
    assert sent == {"prd_markdown": "# PRD: Тест\n\nтело"}


@respx.mock
async def test_ideas_sends_topic_company_and_limit() -> None:
    route = respx.post(f"{BASE}/ideas").mock(
        return_value=httpx.Response(
            200, json={"items": [], "topic": None, "company": None}
        )
    )

    async with _client() as client:
        await client.ideas(topic="ассистент", company="Freedom Bank", limit=4)

    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "topic": "ассистент",
        "company": "Freedom Bank",
        "limit": 4,
    }


@respx.mock
async def test_ideas_omits_empty_filters() -> None:
    """A bare "что можно сделать?" must not send topic="" and get zero hits."""
    route = respx.post(f"{BASE}/ideas").mock(
        return_value=httpx.Response(
            200, json={"items": [], "topic": None, "company": None}
        )
    )

    async with _client() as client:
        await client.ideas()

    import json

    assert json.loads(route.calls[0].request.content) == {"limit": 10}
