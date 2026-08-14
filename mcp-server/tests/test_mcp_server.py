"""Tool bodies and the failure paths around them.

The tools themselves are one-liners over :mod:`aires_mcp.client` and
:mod:`aires_mcp.format`, so what is worth pinning is the wiring between them —
the part that a passing type-check will not catch: a tool that reports success
on an unconfigured server, or one that hands out a link to a deck that was
never generated.

Failures return text rather than raising. An exception crossing the MCP
boundary reaches the user as a transport error with no advice in it; a
sentence explaining what to fix is worth more than a stack trace.
"""

from __future__ import annotations

from typing import Any

import pytest

from aires_mcp import server
from aires_mcp.client import AiresError


class _FakeClient:
    """Stands in for AiresClient; records what the tool asked for."""

    def __init__(self, **canned: Any) -> None:
        self.canned = canned
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    def _record(self, name: str, *args: Any) -> Any:
        self.calls.append((name, args))
        value = self.canned.get(name)
        if isinstance(value, Exception):
            raise value
        return value

    async def search(self, query: str, *, limit: int = 10) -> Any:
        # Mirrors the real client: hits travel with the total, so a capped
        # list can announce itself as capped.
        return self._record("search", query, limit) or [], 0

    async def ideas(
        self,
        *,
        topic: str | None = None,
        company: str | None = None,
        min_relevance: int | None = None,
        limit: int = 10,
    ) -> Any:
        return self._record("ideas", topic, company, limit) or [], 0

    async def get_article(self, article_id: int) -> Any:
        return self._record("get_article", article_id)

    async def list_companies(self) -> Any:
        return self._record("list_companies")

    async def company_articles(
        self, company: str, *, min_relevance: int | None = None, limit: int = 10
    ) -> Any:
        return self._record("company_articles", company, min_relevance, limit)

    async def prd_brief(self, article_id: int) -> Any:
        return self._record("prd_brief", article_id)

    async def submit_prd(self, article_id: int, prd_markdown: str) -> Any:
        return self._record("submit_prd", article_id, prd_markdown)

    def deck_url(self, article_id: int) -> str:
        return f"https://example.test/articles/{article_id}/export.pptx"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Install a fake client and hand it back for assertions."""

    def _install(**canned: Any) -> _FakeClient:
        client = _FakeClient(**canned)
        monkeypatch.setattr(server, "_client_from_env", lambda: client)
        return client

    return _install


# ── configuration ──────────────────────────────────────────────────────


async def test_missing_api_url_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset URL must not surface as a transport crash."""
    monkeypatch.delenv(server.ENV_URL, raising=False)

    out = await server._call(lambda c: server._companies(c))

    assert server.ENV_URL in out
    assert "Ошибка конфигурации" in out


async def test_api_errors_come_back_as_text(fake) -> None:  # type: ignore[no-untyped-def]
    fake(search=AiresError("HTTP 502 — Bad Gateway"))

    out = await server._call(lambda c: server._search(c, "x", 5))

    assert "502" in out
    assert out.startswith("Ошибка")


async def test_the_client_is_closed_even_when_the_call_fails(fake) -> None:  # type: ignore[no-untyped-def]
    client = fake(search=AiresError("boom"))

    await server._call(lambda c: server._search(c, "x", 5))

    assert client.closed is True, "a leaked connection per failed tool call"


# ── tool bodies ────────────────────────────────────────────────────────


async def test_search_passes_the_limit_through(fake) -> None:  # type: ignore[no-untyped-def]
    client = fake(search=[])

    await server._search(client, "fraud", 3)  # type: ignore[arg-type]

    assert client.calls == [("search", ("fraud", 3))]


async def test_company_articles_reports_an_empty_shortlist(fake) -> None:  # type: ignore[no-untyped-def]
    client = fake(company_articles=[])

    out = await server._company_articles(client, "Freedom Bank", 8, 10)  # type: ignore[arg-type]

    assert "Freedom Bank" in out
    assert "list_companies" in out, "give the user somewhere to go"


async def test_company_articles_lists_ids(fake) -> None:  # type: ignore[no-untyped-def]
    client = fake(
        company_articles=[
            {"id": 7, "original_title": "Paper", "name_normalized": "2405_7"}
        ]
    )

    out = await server._company_articles(client, "Freedom Bank", 8, 10)  # type: ignore[arg-type]

    assert "#7" in out
    assert "score" not in out, "a relevance-filtered list has no search score"


async def test_deck_link_refuses_when_no_plan_exists(fake) -> None:  # type: ignore[no-untyped-def]
    """The export would return an error file; say what to run instead."""
    client = fake(get_article={"id": 5, "stages": {"presentation_done": False}})

    out = await server._deck(client, 5)  # type: ignore[arg-type]

    assert "prd_brief(5)" in out
    assert "export.pptx" not in out


async def test_deck_link_is_given_once_the_plan_is_there(fake) -> None:  # type: ignore[no-untyped-def]
    client = fake(get_article={"id": 5, "stages": {"presentation_done": True}})

    out = await server._deck(client, 5)  # type: ignore[arg-type]

    assert "export.pptx" in out


# ── registration ───────────────────────────────────────────────────────


def test_every_tool_is_registered_with_a_description() -> None:
    """A tool the model cannot tell apart from another is a tool it misuses."""
    pytest.importorskip("mcp", reason="MCP SDK is an optional extra")

    import asyncio

    tools = asyncio.run(server.build_server().list_tools())
    names = {t.name for t in tools}

    assert names == {
        "suggest_ideas",
        "search_articles",
        "get_article",
        "list_companies",
        "company_articles",
        "corpus_coverage",
        "get_articles",
        "company_brief",
        "prd_brief",
        "save_prd",
        "get_deck_link",
    }
    for tool in tools:
        assert tool.description and len(tool.description) > 40, tool.name


def test_no_tool_spends_the_project_budget() -> None:
    """Серверная генерация PRD удалена — её не должно быть ни в тулах, ни в
    обещаниях описаний: агент, увидевший «сервер сгенерирует», будет ждать
    документ, которого никто не напишет."""
    pytest.importorskip("mcp", reason="MCP SDK is an optional extra")

    import asyncio

    tools = asyncio.run(server.build_server().list_tools())
    names = {t.name for t in tools}
    assert "generate_prd" not in names
    assert {"prd_brief", "save_prd"} <= names

    brief = next(t for t in tools if t.name == "prd_brief")
    assert "$" not in (brief.description or ""), "платить не за что"


async def test_ideas_tool_passes_blank_filters_as_none(fake) -> None:  # type: ignore[no-untyped-def]
    """A bare "что можно сделать?" arrives as empty strings; sending those
    on as filters would search for nothing and answer nothing."""
    client = fake(ideas=[])

    await server._ideas(client, "", "", 6)  # type: ignore[arg-type]

    assert client.calls == [("ideas", (None, None, 6))]


def test_the_server_tells_the_client_when_to_reach_for_ideas() -> None:
    """The open question matches no tool name. Without instructions the model
    answers from its own knowledge and the corpus goes unused — which is the
    one failure this whole feature exists to prevent."""
    pytest.importorskip("mcp", reason="MCP SDK is an optional extra")

    instructions = server.build_server().instructions or ""

    assert "suggest_ideas" in instructions
    assert "что можно сделать" in instructions.lower()


def test_ideas_tool_advertises_the_open_question() -> None:
    pytest.importorskip("mcp", reason="MCP SDK is an optional extra")

    import asyncio

    tools = asyncio.run(server.build_server().list_tools())
    ideas = next(t for t in tools if t.name == "suggest_ideas")

    assert "что можно сделать" in (ideas.description or "").lower()


# ── пакетное чтение: экономия кругов не должна ломаться об одну статью ─────


class _BatchFakeClient:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def get_article(self, article_id: int) -> dict:
        self.calls.append(article_id)
        if article_id == 500:
            raise RuntimeError("boom")
        return {
            "id": article_id,
            "original_title": f"t{article_id}",
            "descriptions": {"description_ru": "кратко"},
            "artifacts": {},
            "stages": {},
        }


@pytest.mark.asyncio
async def test_batch_read_caps_dedups_and_survives_failures() -> None:
    """Пакет: максимум 5, дубли схлопнуты, сбой одной статьи не валит ответ.

    Смысл инструмента — один круг вместо N; если бы одна битая статья роняла
    весь пакет, модель вернулась бы к одиночным вызовам и выгода исчезла.
    """
    from aires_mcp.server import _articles

    client = _BatchFakeClient()
    out = await _articles(client, [1, 1, 2, 500, 3, 4, 5, 6], "summary")

    assert client.calls.count(1) == 1, "дубль не схлопнут"
    assert len(client.calls) == 5, "кап на пять не сработал"
    assert "#1" in out and "#2" in out
    assert "не удалось загрузить" in out and "boom" in out
    assert "повторите вызов с остальными id" in out


async def test_prd_brief_carries_prompts_verbatim(fake) -> None:  # type: ignore[no-untyped-def]
    """Промпты в брифе не режутся: системный промпт — контракт документа."""
    system = "СИСТЕМНЫЙ КОНТРАКТ " + "х" * 5000
    client = _FakeClient(
        prd_brief={
            "article_id": 7,
            "system": system,
            "user": "Материалы статьи: {json}",
            "already_has_prd": False,
        }
    )
    out = await server._prd_brief(client, 7)  # type: ignore[arg-type]
    assert system in out, "системный промпт обязан войти дословно"
    assert "save_prd(7" in out
    assert "перезапишет" not in out


async def test_prd_brief_warns_about_existing_prd(fake) -> None:  # type: ignore[no-untyped-def]
    client = _FakeClient(
        prd_brief={
            "article_id": 7,
            "system": "s",
            "user": "u",
            "already_has_prd": True,
        }
    )
    out = await server._prd_brief(client, 7)  # type: ignore[arg-type]
    assert "перезапишет" in out


async def test_save_prd_reports_presentation_in_background(fake) -> None:  # type: ignore[no-untyped-def]
    client = _FakeClient(
        submit_prd={
            "accepted": True,
            "article_id": 7,
            "presentation_task_id": "tid-1",
            "queued_at": "2026-08-13T00:00:00Z",
        }
    )
    out = await server._save_prd(client, 7, "# PRD: Тест\n...")  # type: ignore[arg-type]
    assert client.calls == [("submit_prd", (7, "# PRD: Тест\n..."))]
    assert "get_deck_link(7)" in out
