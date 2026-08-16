"""MCP server over the AIres research corpus.

Installed into Claude Code, this turns the corpus into something a person can
interrogate in the middle of their own work: *what has our library got on
graph fraud detection*, *which papers matter to Freedom Bank*, *build me a
deck on this one*.

Deliberately thin. Everything worth testing lives in :mod:`aires_mcp.client`
(HTTP, auth, error prose) and :mod:`aires_mcp.format` (what the model reads);
this module only declares the tools and hands their arguments over. The MCP
SDK is imported at call time so the package can be exercised without it.

Nothing here spends the project's LLM budget: the one expensive artefact, the
PRD, is written by the caller's own model. ``prd_brief`` hands out the locked
prompts; writing the finished document back is deliberately absent — the
Opus pass (~$1.19 per document, 86.5% of all spend) was deleted, not gated.
"""

from __future__ import annotations

import os
from typing import Any

from aires_mcp.client import AiresClient, AiresError
from aires_mcp.format import (
    format_article,
    format_companies,
    format_company_brief,
    format_coverage,
    format_ideas,
    format_prd_brief,
    format_search_hits,
    format_text_page,
)

__all__ = ["build_server", "main"]

ENV_URL = "AIRES_API_URL"
ENV_AUTH = "AIRES_BASIC_AUTH"


def _client_from_env() -> AiresClient:
    base = os.environ.get(ENV_URL, "").strip()
    if not base:
        # Refused loudly rather than defaulting to localhost: a silent default
        # produces "connection refused" and sends the user debugging the wrong
        # thing entirely.
        raise AiresError(
            f"{ENV_URL} не задан. Пропишите адрес API в конфиге MCP-сервера, "
            "например https://aires.ai-marketing.cloud/api"
        )
    return AiresClient(
        base_url=base, basic_auth=os.environ.get(ENV_AUTH) or None
    )


async def _call(fn: Any) -> str:
    """Run one tool body, turning any failure into text the model can use."""
    try:
        client = _client_from_env()
    except AiresError as exc:
        # Before the client exists — nothing to close, and closing in a shared
        # finally would raise NameError over the real message.
        return f"Ошибка конфигурации: {exc}"
    try:
        return await fn(client)
    except AiresError as exc:
        return f"Ошибка: {exc}"
    finally:
        await client.aclose()


def build_server() -> Any:
    """Assemble the FastMCP server and its tools."""
    from mcp.server.fastmcp import FastMCP

    # Instructions reach the client before any tool call. They exist for the
    # open-ended question — "что можно сделать?", "какие есть идеи" — which
    # matches no tool name and would otherwise be answered from the model's
    # own knowledge, silently ignoring the corpus the user is paying for.
    mcp = FastMCP(
        "aires",
        instructions=(
            "Корпус научных статей, разобранных под задачи Freedom "
            "Holding: по каждой статье есть оценка релевантности компаниям, "
            "конкретные возможности и риски.\n\n"
            "Когда собеседник спрашивает открыто — «что можно сделать?», "
            "«какие есть идеи», «что внедрить», «куда развиваться», «что "
            "интересного в исследованиях» — СНАЧАЛА вызовите `suggest_ideas`, "
            "и только потом отвечайте. Идеи должны опираться на статьи из "
            "базы и ссылаться на их номера, а не на общие знания модели. "
            "Если разговор про конкретную компанию — передайте её в "
            "`company`, про конкретную тему — в `topic`.\n\n"
            "PRD: когда просят PRD по статье — НАПИШИТЕ его сами. "
            "`prd_brief(article_id)` вернёт системный промпт и материалы; "
            "следуйте им дословно и отдайте документ человеку. Записи в "
            "корпус у инструментов нет — они только читают."
        ),
    )

    @mcp.tool()
    async def search_articles(query: str, limit: int = 10) -> str:
        """Найти статьи в корпусе по смыслу запроса.

        Семантический поиск по полным текстам — не по заголовкам. Возвращает
        статьи с фрагментом, который совпал, и id для остальных инструментов.
        Запрос лучше формулировать по-английски: корпус англоязычный.
        """
        return await _call(lambda c: _search(c, query, limit))

    @mcp.tool()
    async def suggest_ideas(
        topic: str = "",
        company: str = "",
        limit: int = 10,
        min_relevance: int = 0,
    ) -> str:
        """Что можно сделать: идеи из собранных статей, с номерами источников.

        Вызывайте на любой открытый вопрос — «что можно сделать?», «какие
        есть идеи», «что внедрить», «куда двигаться», «что полезного в
        исследованиях» — в том числе без темы и компании. Возвращает
        возможности и риски, которые аналитический этап уже вывел из статей,
        отсортированные по релевантности бизнесу. ``min_relevance`` (0–10)
        отсекает слабые совпадения: 6 — «стоит смотреть», 8 — только сильные.

        Это материал, а не готовый ответ: выберите сильнейшие идеи под
        контекст разговора, разверните их и сошлитесь на номера статей.
        У статьи бывают углы под несколько компаний — выдача укажет, где
        искать остальные.
        """
        return await _call(
            lambda c: _ideas(c, topic, company, limit, min_relevance)
        )

    @mcp.tool()
    async def get_article(
        article_id: int, section: str = "all", page: int = 1
    ) -> str:
        """Прочитать статью. Секции: all, summary, ipr, analysis, prd,
        presentation, conspect, full.

        Берите `all` — это ОДИН вызов со всеми готовыми артефактами; не
        вызывайте инструмент несколько раз по одной секции. `ipr` — тезисы с
        дословными цитатами из статьи (самый обоснованный артефакт).
        `full` — полный текст статьи постранично (page=1,2,…) — для вопросов
        про метод, формулы и детали, которых нет в пересказах. Для чтения
        нескольких статей сразу есть `get_articles`.
        """
        if section == "full":
            return await _call(lambda c: _article_text(c, article_id, page))
        return await _call(
            lambda c: _article(c, article_id, section)
        )

    @mcp.tool()
    async def get_articles(article_ids: list[int], section: str = "all") -> str:
        """Прочитать НЕСКОЛЬКО статей одним вызовом (до 5 за раз).

        Замена серии одиночных get_article: один круг вместо N. Статьи
        загружаются параллельно и отдаются подряд, каждая под своим номером.
        """
        return await _call(lambda c: _articles(c, article_ids, section))

    @mcp.tool()
    async def company_brief(company: str) -> str:
        """Стартовый бриф по компании ОДНИМ вызовом: покрытие корпуса,
        сильнейшие статьи и идеи.

        Вызывайте первым при вопросе про конкретную компанию — он заменяет
        цепочку corpus_coverage → company_articles → suggest_ideas и экономит
        3–4 круга. Дальше углубляйтесь через get_articles по номерам.
        """
        return await _call(lambda c: _brief(c, company))

    @mcp.tool()
    async def corpus_coverage() -> str:
        """Карта корпуса: сколько статей, по каким компаниям, где дыры.

        Вызывайте ПЕРЕД глубоким поиском по узкой теме: если по компании или
        домену статей единицы — это дыра корпуса, и надо расширять ленты, а
        не перебирать формулировки запроса.
        """
        return await _call(lambda c: _coverage(c))

    @mcp.tool()
    async def list_companies() -> str:
        """Компании, по которым размечен корпус, с числом релевантных статей."""
        return await _call(lambda c: _companies(c))

    @mcp.tool()
    async def company_articles(
        company: str, min_relevance: int = 8, limit: int = 20
    ) -> str:
        """Статьи, релевантные конкретной компании (оценка 0–10).

        min_relevance=8 оставляет только то, что действительно стоит читать.
        """
        return await _call(
            lambda c: _company_articles(c, company, min_relevance, limit)
        )

    @mcp.tool()
    async def prd_brief(article_id: int) -> str:
        """Материалы для написания PRD ВАМИ (бесплатно для проекта).

        Возвращает системный промпт (контракт документа) и пользовательский
        промпт с полями статьи — те же залоченные шаблоны, что у серверного
        пути. Напишите PRD, строго следуя системному промпту (от `# PRD: …`
        до версии в подвале), и отдайте документ человеку — сохранить его
        в корпус можно из кабинета.
        """
        return await _call(lambda c: _prd_brief(c, article_id))

    @mcp.tool()
    async def get_deck_link(article_id: int) -> str:
        """Ссылка на .pptx — презентация собирается из готового плана."""
        return await _call(lambda c: _deck(c, article_id))

    return mcp


# Tool bodies live outside the closures so they read as ordinary functions.


async def _search(client: AiresClient, query: str, limit: int) -> str:
    hits, total = await client.search(query, limit=limit)
    return format_search_hits(hits, query=query, total=total)


async def _ideas(
    client: AiresClient,
    topic: str,
    company: str,
    limit: int,
    min_relevance: int = 0,
) -> str:
    seeds, total = await client.ideas(
        topic=topic or None,
        company=company or None,
        limit=limit,
        min_relevance=min_relevance or None,
    )
    return format_ideas(
        seeds, topic=topic or None, company=company or None, total=total
    )


async def _coverage(client: AiresClient) -> str:
    return format_coverage(await client.corpus_coverage())


async def _article(client: AiresClient, article_id: int, section: str) -> str:
    article = await client.get_article(article_id)
    return format_article(article, section=section)


async def _article_text(client: AiresClient, article_id: int, page: int) -> str:
    return format_text_page(await client.article_text(article_id, page=page))


async def _articles(
    client: AiresClient, article_ids: list[int], section: str
) -> str:
    import asyncio

    # Кап на пять: пакет экономит круги, но токены ответов складываются, и
    # десять статей за раз вытеснили бы из контекста сам разговор.
    ids = list(dict.fromkeys(int(i) for i in article_ids))[:5]
    if not ids:
        return "Передайте хотя бы один id статьи."
    articles = await asyncio.gather(
        *(client.get_article(i) for i in ids), return_exceptions=True
    )
    parts: list[str] = []
    for article_id, article in zip(ids, articles, strict=True):
        if isinstance(article, BaseException):
            parts.append(f"#{article_id}: не удалось загрузить — {article}")
        else:
            parts.append(format_article(article, section=section))
    note = ""
    if len(article_ids) > len(ids):
        note = (
            f"\n(Запрошено {len(article_ids)}, показаны первые {len(ids)} — "
            "повторите вызов с остальными id.)"
        )
    return "\n\n---\n\n".join(parts) + note


async def _brief(client: AiresClient, company: str) -> str:
    import asyncio

    coverage, articles, (seeds, total) = await asyncio.gather(
        client.corpus_coverage(),
        client.company_articles(company, min_relevance=6, limit=10),
        client.ideas(company=company, limit=8),
    )
    return format_company_brief(
        company, coverage, articles, seeds, ideas_total=total
    )


async def _companies(client: AiresClient) -> str:
    return format_companies(await client.list_companies())


async def _company_articles(
    client: AiresClient, company: str, min_relevance: int, limit: int
) -> str:
    items = await client.company_articles(
        company, min_relevance=min_relevance, limit=limit
    )
    if not items:
        return (
            f"Для «{company}» статей с релевантностью {min_relevance}+ нет. "
            "Попробуйте порог пониже или `list_companies()`."
        )
    hits = [
        {
            "id": item.get("id"),
            "title": item.get("original_title"),
            "name_normalized": item.get("name_normalized"),
            "score": 0.0,
            "excerpt": "",
            "description": None,
        }
        for item in items
    ]
    body = format_search_hits(hits, query=f"{company}, релевантность {min_relevance}+")
    return body.replace(" (score 0.00)", "")


async def _prd_brief(client: AiresClient, article_id: int) -> str:
    brief = await client.prd_brief(article_id)
    return format_prd_brief(brief)


async def _deck(client: AiresClient, article_id: int) -> str:
    article = await client.get_article(article_id)
    stages = article.get("stages") or {}
    if not stages.get("presentation_done"):
        return (
            f"Для статьи #{article_id} плана презентации ещё нет. "
            f"Сначала PRD: `prd_brief({article_id})` → напишите документ → "
            "загрузите его в кабинете корпуса; план презентации сервер соберёт сам."
        )
    return (
        f"Презентация по статье #{article_id}: {client.deck_url(article_id)}\n"
        "Файл .pptx собирается на лету из плана, со слайдами и заметками "
        "докладчика."
    )


def main() -> None:
    """Console entry point (``aires-mcp``) — stdio transport."""
    build_server().run()


if __name__ == "__main__":
    main()
