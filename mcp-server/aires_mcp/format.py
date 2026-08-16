"""Rendering API payloads for a model to read.

Every function here returns markdown, not JSON. A tool that hands back the raw
payload makes the model parse and re-summarise it on each call, paying context
for structure nobody reads; markdown that states the answer costs a fraction
and is what ends up quoted to the user anyway.

Two things are non-negotiable in every result:

* **the article id**, because every other tool takes one — a result without it
  is a dead end;
* **what is missing and how to get it**. "No PRD yet" is the normal state, not
  an error: the document is written by the caller's model via ``prd_brief`` →
  the cabinet, so the result must name that path rather than let the model
  conclude the paper is unanalysed.
"""

from __future__ import annotations

from typing import Any

from aires_mcp.client import truncate

__all__ = [
    "SECTIONS",
    "format_article",
    "format_ideas",
    "format_companies",
    "format_prd_brief",
    "format_prd_saved",
    "format_search_hits",
]

# Caps are per section and generous — the point is to stop a runaway artefact
# from evicting the conversation, not to keep answers terse.
_SUMMARY_LIMIT = 2_000
_ANALYSIS_LIMIT = 12_000
_PRD_LIMIT = 16_000
_DECK_LIMIT = 12_000
_CONSPECT_LIMIT = 16_000

SECTIONS = (
    "summary",
    "ipr",
    "analysis",
    "prd",
    "presentation",
    "conspect",
    "all",
)

#: "full" обрабатывается отдельно от SECTIONS: ему нужен второй запрос
#: (постраничный текст), а не поля уже загруженной карточки.


def _title_of(item: dict[str, Any]) -> str:
    """Titles come from OCR and are sometimes absent — fall back to the id."""
    return str(
        item.get("title")
        or item.get("original_title")
        or item.get("name_normalized")
        or "без названия"
    )


def format_search_hits(
    hits: list[dict[str, Any]], *, query: str, total: int | None = None
) -> str:
    if not hits:
        return (
            f'По запросу «{query}» в корпусе ничего не найдено. '
            "Попробуйте переформулировать или искать по-английски — "
            "статьи в основном англоязычные."
        )

    header = f'Статьи по запросу «{query}» ({len(hits)}'
    if total is not None and total > len(hits):
        header += f" из {total} найденных"
    lines = [header + "):", ""]
    for hit in hits:
        score = float(hit.get("score", 0.0))
        lines.append(f"**#{hit.get('id')} — {_title_of(hit)}** (score {score:.2f})")
        meta = str(hit.get("name_normalized", ""))
        if hit.get("section_title"):
            meta += f" · раздел «{hit['section_title']}»"
        if hit.get("matched_chunks"):
            meta += f" · совпадений: {hit['matched_chunks']}"
        lines.append(f"  {meta}")
        excerpt = truncate(str(hit.get("excerpt") or ""), limit=400)
        if excerpt:
            lines.append(f"  > {excerpt}")
        if hit.get("description"):
            lines.append(f"  {truncate(str(hit['description']), limit=300)}")
        lines.append("")

    lines.append(
        "Дальше: `get_article(article_id, section=...)` — "
        "summary / analysis / prd / presentation / conspect."
    )
    if total is not None and total > len(hits):
        lines.append(
            f"Показаны не все: всего подходящих — {total}. "
            "Нужно больше — повторите с бо́льшим `limit`."
        )
    return "\n".join(lines)


def _format_analysis_json(payload: Any) -> str:
    """The SIA verdict per company, which is the part people act on."""
    if not isinstance(payload, dict):
        return ""
    entries = payload.get("analysis")
    if not isinstance(entries, list) or not entries:
        return ""
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("company_name", "?")
        score = entry.get("relevance_score")
        # The key is `reasoning` — checked against production payloads. The
        # first version guessed `rationale`/`summary` and silently rendered
        # every company without its explanation.
        note = entry.get("reasoning") or ""
        row = f"- **{name}** — релевантность {score if score is not None else '—'}"
        if note:
            row += f": {truncate(str(note), limit=400)}"
        rows.append(row)
        for opportunity in entry.get("opportunities") or []:
            rows.append(f"    - Можно: {truncate(str(opportunity), limit=300)}")
        for threat in entry.get("threats") or []:
            rows.append(f"    - Риск: {truncate(str(threat), limit=250)}")
    return "\n".join(rows)


def format_article(article: dict[str, Any], *, section: str = "summary") -> str:
    if section not in SECTIONS:
        return (
            f"Неизвестный раздел «{section}». Доступны: {', '.join(SECTIONS)}."
        )

    article_id = article.get("id")
    artifacts = article.get("artifacts") or {}
    stages = article.get("stages") or {}
    descriptions = article.get("descriptions") or {}

    head = [f"**#{article_id} — {_title_of(article)}**"]
    meta = f"{article.get('name_normalized', '')} · статус {article.get('status', '?')}"
    if article.get("progress_percent") is not None:
        meta += f" · готовность {article['progress_percent']}%"
    # Откуда текст — вопрос доверия к формулам: html = авторский TeX
    # дословно, ocr = распознавание GLM-OCR (возможны неточности в формулах).
    source = article.get("text_source")
    if source:
        label = {
            "html": "HTML (формулы дословно)",
            "src": "исходник+pandoc",
            "ocr": "GLM-OCR (распознавание)",
        }.get(str(source), str(source))
        meta += f" · текст: {label}"
    head.append(meta)
    head.append("")

    want = SECTIONS[:-1] if section == "all" else (section,)
    body: list[str] = []

    for part in want:
        if part == "summary":
            text = descriptions.get("description_ru") or descriptions.get(
                "description_en"
            )
            body.append("## Кратко")
            body.append(
                truncate(text, limit=_SUMMARY_LIMIT)
                or "Описание ещё не сгенерировано."
            )
            # Однострочное описание — это анонс, а не содержание. Рядом в
            # артефактах лежит полноценное IPR-резюме (несколько тысяч
            # знаков); не показывать его здесь означало ровно тот фидбэк,
            # с которого начался этот фикс: «summary бесполезно».
            ipr = artifacts.get("summary") or artifacts.get("summary_eng")
            if ipr:
                body.append("")
                body.append("## Резюме статьи (IPR)")
                body.append(
                    truncate(
                        ipr,
                        limit=_ANALYSIS_LIMIT,
                        hint=f"get_article({article_id}, section='conspect')",
                    )
                )
        elif part == "ipr":
            # IPR — самый обоснованный артефакт: тезисы с дословными цитатами
            # из статьи (проверено: 38/39 цитат посимвольно точны). Его части
            # лежат в БД по отдельности; собираем в один блок.
            body.append("## IPR — тезисы с цитатами из статьи")
            ipr_summary = artifacts.get("summary") or artifacts.get("summary_eng")
            if ipr_summary:
                body.append(truncate(ipr_summary, limit=_ANALYSIS_LIMIT))
            citations = artifacts.get("citation_ru") or artifacts.get("citation_en")
            if citations:
                body.append("")
                body.append("### Цитаты-обоснования")
                body.append(truncate(citations, limit=_ANALYSIS_LIMIT))
            if not ipr_summary and not citations:
                body.append("IPR ещё не готов для этой статьи.")
        elif part == "analysis":
            body.append("## Анализ по компаниям (SIA)")
            scores = _format_analysis_json(article.get("analyse_json"))
            body.append(scores or "Оценок по компаниям пока нет.")
            report = truncate(
                artifacts.get("analyse_text"),
                limit=_ANALYSIS_LIMIT,
                hint=f"get_article({article_id}, section='analysis')",
            )
            if report:
                body.append("")
                body.append(report)
        elif part == "prd":
            body.append("## PRD")
            body.append(
                truncate(
                    artifacts.get("prd_text"),
                    limit=_PRD_LIMIT,
                    hint=f"get_article({article_id}, section='prd')",
                )
                or _missing_prd_note(article_id, stages)
            )
        elif part == "presentation":
            body.append("## План презентации")
            body.append(
                truncate(
                    artifacts.get("presentation_text"),
                    limit=_DECK_LIMIT,
                    hint=f"get_article({article_id}, section='presentation')",
                )
                or _missing_prd_note(article_id, stages)
            )
        elif part == "conspect":
            body.append("## Конспект")
            body.append(
                truncate(
                    artifacts.get("conspect_text_ru")
                    or artifacts.get("conspect_text"),
                    limit=_CONSPECT_LIMIT,
                    hint=f"get_article({article_id}, section='conspect')",
                )
                or "Конспект ещё не готов."
            )
        body.append("")

    return "\n".join(head + body).rstrip() + "\n"


def _missing_prd_note(article_id: Any, stages: dict[str, Any]) -> str:
    """Absence here is the designed state, so it must not read as a failure."""
    if stages.get("vibe_prd"):
        return (
            "PRD отмечен как готовый, но текста нет — вероятно, сбой выгрузки. "
            "Перезаписать документ можно из кабинета корпуса."
        )
    return (
        "PRD ещё не писали: документ создаёте ВЫ, а не сервер. "
        f"`prd_brief({article_id})` вернёт промпт и материалы; готовый текст "
        "загружают в корпус из кабинета — у инструментов доступа на запись нет."
    )


def format_companies(companies: list[dict[str, Any]]) -> str:
    if not companies:
        return "Список компаний пуст — SIA-анализ ещё не дал оценок."

    # Sorted by how much is actually worth reading, not by raw article count:
    # a company with 30 mentions and none above the bar is noise.
    ordered = sorted(
        companies,
        key=lambda c: (c.get("high_relevance") or 0, c.get("articles") or 0),
        reverse=True,
    )
    lines = ["Компании в корпусе (по числу сильных совпадений):", ""]
    for company in ordered:
        name = company.get("name", "?")
        line = f"- **{name}**"
        if company.get("category"):
            line += f" ({company['category']})"
        line += (
            f" — статей {company.get('articles', 0)}, "
            f"из них релевантных 8+: {company.get('high_relevance', 0)}"
        )
        if company.get("avg_relevance") is not None:
            line += f", средняя {company['avg_relevance']}"
        lines.append(line)
    lines.append("")
    lines.append(
        "Дальше: `company_articles(company, min_relevance=8)`."
    )
    return "\n".join(lines)


def format_ideas(
    seeds: list[dict[str, Any]],
    *,
    topic: str | None,
    company: str | None,
    total: int | None = None,
) -> str:
    """Material for a proposal, plus instructions on what to do with it.

    The seeds are the analyst's own words. Handing them over verbatim and
    stopping would leave the user reading a database dump, so the block ends
    by telling the model to compose — pick a few, argue them, and offer the
    PRD. Without that nudge a model tends to paraphrase the list back.
    """
    scope = ""
    if topic:
        scope += f" по теме «{topic}»"
    if company:
        scope += f" для {company}"

    if not seeds:
        return (
            f"В корпусе нет разобранных статей{scope}. "
            "Попробуйте другую формулировку, снимите фильтр по компании или "
            "посмотрите `list_companies()` — возможно, анализ ещё идёт."
        )

    counted = f"{len(seeds)}"
    if total is not None and total > len(seeds):
        counted += f" из {total}"
    lines = [
        f"Что можно сделать{scope} — из статей, которые мы собрали и разобрали "
        f"({counted} направлений):",
        "",
    ]

    # Action first, paper second. The question was "what can we do", so a list
    # led by paper titles makes the reader translate every line before they can
    # judge it. The source stays one line below, because without the id nothing
    # can be checked or turned into a PRD.
    number = 0
    for seed in seeds:
        source = (
            f"источник: #{seed.get('article_id')} "
            f"«{seed.get('title')}», релевантность "
            f"{seed.get('relevance')}/10"
        )
        # Правило «одна статья — одна идея» бережёт разнообразие выдачи, но
        # молча прячет кросс-компанийные находки (MLOps-аудит из #265 нашли
        # руками именно в analysis). Счётчик делает срез честным.
        other = int(seed.get("other_angles") or 0)
        if other:
            source += (
                f" · ещё {other} угол(а) у других компаний — "
                f"get_article({seed.get('article_id')}, section='analysis')"
            )
        for opportunity in seed.get("opportunities") or []:
            number += 1
            lines.append(
                f"**{number}. {truncate(str(opportunity), limit=400)}** "
                f"— {seed.get('company')}"
            )
            if seed.get("reasoning"):
                lines.append(
                    f"   Зачем: {truncate(str(seed['reasoning']), limit=500)}"
                )
            for threat in (seed.get("threats") or [])[:2]:
                lines.append(f"   Риск: {truncate(str(threat), limit=300)}")
            lines.append(f"   {source}")
            lines.append("")

    lines.append(
        "Это сырьё, а не ответ. Выберите 3–5 самых сильных под задачу "
        "собеседника, соберите из них внятные предложения (что делаем, кому, "
        "за счёт чего выигрываем, чем рискуем) и обязательно сошлитесь на "
        "номера статей. За подробностями — `get_article(id, section='analysis')`; "
        "если идею берут в работу — `prd_brief(id)`, и ТЗ пишете вы."
    )
    if total is not None and total > len(seeds):
        lines.append(
            f"Это лишь верх списка: подходящих статей — {total}. "
            "Если нужного не видно, повторите с бо́льшим `limit` "
            "или задайте `topic`/`company`."
        )
    return "\n".join(lines)


def format_prd_brief(brief: dict[str, Any]) -> str:
    """Бриф для клиентской генерации: промпты отдаются ДОСЛОВНО.

    Никаких сокращений: системный промпт — контракт документа, урезать его
    значит получить PRD не по шаблону. Материалы статьи и так уже сжаты
    IPR-этапом до тезисов и цитат.
    """
    article_id = brief.get("article_id")
    parts = [
        f"# Бриф PRD для статьи #{article_id}",
        "Напишите PRD сами, строго следуя системному промпту ниже, затем "
        "отдайте документ человеку — загрузить его в корпус можно из кабинета.",
    ]
    if brief.get("already_has_prd"):
        parts.append(
            "⚠ У статьи уже есть PRD — сохранение перезапишет его. "
            f"Посмотреть текущий: `get_article({article_id}, section='prd')`."
        )
    parts.append(
        "## Системный промпт (контракт документа)\n" + str(brief.get("system", ""))
    )
    parts.append(
        "## Пользовательский промпт (материалы статьи)\n" + str(brief.get("user", ""))
    )
    return "\n\n".join(parts)


def format_prd_saved(result: dict[str, Any], *, article_id: int) -> str:
    return (
        f"PRD статьи #{article_id} сохранён. Презентацию сервер строит сам "
        "в фоне (несколько минут) — ссылка на деку появится в "
        f"`get_deck_link({article_id})`."
    )


def format_coverage(data: dict[str, Any]) -> str:
    """Карта корпуса: где густо, где пусто.

    Существует, чтобы слепые зоны были видны ДО работы: аналитик по travel
    сначала поставил поиску 4/10 и лишь потом узнал, что travel-статей в
    корпусе почти нет.
    """
    lines = [
        f"**Корпус: {data.get('total_articles', 0)} статей, "
        f"разобрано аналитикой {data.get('analysed', 0)}.**"
    ]
    sources = data.get("by_text_source") or {}
    if sources:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(sources.items()))
        lines.append(f"Источник текста — {parts}.")
    if data.get("oldest"):
        lines.append(f"Даты поступления: {data['oldest']} … {data.get('newest')}.")
    lines.append("")
    lines.append("| Компания | статей | оценка ≥6 | оценка ≥8 |")
    lines.append("|---|---|---|---|")
    for row in data.get("companies") or []:
        lines.append(
            f"| {row.get('company')} | {row.get('articles')} "
            f"| {row.get('ge6')} | {row.get('ge8')} |"
        )
    lines.append("")
    lines.append(
        "Мало статей у нужной компании — значит, дыра в корпусе, а не в "
        "поиске: сначала расширяются RSS-ленты и ключевые слова "
        "(`/config/feeds`), и только потом имеет смысл судить о качестве "
        "`search_articles`."
    )
    return "\n".join(lines)


def format_text_page(data: dict[str, Any]) -> str:
    """Страница полного текста — для «докопаться до метода»."""
    pages = int(data.get("pages") or 0)
    page = int(data.get("page") or 1)
    article_id = data.get("article_id")
    if pages == 0:
        return f"У статьи #{article_id} нет извлечённого текста."
    text = str(data.get("text") or "")
    if not text:
        return (
            f"У статьи #{article_id} страниц {pages}, страницы {page} нет. "
            f"Запросите page от 1 до {pages}."
        )
    head = (
        f"**#{article_id} — полный текст, страница {page} из {pages}** "
        f"(всего {data.get('total_chars', '?')} символов)\n"
    )
    tail = ""
    if page < pages:
        tail = (
            f"\n\n→ продолжение: `get_article({article_id}, "
            f"section='full', page={page + 1})`"
        )
    return head + "\n" + text + tail


def format_company_brief(
    company: str,
    coverage: dict[str, Any],
    articles: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    ideas_total: int,
) -> str:
    """Один ответ вместо стартовой цепочки из 4–5 вызовов.

    Наблюдение с реального использования: каждая сессия по компании начинается
    одинаково — покрытие, топ статей, идеи. Каждый круг — это токены ответа в
    контексте и новый шаг рассуждения; композит складывает всё в один.
    """
    row = next(
        (c for c in coverage.get("companies") or [] if c.get("company") == company),
        None,
    )
    lines = [f"# {company} — бриф по корпусу", ""]
    if row:
        lines.append(
            f"Статей: {row.get('articles')} "
            f"(релевантность ≥6: {row.get('ge6')}, ≥8: {row.get('ge8')}) "
            f"из {coverage.get('total_articles')} в корпусе."
        )
    else:
        lines.append(
            f"В корпусе нет статей, размеченных на «{company}». "
            "Проверьте имя через `list_companies()`."
        )
    lines.append("")

    if articles:
        lines.append("## Сильнейшие статьи")
        for item in articles[:10]:
            lines.append(
                f"- #{item.get('id')} «{item.get('original_title') or item.get('name_normalized')}»"
            )
        lines.append("")

    if seeds:
        lines.append(f"## Идеи ({len(seeds)} из {ideas_total})")
        number = 0
        for seed in seeds:
            for opportunity in (seed.get("opportunities") or [])[:2]:
                number += 1
                lines.append(
                    f"{number}. {truncate(str(opportunity), limit=300)} "
                    f"(#{seed.get('article_id')}, {seed.get('relevance')}/10)"
                )
        lines.append("")

    lines.append(
        "Дальше за один вызов: `get_articles(ids=[...], section='all')` по "
        "выбранным номерам, либо `get_article(id, section='full')` для "
        "полного текста."
    )
    return "\n".join(lines)
