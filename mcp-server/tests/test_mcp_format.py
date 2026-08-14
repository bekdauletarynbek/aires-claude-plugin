"""Turning API JSON into what the model actually reads.

A tool that dumps raw JSON makes the model re-derive the same summary every
call and spends context doing it. These helpers return short markdown: the
identifier it needs for the next call, the fact that answers the question, and
nothing else.

The article id is the load-bearing part — every other tool takes one, so a
result that omits it is a dead end.
"""

from __future__ import annotations

from aires_mcp.format import (
    format_article,
    format_companies,
    format_search_hits,
)


def _hit(**over: object) -> dict[str, object]:
    base = {
        "id": 42,
        "name_normalized": "2405_15604",
        "title": "Graph-based fraud detection",
        "lang": "en",
        "status": "completed",
        "score": 0.9134,
        "matched_chunks": 3,
        "excerpt": "We propose a GNN over transaction graphs.",
        "section_title": "Method",
        "description": "Граф-обнаружение мошенничества.",
    }
    base.update(over)
    return base


def test_search_result_carries_the_id_needed_for_every_other_tool() -> None:
    out = format_search_hits([_hit()], query="fraud")

    assert "42" in out, "without the id the model cannot call anything else"
    assert "Graph-based fraud detection" in out
    assert "GNN over transaction graphs" in out, "the matched passage"
    assert "0.91" in out


def test_empty_search_says_so_instead_of_returning_nothing() -> None:
    """A blank result reads as a broken tool; say it worked and found none."""
    out = format_search_hits([], query="квантовые улитки")

    assert "квантовые улитки" in out
    assert "не найдено" in out.lower()


def test_a_hit_without_a_title_still_identifies_itself() -> None:
    """Titles come from OCR and are sometimes missing."""
    out = format_search_hits([_hit(title=None)], query="x")

    assert "2405_15604" in out


def _article(**over: object) -> dict[str, object]:
    base = {
        "id": 42,
        "name_normalized": "2405_15604",
        "original_title": "Graph-based fraud detection",
        "status": "completed",
        "lang": "en",
        "progress_percent": 100,
        "stages": {"vibe_prd": False, "presentation_done": False},
        "descriptions": {
            "description_ru": "Граф-обнаружение мошенничества.",
            "description_en": "Graph fraud detection.",
        },
        "artifacts": {
            "prd_text": None,
            "presentation_text": None,
            "analyse_text": "## Стратегический отчёт\nПодробности.",
            "conspect_text_ru": "Конспект",
        },
        "analyse_json": {
            "analysis": [
                {"company_name": "Freedom Bank", "relevance_score": 9},
                {"company_name": "Freedom Broker", "relevance_score": 4},
            ]
        },
        "demo_link": None,
        "has_pdf": True,
    }
    base.update(over)
    return base


def test_summary_section_stays_short() -> None:
    out = format_article(_article(), section="summary")

    assert "Граф-обнаружение" in out
    assert "Подробности." not in out, "the SIA report belongs to section=analysis"


def test_missing_prd_explains_how_to_get_one() -> None:
    """The whole point of on-demand: absence is normal and actionable."""
    out = format_article(_article(), section="prd")

    assert "prd_brief" in out, "tell the model the tool that fixes this"
    assert "42" in out


def test_existing_prd_is_returned_and_capped() -> None:
    out = format_article(
        _article(artifacts={"prd_text": "П" * 40_000}), section="prd"
    )

    assert "40000" in out, "say how much was withheld"
    assert len(out) < 20_000


def test_relevance_scores_come_through_for_the_analysis_section() -> None:
    out = format_article(_article(), section="analysis")

    assert "Freedom Bank" in out
    assert "9" in out


def test_unknown_section_lists_the_valid_ones() -> None:
    out = format_article(_article(), section="нечто")

    assert "summary" in out and "prd" in out


def test_companies_are_ranked_by_what_is_worth_reading() -> None:
    out = format_companies(
        [
            {
                "name": "Freedom Bank",
                "category": "bank",
                "articles": 30,
                "high_relevance": 12,
                "avg_relevance": 6.4,
            },
            {
                "name": "Freedom Broker",
                "category": None,
                "articles": 5,
                "high_relevance": 0,
                "avg_relevance": 2.0,
            },
        ]
    )

    assert out.index("Freedom Bank") < out.index("Freedom Broker")
    assert "12" in out


def _seed(**over: object) -> dict[str, object]:
    base = {
        "article_id": 14,
        "name_normalized": "2405_1",
        "title": "Sparse graph embeddings",
        "company": "Freedom AI",
        "relevance": 9,
        "reasoning": "Лёгкие модели снижают инфраструктурные затраты.",
        "opportunities": [
            "Локальные NLP-модели на устройствах пользователей",
            "Снижение расходов на GPU",
        ],
        "threats": ["Уступает LLM на больших корпусах"],
        "description": "Разреженные графовые эмбеддинги.",
    }
    base.update(over)
    return base


def test_ideas_lead_with_what_can_be_done() -> None:
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None)

    assert "Локальные NLP-модели" in out
    assert "#14" in out, "the id is how the model checks or drills in"
    assert "Freedom AI" in out
    assert "9" in out


def test_ideas_carry_the_risk_next_to_the_opportunity() -> None:
    """A pitch without the caveat is how a bad idea gets approved."""
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None)

    assert "Уступает LLM" in out


def test_ideas_tell_the_model_what_to_do_next() -> None:
    """Otherwise it lists seeds verbatim instead of composing a proposal."""
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None)

    assert "prd_brief" in out
    assert "get_article" in out


def test_empty_ideas_explain_why_rather_than_return_nothing() -> None:
    from aires_mcp.format import format_ideas

    out = format_ideas([], topic="квантовые улитки", company=None)

    assert "квантовые улитки" in out
    assert "не нашл" in out.lower() or "нет" in out.lower()


def test_ideas_say_how_many_more_there_are() -> None:
    """The failure this prevents: the model takes the default slice, sees no
    sign of a larger set, and answers as if the corpus held six papers."""
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None, total=34)

    assert "34" in out
    assert "limit" in out, "and it must know how to ask for the rest"


def test_ideas_do_not_nag_when_everything_is_shown() -> None:
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None, total=1)

    assert "из 1" not in out


def test_search_hits_say_how_many_matched() -> None:
    from aires_mcp.format import format_search_hits

    out = format_search_hits([_hit()], query="fraud", total=12)

    assert "12" in out


def test_ideas_lead_with_the_action_not_the_paper_title() -> None:
    """The question is "что можно сделать", so the answer is a list of things
    to do. A list led by paper titles makes the reader translate every line
    before they can judge it."""
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None)
    first_line = next(
        line for line in out.splitlines() if line.startswith(("1.", "**1"))
    )

    assert "Локальные NLP-модели" in first_line
    assert "Sparse graph embeddings" not in first_line


def test_the_source_paper_is_still_one_line_away() -> None:
    """Action-first must not cost traceability — the id is how anything gets
    checked or turned into a PRD."""
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None)

    assert "#14" in out
    assert "Sparse graph embeddings" in out
    assert "9/10" in out


def test_every_opportunity_of_a_paper_is_offered_not_just_the_first() -> None:
    from aires_mcp.format import format_ideas

    out = format_ideas([_seed()], topic=None, company=None)

    assert "Снижение расходов на GPU" in out


# ── MCP v2: фиксы по фидбэку аналитика Freedom Travel ──────────────────────


def test_summary_includes_the_full_ipr_resume() -> None:
    """«section=summary возвращает одну строку. Бесполезно» — фидбэк.

    Короткое описание остаётся анонсом, но следом обязан идти полный
    IPR-резюме из артефактов: он уже сгенерирован и оплачен.
    """
    from aires_mcp.format import format_article

    out = format_article(
        {
            "id": 265,
            "original_title": "Auditing Algorithmic Collusion",
            "descriptions": {"description_ru": "Одна строка."},
            "artifacts": {"summary": "Развёрнутое резюме на несколько абзацев " * 20},
            "stages": {},
        },
        section="summary",
    )

    assert "Одна строка." in out
    assert "Резюме статьи (IPR)" in out
    assert "Развёрнутое резюме" in out


def test_summary_without_ipr_does_not_invent_a_section() -> None:
    from aires_mcp.format import format_article

    out = format_article(
        {
            "id": 1,
            "original_title": "t",
            "descriptions": {"description_ru": "Кратко."},
            "artifacts": {},
            "stages": {},
        },
        section="summary",
    )

    assert "Резюме статьи (IPR)" not in out


def test_ideas_point_to_the_other_angles() -> None:
    """Кросс-компанийный слой (#265, MLOps-аудит) нашли руками в analysis.

    Выдача идей обязана говорить, что у статьи есть ещё углы, и куда за ними
    идти — иначе правило «одна статья — одна идея» прячет находки.
    """
    from aires_mcp.format import format_ideas

    out = format_ideas(
        [
            {
                "article_id": 265,
                "title": "Auditing",
                "company": "Freedom Travel",
                "relevance": 9,
                "reasoning": "why",
                "opportunities": ["аудит ценообразования"],
                "threats": [],
                "other_angles": 2,
            }
        ],
        topic=None,
        company="Freedom Travel",
        total=1,
    )

    assert "ещё 2 угол" in out
    assert "get_article(265, section='analysis')" in out


def test_ideas_stay_quiet_when_there_are_no_other_angles() -> None:
    from aires_mcp.format import format_ideas

    out = format_ideas(
        [
            {
                "article_id": 7,
                "title": "t",
                "company": "Arbuz.kz",
                "relevance": 8,
                "reasoning": "",
                "opportunities": ["идея"],
                "threats": [],
                "other_angles": 0,
            }
        ],
        topic=None,
        company=None,
        total=1,
    )

    assert "угол" not in out.split("Это сырьё")[0]


def test_coverage_reads_as_a_map_with_a_conclusion() -> None:
    """Таблица без вывода — просто цифры; вывод «дыра в корпусе, а не в
    поиске» и есть причина существования инструмента."""
    from aires_mcp.format import format_coverage

    out = format_coverage(
        {
            "total_articles": 266,
            "analysed": 256,
            "by_text_source": {"html": 8, "ocr": 258},
            "oldest": "2026-05-01",
            "newest": "2026-08-13",
            "companies": [
                {"company": "Freedom Travel", "articles": 23, "ge6": 14, "ge8": 2}
            ],
        }
    )

    assert "266" in out and "256" in out
    assert "| Freedom Travel | 23 | 14 | 2 |" in out
    assert "дыра в корпусе" in out


# ── MCP v2, заход 2: глубина чтения и экономия кругов ──────────────────────


def test_ipr_section_shows_thesis_and_citations() -> None:
    """IPR — тезисы с дословными цитатами; обе части лежат в БД раздельно."""
    from aires_mcp.format import format_article

    out = format_article(
        {
            "id": 265,
            "original_title": "Auditing",
            "descriptions": {},
            "artifacts": {
                "summary": "Тезисы статьи.",
                "citation_ru": "«дословная цитата из статьи»",
            },
            "stages": {},
        },
        section="ipr",
    )

    assert "IPR" in out
    assert "Тезисы статьи." in out
    assert "дословная цитата" in out


def test_text_page_names_the_next_page() -> None:
    """Продолжение должно быть на расстоянии одного copy-paste."""
    from aires_mcp.format import format_text_page

    out = format_text_page(
        {"article_id": 265, "page": 1, "pages": 3, "total_chars": 50000, "text": "тело"}
    )

    assert "страница 1 из 3" in out
    assert "section='full', page=2" in out


def test_last_text_page_does_not_invite_a_pointless_call() -> None:
    from aires_mcp.format import format_text_page

    out = format_text_page(
        {"article_id": 265, "page": 3, "pages": 3, "total_chars": 50000, "text": "хвост"}
    )

    assert "page=4" not in out


def test_brief_covers_the_whole_opening_chain() -> None:
    """Бриф заменяет цепочку покрытие → статьи → идеи одним ответом."""
    from aires_mcp.format import format_company_brief

    out = format_company_brief(
        "Freedom Travel",
        {
            "total_articles": 276,
            "companies": [
                {"company": "Freedom Travel", "articles": 23, "ge6": 14, "ge8": 2}
            ],
        },
        [{"id": 265, "original_title": "Auditing", "name_normalized": "2608_1"}],
        [
            {
                "article_id": 265,
                "relevance": 9,
                "opportunities": ["аудит ценообразования"],
            }
        ],
        ideas_total=14,
    )

    assert "Статей: 23" in out and "≥6: 14" in out
    assert "#265" in out
    assert "аудит ценообразования" in out
    assert "get_articles" in out


def test_brief_says_when_the_company_is_unknown() -> None:
    from aires_mcp.format import format_company_brief

    out = format_company_brief(
        "Нет Такой", {"total_articles": 276, "companies": []}, [], [], ideas_total=0
    )

    assert "нет статей, размеченных" in out
    assert "list_companies" in out
