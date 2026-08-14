"""HTTP client for the AIres API, shaped for a language model to read.

Deliberately separate from the tool wiring in :mod:`aires_mcp.server`: this
half has no MCP dependency and can be exercised over a mock transport, which
is where the behaviour worth pinning actually lives — auth, error text, and
how much prose comes back.

Two rules run through it:

*Errors are prose.* Whatever this raises ends up in the model's context as the
tool result. ``AiresError("PRD_ALREADY_GENERATED: ...")`` gives it something
to act on; an httpx traceback does not.

*Length is a budget.* Tool results are pasted into the conversation, and a PRD
here can run to twenty thousand tokens. Long fields come back cut, with the
original size and the argument that fetches the rest.
"""

from __future__ import annotations

import base64
from types import TracebackType
from typing import Any

import httpx

__all__ = ["AiresClient", "AiresError", "truncate"]

DEFAULT_TIMEOUT = 30.0


class AiresError(RuntimeError):
    """Anything the caller should be told about, phrased for a reader."""


def truncate(text: str | None, *, limit: int, hint: str | None = None) -> str:
    """Cut *text* to *limit* characters, saying what was cut.

    A silent cut is worse than a long answer: the model treats the fragment as
    the whole document and reasons from a truncated PRD without knowing it.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    tail = f"\n\n[…обрезано: показано {limit} из {len(text)} символов"
    tail += f"; полностью — {hint}]" if hint else "]"
    return text[:limit] + tail


class AiresClient:
    """Thin async wrapper over the AIres HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        basic_auth: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base = base_url.rstrip("/")
        headers: dict[str, str] = {"User-Agent": "aires-mcp/0.1"}
        if basic_auth:
            # Production sits behind nginx Basic Auth. Encoded here rather than
            # passed to httpx's auth= so it also survives redirects to the
            # same host, which is how the prod vhost serves /api.
            token = base64.b64encode(basic_auth.encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        self._client = httpx.AsyncClient(
            base_url=self._base, headers=headers, timeout=timeout
        )

    async def __aenter__(self) -> AiresClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── plumbing ───────────────────────────────────────────────────────

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            # Name the host: "connection refused" alone leaves the user
            # guessing which of AIRES_API_URL / VPN / nginx is wrong.
            raise AiresError(
                f"не удалось обратиться к {self._base}{path}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise AiresError(self._describe_failure(response))
        try:
            return response.json()
        except ValueError as exc:
            raise AiresError(
                f"{self._base}{path} вернул не JSON "
                f"(HTTP {response.status_code})"
            ) from exc

    @staticmethod
    def _describe_failure(response: httpx.Response) -> str:
        """Turn an error response into one line worth reading.

        The API answers with ``{"error": {"code", "message"}}``; the code is
        the part a caller can branch on, so it leads.
        """
        detail = ""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                code = err.get("code", "")
                message = err.get("message", "")
                detail = f"{code}: {message}".strip(": ")
            elif body.get("detail"):
                detail = str(body["detail"])
        if not detail:
            detail = response.text[:200].strip() or response.reason_phrase
        return f"HTTP {response.status_code} — {detail}"

    # ── calls ──────────────────────────────────────────────────────────

    async def article_text(
        self, article_id: int, *, page: int = 1
    ) -> dict[str, Any]:
        """Одна страница полного текста статьи (~20К символов)."""
        return await self._request(
            "GET", f"/articles/{article_id}/text", params={"page": page}
        )

    async def corpus_coverage(self) -> dict[str, Any]:
        """Сводка покрытия корпуса — по компаниям, ярусам и датам."""
        return await self._request("GET", "/corpus/coverage")

    async def search(
        self, query: str, *, limit: int = 10
    ) -> tuple[list[dict[str, Any]], int]:
        """Hits plus how many distinct articles matched in total.

        The total travels with the hits because a capped list that does not
        say it is capped gets treated as the whole corpus.
        """
        data = await self._request(
            "POST", "/search", json={"query": query, "limit": limit}
        )
        items = list(data.get("items", []))
        return items, int(data.get("total_found", len(items)))

    async def ideas(
        self,
        *,
        topic: str | None = None,
        company: str | None = None,
        min_relevance: int | None = None,
        limit: int = 10,
    ) -> tuple[list[dict[str, Any]], int]:
        """Idea seeds plus the total that matched — see :meth:`search`."""
        payload: dict[str, Any] = {"limit": limit}
        # Empty strings are not "no filter" to the API: topic="" would be a
        # blank semantic query, and the bare "что можно сделать?" case would
        # come back empty.
        if topic and topic.strip():
            payload["topic"] = topic.strip()
        if company and company.strip():
            payload["company"] = company.strip()
        if min_relevance is not None:
            payload["min_relevance"] = min_relevance
        data = await self._request("POST", "/ideas", json=payload)
        items = list(data.get("items", []))
        return items, int(data.get("total_available", len(items)))

    async def get_article(self, article_id: int) -> dict[str, Any]:
        return dict(await self._request("GET", f"/articles/{article_id}"))

    async def list_companies(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/companies")
        return list(data.get("items", []))

    async def company_articles(
        self,
        company: str,
        *,
        min_relevance: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"company": company, "limit": limit}
        if min_relevance is not None:
            params["min_relevance"] = min_relevance
        data = await self._request("GET", "/articles", params=params)
        return list(data.get("items", []))

    async def prd_brief(self, article_id: int) -> dict[str, Any]:
        """Промпты + материалы статьи для клиентской генерации PRD."""
        return dict(
            await self._request("GET", f"/articles/{article_id}/prd/brief")
        )

    async def submit_prd(
        self, article_id: int, prd_markdown: str
    ) -> dict[str, Any]:
        """Сохранить написанный клиентом PRD; сервер достроит презентацию."""
        return dict(
            await self._request(
                "PUT",
                f"/articles/{article_id}/prd",
                json={"prd_markdown": prd_markdown},
            )
        )

    def deck_url(self, article_id: int) -> str:
        """Link to the .pptx.

        A link rather than bytes: the deck is megabytes of zip, and MCP tool
        results are text going into a context window.
        """
        return f"{self._base}/articles/{article_id}/export.pptx"

    def pdf_url(self, article_id: int) -> str:
        return f"{self._base}/articles/{article_id}/pdf"
