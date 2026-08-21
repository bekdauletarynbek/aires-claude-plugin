"""HTTP-режим: кого пускаем к базе данных и как именно.

Этот сервер, в отличие от stdio-режима, виден из интернета — значит охрана
двери и есть его главная функция. Ошибка здесь стоит не падения, а утечки
всей базы данных, поэтому проверяется каждый способ авторизации и каждый отказ.
"""

from __future__ import annotations

import pytest

from aires_mcp.http_server import TokenGuard, TokenGuardError, load_tokens

TOKEN = "tok_aaaaaaaaaaaaaaaaaaaa"
TOKENS = {TOKEN: "Бекдаулет"}


# ── разбор конфигурации ────────────────────────────────────────────────


def test_empty_config_refuses_to_start() -> None:
    """Молчаливый старт без токенов означал бы открытую базу данных."""
    with pytest.raises(TokenGuardError):
        load_tokens("")


def test_short_token_rejected() -> None:
    """Короткий токен перебирается за часы — это не защита."""
    with pytest.raises(TokenGuardError):
        load_tokens("short:Кто-то")


def test_tokens_parsed_with_names() -> None:
    parsed = load_tokens(f"{TOKEN}:Бекдаулет, tok_bbbbbbbbbbbbbbbbbbbb :Сергей")
    assert parsed == {TOKEN: "Бекдаулет", "tok_bbbbbbbbbbbbbbbbbbbb": "Сергей"}


def test_token_without_name_still_works() -> None:
    assert load_tokens(TOKEN) == {TOKEN: "без имени"}


# ── охрана двери ───────────────────────────────────────────────────────


class _Spy:
    """Заглушка MCP-приложения: запоминает, что до него дошло."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        self.calls.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


async def _call(guard: TokenGuard, path: str, headers: list[tuple[bytes, bytes]] | None = None):
    sent: list[dict] = []

    async def receive():  # noqa: ANN202
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):  # noqa: ANN001, ANN202
        sent.append(msg)

    await guard({"type": "http", "path": path, "headers": headers or []}, receive, send)
    return sent


@pytest.mark.anyio
async def test_no_token_is_401() -> None:
    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), "/mcp")
    assert sent[0]["status"] == 401
    assert not spy.calls, "запрос без токена не должен доходить до инструментов"


@pytest.mark.anyio
async def test_wrong_token_is_401() -> None:
    spy = _Spy()
    sent = await _call(
        TokenGuard(spy, TOKENS), "/mcp", [(b"authorization", b"Bearer tok_wrongwrongwrong")]
    )
    assert sent[0]["status"] == 401
    assert not spy.calls


@pytest.mark.anyio
async def test_bearer_header_accepted() -> None:
    spy = _Spy()
    sent = await _call(
        TokenGuard(spy, TOKENS), "/mcp", [(b"authorization", f"Bearer {TOKEN}".encode())]
    )
    assert sent[0]["status"] == 200
    assert spy.calls[0]["state"]["aires_client"] == "Бекдаулет"


@pytest.mark.anyio
async def test_api_key_header_accepted() -> None:
    """Второй заголовок из разрешённых в claude.ai."""
    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), "/mcp", [(b"x-api-key", TOKEN.encode())])
    assert sent[0]["status"] == 200


@pytest.mark.anyio
async def test_token_in_path_accepted_and_stripped() -> None:
    """Путь /t/<токен>/mcp работает там, где заголовки задать негде;
    до MCP-приложения должен дойти обычный /mcp."""
    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), f"/t/{TOKEN}/mcp")
    assert sent[0]["status"] == 200
    assert spy.calls[0]["path"] == "/mcp", "токен обязан быть вырезан из пути"


@pytest.mark.anyio
async def test_wrong_token_in_path_is_401() -> None:
    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), "/t/tok_nopenopenopenopen/mcp")
    assert sent[0]["status"] == 401
    assert not spy.calls


@pytest.mark.anyio
async def test_healthz_is_open_but_says_nothing() -> None:
    """Мониторингу нужно видеть, что сервис жив; база данных при этом не выдаётся."""
    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), "/healthz")
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"ok"
    assert not spy.calls


def test_allowed_hosts_include_public_domain_and_localhost(monkeypatch) -> None:
    """За обратным прокси Host приходит внешний — без него SDK отвечает
    «Invalid Host header» и коннектор не подключается вовсе."""
    from aires_mcp.http_server import _allowed_hosts

    monkeypatch.setenv("AIRES_MCP_HOSTS", "mcp-api.example.com, other.example.com")
    hosts = _allowed_hosts()
    assert "mcp-api.example.com" in hosts
    assert "other.example.com" in hosts
    assert "127.0.0.1" in hosts, "проба живости ходит на loopback"


# ─── авторизация через гугловый вход ───────────────────────────────────


@pytest.mark.anyio
async def test_unauthorized_points_to_the_login_server() -> None:
    """По этому заголовку клиент Claude и находит, где авторизоваться."""
    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), "/mcp")

    header = dict(sent[0]["headers"])[b"www-authenticate"].decode()
    assert "resource_metadata=" in header
    assert "/.well-known/oauth-protected-resource" in header


@pytest.mark.anyio
async def test_resource_metadata_is_public_and_names_the_auth_server() -> None:
    """Документ обязан читаться без токена: его смотрят ДО авторизации."""
    import json as js

    spy = _Spy()
    sent = await _call(TokenGuard(spy, TOKENS), "/.well-known/oauth-protected-resource")

    assert sent[0]["status"] == 200
    body = js.loads(sent[1]["body"])
    assert body["authorization_servers"], "не указан сервер авторизации"
    assert not spy.calls, "метаданные не должны доходить до инструментов"


@pytest.mark.anyio
async def test_oauth_token_is_checked_against_auth_server(monkeypatch) -> None:
    """Токен гуглового входа принимается, только если сервер подтвердил его."""
    from aires_mcp import http_server

    seen: dict[str, str] = {}

    def fake_introspect(token: str) -> bool:
        seen["token"] = token
        return token.endswith("good")

    monkeypatch.setattr(http_server, "_introspect", fake_introspect)

    spy = _Spy()
    ok = await _call(TokenGuard(spy, TOKENS), "/mcp",
                     [(b"authorization", b"Bearer mcpat_good")])
    assert ok[0]["status"] == 200
    assert seen["token"] == "mcpat_good"

    spy2 = _Spy()
    denied = await _call(TokenGuard(spy2, TOKENS), "/mcp",
                         [(b"authorization", b"Bearer mcpat_bad")])
    assert denied[0]["status"] == 401
    assert not spy2.calls
