"""HTTP-режим сервера: подключение из claude.ai и Claude Desktop.

Локальный (stdio) режим запускает сервер на машине пользователя. Веб так не
умеет по построению: к серверу подключается облако Anthropic, а не браузер, —
значит нужен адрес в интернете. Тот же самый набор инструментов, другой
транспорт.

Доступ — персональные токены, а не общий пароль: у каждого свой, отзывается
по одному, в логах видно, кто ходил. Токен принимается тремя способами, потому
что интерфейсы Claude отличаются:

* ``Authorization: Bearer <токен>`` — как настраивают заголовки в claude.ai
  (функция в бете и есть не у всех);
* ``X-Api-Key: <токен>`` — второй разрешённый там заголовок;
* ``/t/<токен>/mcp`` в пути — работает всегда и везде, без бет. Токен виден в
  настройках коннектора, поэтому путь не пишется в логи целиком.

Переменные окружения:
  AIRES_MCP_TOKENS   ``токен1:Имя,токен2:Другое Имя`` — обязательна;
                     без неё сервер не поднимется, иначе корпус оказался бы
                     открыт всему интернету.
  AIRES_API_URL      адрес API корпуса (как в stdio-режиме).
  AIRES_BASIC_AUTH   доступ сервера к API — свой, общий на все токены.
"""

from __future__ import annotations

import os
from typing import Any

from aires_mcp.server import build_server

_TOKEN_PREFIX = "/t/"


class TokenGuardError(RuntimeError):
    """Конфигурация не позволяет безопасно поднять сервер."""


def load_tokens(raw: str | None = None) -> dict[str, str]:
    """``токен:Имя`` через запятую → ``{токен: имя}``.

    Пустая конфигурация — ошибка, а не «пускаем всех»: HTTP-сервер виден из
    интернета, и молчаливое открытие корпуса было бы худшим из возможных
    поведений по умолчанию.
    """
    raw = raw if raw is not None else os.environ.get("AIRES_MCP_TOKENS", "")
    tokens: dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        token, _, name = chunk.partition(":")
        token = token.strip()
        if len(token) < 16:
            raise TokenGuardError(
                f"токен «{token[:4]}…» короче 16 символов — подбирается перебором"
            )
        tokens[token] = name.strip() or "без имени"
    if not tokens:
        raise TokenGuardError(
            "AIRES_MCP_TOKENS пуст: HTTP-сервер без токенов открыл бы корпус всем"
        )
    return tokens


def _unauthorized(message: str) -> dict[str, Any]:
    body = (
        b'{"jsonrpc":"2.0","error":{"code":-32001,"message":"'
        + message.encode()
        + b'"},"id":null}'
    )
    return {
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            # Подсказка клиенту, каким способом авторизоваться.
            (b"www-authenticate", b'Bearer realm="aires"'),
        ],
        "body": body,
    }


class TokenGuard:
    """ASGI-обёртка: пускает дальше только с известным токеном.

    Вырезает токен из пути перед тем, как отдать запрос MCP-приложению, —
    само приложение обслуживает один-единственный путь и про токены не знает.
    """

    def __init__(self, app: Any, tokens: dict[str, str], mount: str = "/mcp") -> None:
        self._app = app
        self._tokens = tokens
        self._mount = mount

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Проба живости — единственное, что открыто: она не выдаёт ничего о
        # корпусе, зато позволяет мониторингу видеть, что сервис поднят.
        if path == "/healthz":
            await self._plain(send, 200, b"ok")
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        token = None

        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if not token:
            token = headers.get("x-api-key", "").strip() or None

        # Токен в пути: /t/<токен>/mcp → /mcp
        if path.startswith(_TOKEN_PREFIX):
            rest = path[len(_TOKEN_PREFIX) :]
            candidate, _, tail = rest.partition("/")
            if candidate:
                token = token or candidate
                scope = dict(scope)
                scope["path"] = "/" + tail if tail else self._mount
                scope["raw_path"] = scope["path"].encode()

        name = self._tokens.get(token or "")
        if name is None:
            resp = _unauthorized("unauthorized: valid AIRES token required")
            await self._plain(send, resp["status"], resp["body"], resp["headers"])
            return

        # Имя владельца токена — в scope: пригодится логам доступа.
        scope = dict(scope)
        scope.setdefault("state", {})
        scope["state"] = {**scope.get("state", {}), "aires_client": name}
        await self._app(scope, receive, send)

    @staticmethod
    async def _plain(send: Any, status: int, body: bytes, headers: Any = None) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers
                or [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def build_http_app(tokens: dict[str, str] | None = None) -> Any:
    """MCP поверх streamable-HTTP, закрытый персональными токенами."""
    mcp = build_server()
    return TokenGuard(mcp.streamable_http_app(), tokens or load_tokens())


def main() -> None:
    """Точка входа ``aires-mcp-http``."""
    import uvicorn

    app = build_http_app()
    uvicorn.run(
        app,
        host=os.environ.get("AIRES_MCP_HOST", "127.0.0.1"),
        port=int(os.environ.get("AIRES_MCP_PORT", "8765")),
        # Пути не логируем: в них может лежать токен.
        access_log=False,
    )


__all__ = ["TokenGuard", "TokenGuardError", "build_http_app", "load_tokens", "main"]
