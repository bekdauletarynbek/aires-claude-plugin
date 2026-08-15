# AIRES — плагин для Claude Code

Подключает Claude к корпусу проанализированных arXiv-статей: семантический
поиск по полным текстам, разбор статей под конкретные компании, идеи с
возможностями и рисками, генерация PRD.

Плагин ставит три вещи: MCP-сервер (доступ к корпусу), агента-исследователя и
четыре команды.

## Установка

```
/plugin marketplace add bekdauletarynbek/aires-claude-plugin
/plugin install aires@aires
```

При установке Claude спросит доступ к API — `логин:пароль`. **Плагин открытый,
а данные корпуса — нет:** доступы выдаются лично — телеграм [@zzzSergeyzzz](https://t.me/zzzSergeyzzz). Пароль
хранится в конфиге Claude как секрет.

Из зависимостей нужен только [`uv`](https://astral.sh/uv) — MCP-сервер
подтянется из этого репозитория автоматически:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Что появляется

**Агент `aires-research`.** Claude зовёт его сам, когда вопрос требует опоры на
статьи: «что есть по теме», «какие идеи для компании», «применимо ли это к
нам». Агент отвечает только из корпуса, ссылается на номера статей, отличает
дословные цитаты от пересказа и отдельным разделом пишет, что осталось
непроверенным.

**Команды:**

| Команда | Что делает |
|---|---|
| `/aires:brief <компания>` | Бриф: покрытие корпуса, возможности, риски |
| `/aires:find <тема>` | Поиск и разбор применимости |
| `/aires:ideas [тема] [компания]` | Идеи, выведенные из анализа статей |
| `/aires:prd <id статьи>` | Написать PRD по статье и сохранить в корпус |

**Инструменты MCP:** `corpus_coverage`, `company_brief`, `search_articles`,
`get_article`, `get_articles`, `suggest_ideas`, `list_companies`,
`company_articles`, `prd_brief`, `save_prd`, `get_deck_link`.

## Подключение из claude.ai и Claude Desktop

Плагин (агент и команды) работает только в Claude Code. Сам корпус подключается
и в других клиентах — как коннектор по личной ссылке:

```
https://mcp-api.ai-marketing.cloud/t/ВАШ-ТОКЕН/mcp
```

claude.ai: Настройки → Connectors → Add custom connector → вставить ссылку.
Team/Enterprise: владелец добавляет её в Organization settings → Connectors, дальше
участники жмут «Connect» у себя.

Если у вас включена бета «Request headers», можно указать адрес без токена
(`https://mcp-api.ai-marketing.cloud/mcp`) и заголовок `Authorization` со значением
`Bearer ВАШ-ТОКЕН`.

Ссылку с токеном берите у [@zzzSergeyzzz](https://t.me/zzzSergeyzzz).

## Настройки

| Переменная | Значение |
|---|---|
| `AIRES_API_URL` | Адрес API инстанса, обязательно с `/api` на конце |
| `AIRES_BASIC_AUTH` | `логин:пароль` — строго в таком виде, без «Basic» и без base64 |

Обе задаются при установке плагина; для голого MCP без плагина — через
`claude mcp add` или конфиг Claude Desktop.

## PRD пишет ваша модель

`prd_brief(id)` отдаёт шаблон документа и материалы статьи, вы (точнее, ваш
Claude) пишете PRD, `save_prd(id, ...)` сохраняет его в корпус и запускает
сборку презентации. Серверной генерации нет — она стоила около $1.19 за
документ из общего бюджета и была убрана намеренно.

## Разработка

MCP-сервер — в `mcp-server/`, тесты запускаются так:

```bash
cd mcp-server
uv venv && uv pip install -e . pytest pytest-asyncio respx
.venv/bin/pytest tests -q
```

После правок поднимайте `version` в `.claude-plugin/plugin.json`: без этого
обновление не приедет установленным пользователям.
