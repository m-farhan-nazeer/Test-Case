# Repository Configuration

## Layout Overview
| Path | Description |
| --- | --- |
| `main.py` | FastAPI entrypoint configuring routes, middleware, auth, SSE, and WebSockets.【F:main.py†L123-L1961】 |
| `endpoints/` | HTTP handlers for questions, web search, and WebSockets.【F:endpoints/api_endpoints.py†L1-L323】【F:endpoints/websocket_handlers.py†L19-L170】 |
| `api/` | Chat router and orchestrator logic for multi-source retrieval.【F:api/chats_router.py†L14-L1204】【F:api/orchestrator.py†L1-L200】 |
| `processing/` | Unified question processing pipeline with retrieval, scraping, and synthesis coordination.【F:processing/core_processing.py†L1-L200】 |
| `ai/` | OpenAI client wrapper providing embeddings, chat completions, streaming, and retry logic.【F:ai/openai_integration.py†L1-L384】 |
| `agents/` | Legacy orchestrator components and router glue for multi-agent workflows.【F:agents/router.py†L1-L200】 |
| `core/database.py` | Milvus connection manager, vector store, user/system managers.【F:core/database.py†L16-L695】 |
| `database/` | SQLite helpers and Milvus utility wrappers (note chat helpers missing).【F:database/database_functions.py†L14-L196】【F:database/milvus_functions.py†L1-L32】 |
| `models/` | Pydantic schemas for API payloads (chat/auth).【F:models/pydantic_models.py†L5-L129】【F:models/chat.py†L1-L162】 |
| `services/` | Retrieval and citation utilities used across endpoints.【F:services/retreival.py†L1-L146】【F:services/citations.py†L1-L73】 |
| `search/`, `vector/`, `synthesis/`, `scoring/`, `context/`, `content/`, `utils/`, `files/` | Supporting modules for search orchestration, similarity functions, synthesis agent scoring, prompt formatting, document chunking, utilities, and file extraction.【F:search/vectorized_search.py†L1-L200】【F:vector/similarity_functions.py†L1-L120】【F:synthesis/synthesis_agent.py†L20-L193】【F:context/context_creation.py†L1-L200】【F:content/document_processing.py†L1-L200】【F:files/file_processing.py†L1-L200】 |
| `docker-compose.yml` | Milvus + dependencies for local development.【F:docker-compose.yml†L11-L98】 |
| `scripts/` | Legacy setup scripts referencing external services (PostgreSQL, Redis, Elasticsearch, multi-language components).【F:scripts/setup.py†L1-L200】【F:scripts/init_databases.py†L1-L200】 |
| `.env.example` | Comprehensive list of optional environment variables for adjacent services (Stripe, Redis, Kafka, etc.).【F:.env.example†L1-L154】 |

## Branching & Commit Conventions
- No documented branching strategy or commit message convention (TBD).

## Code Quality & Tooling
- **Linters/formatters**: None configured; no `pyproject.toml`, `setup.cfg`, or lint config present (TBD).
- **Type checking**: No mypy/pyright configuration (TBD).
- **Tests**: No automated tests in repository (TBD). `scripts/test_service.py` appears to be a manual integration script rather than a unit test suite.【F:scripts/test_service.py†L1-L200】

## Continuous Integration / Delivery
- No GitHub Actions, GitLab pipelines, or other CI/CD workflows included (TBD).

## Pre-commit Hooks
- No `.pre-commit-config.yaml` or Husky configuration present (TBD).

## Dependency Management
- No dependency manifest provided (`requirements.txt`, `poetry.lock`, etc.); dependencies inferred from imports (FastAPI, Uvicorn, pymilvus, openai, passlib, google-auth, BeautifulSoup, docx2txt, PyPDF2, structlog, aiohttp, etc.). Documented in README and operations notes as TBD.【F:main.py†L18-L118】【F:ai/openai_integration.py†L13-L120】【F:web/scraping_helpers.py†L8-L16】
- Environment variables documented in `README.md` section above; `.env` loading handled by `config/settings.py` but no sample file checked in.【F:config/settings.py†L1-L34】【F:scripts/setup.py†L79-L97】

## IDE / Editor Settings
- No `.editorconfig`, VS Code workspace, or recommended extensions present (TBD).

## Known Configuration Gaps
- Chat persistence helpers absent; implement in `database/database_functions.py` before enabling chat routes.【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】
- SQLite schema mismatches (`documents.embedding`, `users` table) need migrations to align with endpoint expectations.【F:main.py†L766-L1855】【F:database/database_functions.py†L38-L103】
- Provide dependency lockfile and CI pipeline to ensure reproducible builds (TBD).
