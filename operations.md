# Operations

## Local Development
- **Prerequisites**:
  - Python 3.11+ (implied by FastAPI/async usage). No `requirements.txt` is provided; dependencies must be inferred from imports (FastAPI, Uvicorn, sqlite3, pymilvus, openai, passlib, google-auth, bs4, docx2txt, PyPDF2, aiohttp, structlog, etc.).【F:main.py†L18-L118】【F:ai/openai_integration.py†L13-L120】【F:web/scraping_helpers.py†L8-L16】 Document missing dependency manifest as TBD.
  - Milvus stack (etcd, MinIO) optional for vector storage; `docker-compose.yml` provisions containers.【F:docker-compose.yml†L11-L98】
  - OpenAI API key (`OPENAI_API_KEY`) for embeddings/completions; Google OAuth configuration optional (`GOOGLE_CLIENT_ID`).【F:ai/openai_integration.py†L44-L78】【F:main.py†L221-L312】

- **Environment configuration**:
  - `.env` expected by `config/settings.py` with variables such as `MILVUS_HOST`, `MILVUS_PORT`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SECRET_KEY`, `CACHE_TTL`, etc.【F:config/settings.py†L10-L47】
  - Additional env vars read directly include `SQLITE_DB_PATH`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `JWT_SECRET_KEY`, `GOOGLE_CLIENT_ID`, and server `PORT` overrides.【F:main.py†L100-L312】【F:auth/authentication.py†L16-L29】
  - `.env.example` documents extended optional settings (Stripe, Redis, Kafka, alternative databases) that are not wired into the FastAPI code yet; treat them as placeholders unless the corresponding integrations are implemented.【F:.env.example†L1-L154】

- **Database bootstrap**:
  - SQLite tables are created lazily through `ensure_tables`, `insert_chunk`, etc.【F:database/database_functions.py†L38-L112】
  - Chat schema initialization is attempted on startup (`init_chat_schema`, `enable_sqlite_wal`, `ensure_chat_indices`), but corresponding implementations are missing; chat persistence remains TBD.【F:main.py†L1862-L1880】【F:database/database_functions.py†L14-L196】
  - Milvus collections auto-create at startup via `DatabaseManager.initialize()` in `lifespan` context.【F:lifecycle/app_lifecycle.py†L1-L26】【F:core/database.py†L16-L210】

- **Running services**:
  - Start optional Milvus stack: `docker compose up -d` (from repo root) using provided compose file.【F:docker-compose.yml†L11-L98】
  - Launch API (development): `uvicorn main:app --reload --port ${PORT:-8000}`; same command executed in `__main__` guard.【F:main.py†L1958-L1961】
  - WebSocket endpoint listens at `/ws`; SSE streaming available at `/api/chat/ask/stream` for incremental chat updates.【F:main.py†L1955-L1959】【F:api/chats_router.py†L394-L505】

- **Testing & linting**:
  - No automated tests, lint scripts, or CI configs are present (TBD). `scripts/test_service.py` references external services but not wired into tooling.【F:scripts/test_service.py†L1-L200】

- **Sample workflow**:
  ```bash
  # Optional: start Milvus stack
  docker compose up -d

  # Export required keys
  export OPENAI_API_KEY=sk-...
  export JWT_SECRET_KEY=dev-secret

  # Run API
  uvicorn main:app --reload
  ```

## Observability
- **Logging**: Global logging configured via `logging.basicConfig(level=logging.INFO)` and `structlog` inside database/orchestrator modules. HTTP middleware logs request path and response status/time.【F:main.py†L102-L203】【F:processing/core_processing.py†L53-L58】
- **Metrics/Stats**: `system_stats` dictionary tracks counts (questions processed, cache hits, documents uploaded, API calls, WebSocket connections) and is surfaced in `/api/v1/stats` and `/health`.【F:main.py†L106-L200】【F:endpoints/api_endpoints.py†L118-L214】【F:main.py†L1898-L1949】
- **Health checks**: `/health` performs Milvus connectivity test, returns metrics, and clears expired cache entries.【F:main.py†L1898-L1949】
- **Debug endpoints**: `/__debug/sqlite` and `/__debug/routes` expose database info and route listings; `/api/v1/stats` provides operational snapshot.【F:main.py†L150-L177】【F:endpoints/api_endpoints.py†L118-L214】
- **Tracing/metrics exporters**: None configured (TBD).

## Deployments
- **Container orchestration**: `docker-compose.yml` provisions Milvus dependencies but not the FastAPI app; manual build/publish steps are absent.【F:docker-compose.yml†L11-L98】
- **Scripts**: `scripts/setup.py` targets a larger multi-service “Branch2” workspace (frontend, Go, Java, Rust) not present in this repo; treat as legacy/unverified instructions.【F:scripts/setup.py†L1-L200】
- **CI/CD**: No GitHub Actions or pipeline definitions (TBD).
- **Manual deployment**: Run `uvicorn main:app --host 0.0.0.0 --port ${PORT}`. Ensure Milvus reachable and environment variables set.

## Scaling & Performance
- **Server concurrency**: Depends on Uvicorn defaults; no Gunicorn or worker configuration provided. SSE/WebSocket endpoints imply asynchronous event loop usage.【F:main.py†L394-L727】【F:endpoints/websocket_handlers.py†L19-L170】
- **Vector search**: Milvus index choices (IVF_FLAT, HNSW) and thresholds configured in code; fallback to SQLite cosine similarity if Milvus offline.【F:core/database.py†L214-L366】【F:services/retreival.py†L112-L146】
- **Caching**: Simple in-memory cache via `db_manager.cache`; TTL and size configured in settings but used sparingly (`set_cache`, `clear_expired_cache`).【F:core/database.py†L24-L200】【F:core/database.py†L571-L690】【F:config/settings.py†L23-L28】
- **Resource limits**: OpenAI embedding requests truncate text to fit token budgets; file uploads limited to 5 MB and restricted MIME types.【F:ai/openai_integration.py†L74-L117】【F:main.py†L1689-L1729】

## Security
- **Authentication**: JWT (HS256) issued via `create_access_token` with a default 30-day expiry; tokens verified in dependency `get_current_user` and used for profile/system prompt endpoints. Secrets sourced from `JWT_SECRET_KEY` env var.【F:auth/authentication.py†L21-L92】【F:main.py†L425-L1674】
- **Password hashing**: Bcrypt via `passlib` with 12 rounds.【F:auth/authentication.py†L20-L43】
- **OAuth**: Google ID token verification with issuer check; fallback errors logged.【F:main.py†L221-L312】
- **CORS**: Strict origin allowlist configured globally for development/production frontends.【F:main.py†L131-L149】
- **File uploads**: MIME-type validation and size limit enforced; file contents processed in memory and may be base64 encoded for avatars (no persistent storage).【F:main.py†L604-L636】【F:main.py†L1689-L1772】
- **Secrets management**: All secrets read directly from environment variables; no vault integration (TBD).
- **Dependencies**: No lockfile; vulnerability scanning not configured (TBD).

## Backup / Restore & Migrations
- **SQLite**: No migration tooling; backups require manual copying of database file. Schema mismatches already exist (e.g., expected `embedding` column).【F:database/database_functions.py†L38-L103】【F:main.py†L871-L1188】
- **Milvus**: No backup scripts provided; operations rely on runtime collection creation. Compose stack stores data in Docker volumes (`etcd_data`, `minio_data`, `milvus_data`).【F:docker-compose.yml†L11-L98】
- **Migrations**: Not implemented; flagged as TBD.

## Incident Response
- **Error handling**: Endpoints raise `HTTPException` with descriptive messages; SSE/WebSocket flows emit structured error events for clients.【F:main.py†L252-L1949】【F:api/chats_router.py†L400-L717】【F:endpoints/websocket_handlers.py†L95-L170】
- **Logging**: Errors logged with stack traces (`logger.error/exception`) across modules (auth, scraping, processing).【F:main.py†L244-L318】【F:main.py†L857-L1858】【F:processing/core_processing.py†L53-L200】
- **Feature flags / kill switches**: None implemented (TBD).
- **Runbook tips**:
  - Check `/health` for Milvus connectivity and cache metrics.【F:main.py†L1898-L1949】
  - Inspect `/__debug/sqlite` to confirm SQLite availability and schema.【F:main.py†L150-L170】
  - Monitor logs for OpenAI rate limit warnings and fallback behavior.【F:ai/openai_integration.py†L13-L120】
  - Known issues: Missing chat persistence helpers, absent SQLite columns; errors will appear when chat endpoints invoked.

## Release Management
- **Versioning**: Application version string hard-coded as `2.0.0` in FastAPI metadata and health endpoint.【F:main.py†L124-L213】【F:main.py†L1900-L1907】
- **Changelog**: None present (TBD).
- **Branching strategy**: Not documented (TBD).

## Known Gaps / TODOs
- Provide dependency manifest (`requirements.txt` or `pyproject.toml`).【F:main.py†L18-L118】
- Implement missing SQLite chat helpers and schema migrations for expected columns/tables.【F:api/chats_router.py†L14-L1204】【F:main.py†L871-L1188】
- Add tests and CI pipelines to validate functionality (TBD).【F:scripts/test_service.py†L1-L200】
