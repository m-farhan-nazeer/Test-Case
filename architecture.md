# Architecture

## Overview
The system is a FastAPI application that exposes REST, WebSocket, and Server-Sent Events (SSE) interfaces for question answering, chat, document management, and user authentication. A single FastAPI instance is created in `main.py`, where CORS, middleware, WebSocket handlers, and the chat router are mounted once the `lifespan` manager wires Milvus initialization and shutdown.【F:main.py†L123-L215】【F:lifecycle/app_lifecycle.py†L15-L30】 Unified question handling is delegated to `processing/core_processing.py`, which orchestrates retrieval, web augmentation, synthesis, and stats tracking for both HTTP and streaming flows while sharing global caches and stats across connection types.【F:processing/core_processing.py†L39-L200】【F:endpoints/websocket_handlers.py†L19-L143】

## Runtime Topology
- **API server**: FastAPI served by Uvicorn; request logging middleware, global CORS, and shared `system_stats` counters are configured in `main.py`, which also binds the WebSocket endpoint and chat SSE router for streaming conversations.【F:main.py†L131-L213】【F:main.py†L183-L205】【F:main.py†L1898-L1959】
- **Database layer**:
  - **SQLite**: Default persistence for documents, chunks, and embeddings via helpers in `database/database_functions.py`; also used for debugging endpoints.【F:database/database_functions.py†L14-L112】【F:main.py†L150-L170】
  - **Milvus**: Vector database accessed through `core/database.py`; collections for documents, users, chats, and system stats are created at initialization and used for similarity search, analytics, and user storage.【F:core/database.py†L16-L210】【F:core/database.py†L214-L351】
- **External services**:
  - **OpenAI**: Embedding and chat completion requests through `ai/openai_integration.py`, including retry handling, streaming token generation, and shared embedding caches that avoid duplicate computations inside a request.【F:ai/openai_integration.py†L18-L132】【F:ai/openai_integration.py†L198-L384】
  - **Google OAuth**: Token verification to support `/api/auth/google`.【F:main.py†L221-L312】
  - **Web scraping & search**: Custom scraping utilities in `web/scraping_helpers.py` and Google search integration invoked by web-search endpoints.【F:web/scraping_helpers.py†L19-L198】【F:endpoints/web_search_endpoints.py†L17-L144】
- **Optional infrastructure**: `docker-compose.yml` provisions Milvus (with etcd and MinIO) and Attu explorer for local development.【F:docker-compose.yml†L11-L98】 Scripts reference PostgreSQL, Redis, and Elasticsearch bootstrapping, but no runtime code currently consumes those services.【F:scripts/init_databases.py†L1-L200】

## Component Map
- **Routing layer**: FastAPI routes in `main.py` for auth, users, documents, search, settings, uploads, health, and WebSockets; chat-specific routes live under `api/chats_router.py` and are mounted at `/api` with REST and SSE handlers for chat creation, streaming responses, and document attachment flows.【F:main.py†L252-L1889】【F:api/chats_router.py†L194-L706】
- **Application services**:
  - `processing/core_processing.py`: Implements the question pipeline (context assembly, retrieval selection, synthesis) and is invoked by both `/api/v1/ask` and WebSocket handlers.【F:main.py†L833-L836】【F:processing/core_processing.py†L39-L200】
  - `services/retreival.py`: Provides document ingestion and embedding lookup helpers for chunk-level similarity; used indirectly by chat retrieval stubs and other modules.【F:services/retreival.py†L38-L146】
  - `api/orchestrator.py` and `agents/*`: Define an alternative multi-source orchestration pipeline (vector, graph, search, cache) that coordinates question analysis and routing via LangChain and custom managers.【F:api/orchestrator.py†L1-L200】【F:agents/router.py†L1-L200】
- **Data access**: `core/database.py` encapsulates Milvus operations (upserts, search, user/system managers). `database/database_functions.py` provides SQLite CRUD but lacks the chat-specific helpers expected by the chat router, leaving persistence incomplete.【F:core/database.py†L214-L695】【F:database/database_functions.py†L14-L196】【F:api/chats_router.py†L14-L1204】
- **Presentation utilities**: `services/citations.py`, `context/*`, and `endpoints/websocket_handlers.py` generate citation blocks, prompt formatting, and structured streaming payloads consumed by SSE/WebSocket clients.【F:services/citations.py†L4-L73】【F:context/context_creation.py†L1-L200】【F:endpoints/websocket_handlers.py†L19-L160】

## Data Flow
A typical REST question lifecycle:

```
Client -> FastAPI route (/api/v1/ask)
       -> Question request validation (QuestionRequest model)
       -> processing.core_processing.process_question_unified()
           -> URL detection & optional web scraping/search
           -> Milvus/SQLite retrieval and context assembly
           -> SynthesisAgent for ranking & answer generation
       -> Response serialization with processing metrics
```

The `/api/v1/ask` route forwards validated requests to `process_question_unified`, which clears embedding caches, merges personalization context, optionally scrapes URLs, performs Milvus and in-memory searches, and synthesizes an answer before returning JSON with confidence, citations, and timing. Stats are updated on the shared `system_stats` object during middleware and processing steps.【F:main.py†L833-L949】【F:processing/core_processing.py†L59-L200】【F:main.py†L106-L205】 WebSocket questions reuse the same logic but provide progress updates via the `ConnectionManager` callbacks, while SSE chat streaming emits incremental `start`, `meta`, `delta`, `saved`, and `done` events as the OpenAI streaming client yields tokens.【F:endpoints/websocket_handlers.py†L19-L160】【F:api/chats_router.py†L394-L505】 Chat SSE flows still expect SQLite helpers for persistence which are currently missing.【F:api/chats_router.py†L444-L706】【F:database/database_functions.py†L14-L196】

## Cross-Cutting Concerns
- **Authentication**: JWT-based auth with bcrypt hashing in `auth/authentication.py`; `HTTPBearer` guards profile endpoints and identity is stored in Milvus collections using delete-and-recreate semantics for profile changes.【F:auth/authentication.py†L16-L92】【F:main.py†L425-L596】【F:core/database.py†L456-L571】
- **CORS**: Allowed origins enumerated and middleware registered globally for REST clients.【F:main.py†L131-L149】
- **Logging**: Standard logging via `logging` and `structlog` across modules; request middleware logs both request and response metadata.【F:main.py†L183-L203】【F:processing/core_processing.py†L53-L58】
- **Configuration**: Environment-backed settings via `config/settings.py`; additional `os.getenv` checks exist for database paths, API keys, and ports.【F:config/settings.py†L1-L34】【F:main.py†L100-L200】
- **Error handling**: Extensive `HTTPException` usage for API errors; SSE/WebSocket handlers emit structured error events; retries for OpenAI rate limits.【F:main.py†L252-L1180】【F:api/chats_router.py†L430-L717】【F:ai/openai_integration.py†L13-L116】
- **Stats/metrics**: `system_stats` tracks processed questions, cache hits, API calls, uptime, and Milvus availability; `/api/v1/stats`, `/health`, middleware, and WebSocket handlers increment counters and expose metrics to clients.【F:main.py†L106-L205】【F:endpoints/api_endpoints.py†L39-L214】【F:main.py†L1898-L1949】【F:endpoints/websocket_handlers.py†L72-L143】

## Key Design Decisions
- **Milvus-first architecture**: Collections for documents, users, chats, and system stats are eagerly created with indices; many API operations short-circuit when `db_manager.milvus_connected` is false, emphasizing Milvus as the primary store.【F:core/database.py†L16-L351】【F:main.py†L289-L360】
- **LLM orchestration**: The core processing pipeline couples Milvus retrieval, optional web search, and an OpenAI-driven synthesis agent, enabling consistent responses for synchronous HTTP, SSE streaming, and WebSocket flows through shared helpers and streaming APIs.【F:processing/core_processing.py†L59-L200】【F:api/chats_router.py†L394-L626】【F:ai/openai_integration.py†L236-L384】
- **Fallback behavior**: Numerous endpoints fall back to in-memory document storage or return limited responses when Milvus or SQLite are unavailable, prioritizing resilience over strict persistence.【F:main.py†L891-L1150】【F:processing/core_processing.py†L170-L200】
- **Known gaps**: Chat persistence helpers (`ensure_chat`, `append_message`, etc.) referenced by the chat router are absent from `database/database_functions.py`, preventing chat history from being saved; document queries expect SQLite `embedding` columns that are not created by the schema helpers. These missing implementations are explicitly noted and require completion for full functionality.【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】【F:main.py†L871-L1160】

## Integrations
- **OpenAI API** for embeddings and completions, including streaming responses and retry policies.【F:ai/openai_integration.py†L13-L384】
- **Google OAuth** for social login flows.【F:main.py†L221-L312】
- **Custom web scraping** for augmenting context and storing documents scraped from URLs.【F:web/scraping_helpers.py†L19-L198】【F:main.py†L1083-L1194】
- **Milvus vector database** for document embeddings, user storage, and metrics persistence.【F:core/database.py†L16-L351】
- **Optional infrastructure scripts** referencing PostgreSQL, Redis, and Elasticsearch initialization, though no production code currently consumes those services (marked TBD for confirmation).【F:scripts/init_databases.py†L1-L200】

## Environment Layout
- `.env` is expected by `config/settings.py`; required variables include Milvus host/port, OpenAI keys, and JWT secrets referenced via `os.getenv` calls, while `.env.example` enumerates additional optional integrations (Stripe, Redis, Kafka, alternate databases) that are not wired into the FastAPI code yet.【F:config/settings.py†L10-L47】【F:main.py†L221-L312】【F:auth/authentication.py†L16-L29】【F:.env.example†L1-L154】
- `docker-compose.yml` defines a development stack for Milvus, MinIO, etcd, and Attu; no profiles for staging/production are encoded.【F:docker-compose.yml†L11-L98】
- Scripts under `scripts/` assume a broader “Branch2” multi-service workspace and attempt to provision additional services (PostgreSQL, Redis, Elasticsearch); these directories are absent here and should be treated as legacy or TBD integrations.【F:scripts/setup.py†L1-L200】【F:scripts/init_databases.py†L1-L200】

## Limitations / TBDs
- SQLite helper functions for chats, document embeddings, and vector metadata referenced throughout the chat router and Milvus wrappers are missing, blocking chat persistence, RAG gating, and hybrid retrieval (`ensure_chat`, `append_message`, `fetch_chat_messages`, `save_document_with_vector`, etc.). Evidence expected in `database/database_functions.py` but absent.【F:api/chats_router.py†L14-L1204】【F:retreival/milvus_store.py†L1-L206】【F:database/database_functions.py†L14-L196】
- Document queries expect `documents` table columns (`embedding`, `metadata` JSON) that are not created by the current schema; migrations or table alterations are TBD.【F:main.py†L875-L1160】【F:database/database_functions.py†L38-L103】
- Several service managers (vector, graph, cache) depend on modules or collections (`vector_store`, structured DB access) that are not implemented in this repository; implementation status should be verified in corresponding directories (e.g., `core/database.py`).【F:data_sources/vector_manager.py†L1-L127】【F:core/database.py†L214-L351】
