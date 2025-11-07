# Services

## HTTP API Reference

### Diagnostics & Root
#### GET /
- **Summary**: Returns API banner, version, and `/docs` link.【F:main.py†L205-L213】
- **Auth**: None.
- **Request**: No parameters.
- **Responses**:
  - `200 OK`: `{"message": "Agentic RAG System API", "version": "2.0.0", "timestamp": ..., "docs": "/docs"}`.【F:main.py†L205-L213】
- **Side effects**: Increments `system_stats["api_calls"]` via middleware.【F:main.py†L183-L203】
- **Source**: `main.py`。【F:main.py†L205-L213】
- **cURL**:
  ```bash
  curl -s http://localhost:8000/
  ```

#### GET /__debug/sqlite
- **Summary**: Reports SQLite database path and schema info for diagnostics.【F:main.py†L150-L170】
- **Auth**: None.
- **Request**: No body.
- **Responses**:
  - `200 OK`: JSON containing `db_path_env`, `database_list`, `pragma`, or `error` if connection fails.【F:main.py†L150-L170】
- **Side effects**: Opens SQLite connection via `get_db_connection` and closes it; no writes.【F:main.py†L150-L170】【F:database/database_functions.py†L18-L24】
- **Source**: `main.py`。【F:main.py†L150-L170】
- **cURL**:
  ```bash
  curl -s http://localhost:8000/__debug/sqlite | jq
  ```

#### GET /__debug/routes
- **Summary**: Lists registered FastAPI route paths for inspection.【F:main.py†L172-L177】
- **Auth**: None.
- **Request**: None.
- **Responses**:
  - `200 OK`: Array of path strings.【F:main.py†L172-L177】
- **Side effects**: None beyond route introspection.【F:main.py†L172-L177】
- **Source**: `main.py`。【F:main.py†L172-L177】
- **cURL**:
  ```bash
  curl -s http://localhost:8000/__debug/routes | jq
  ```

#### OPTIONS /{path}
- **Summary**: CORS preflight responder returning `{ "message": "OK" }`.【F:main.py†L215-L218】
- **Auth**: None.
- **Request**: Path wildcard.
- **Responses**: `200 OK` with static message.【F:main.py†L215-L218】
- **Side effects**: None.
- **Source**: `main.py`。【F:main.py†L215-L218】
- **cURL**:
  ```bash
  curl -X OPTIONS http://localhost:8000/api/v1/ask
  ```

### Authentication
Pydantic request models live in `models/pydantic_models.py` for signup/login flows.【F:models/pydantic_models.py†L5-L59】 JWT utilities are in `auth/authentication.py` and Milvus user storage is accessed through `core/database.py` managers.【F:auth/authentication.py†L16-L92】【F:core/database.py†L456-L571】

#### POST /api/auth/google
- **Summary**: Validates a Google ID token, creates or fetches a user, and issues a JWT.【F:main.py†L252-L312】
- **Auth**: None; expects Google token in request body.
- **Request**: JSON `{"token": "<id_token>"}`.【F:main.py†L252-L269】
- **Responses**:
  - `200 OK`: `AuthResponse` payload with JWT, expiry, and user metadata.【F:main.py†L304-L312】【F:models/pydantic_models.py†L43-L59】
  - `400 Bad Request`: Missing token.【F:main.py†L256-L259】
  - `401/500`: OAuth verification errors.【F:main.py†L244-L249】【F:main.py†L314-L318】
- **Side effects**: Calls `verify_google_token`, upserts Milvus user, and updates system stats activity manager.【F:main.py†L221-L312】【F:core/database.py†L456-L571】
- **Source**: `main.py`。【F:main.py†L252-L312】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/auth/google \
    -H 'Content-Type: application/json' \
    -d '{"token":"<GOOGLE_ID_TOKEN>"}'
  ```

#### POST /api/auth/signup
- **Summary**: Registers a user with email/password, stores hashed credentials in Milvus, and returns JWT.【F:main.py†L320-L370】
- **Auth**: None.
- **Request**: `UserSignup` model (`name`, `email`, `password`, optional `description`).【F:models/pydantic_models.py†L18-L42】
- **Responses**:
  - `200 OK`: `AuthResponse` with user metadata.【F:main.py†L361-L368】
  - `400 Bad Request`: Email already registered.【F:main.py†L326-L330】
  - `500`: Milvus or hashing errors.【F:main.py†L347-L375】
- **Side effects**: Hashes password via `get_password_hash`, persists in Milvus users collection.【F:auth/authentication.py†L30-L43】【F:core/database.py†L456-L571】
- **Source**: `main.py`。【F:main.py†L320-L370】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/auth/signup \
    -H 'Content-Type: application/json' \
    -d '{"name":"Alice","email":"alice@example.com","password":"Secret123"}'
  ```

#### POST /api/auth/login
- **Summary**: Authenticates by email/password, returns JWT with stored profile fields.【F:main.py†L377-L423】
- **Auth**: None (credentials required).
- **Request**: `UserLogin` (`email`, `password`).【F:models/pydantic_models.py†L44-L59】
- **Responses**:
  - `200 OK`: `AuthResponse` with stored `created_at` if available.【F:main.py†L409-L417】
  - `401 Unauthorized`: Invalid credentials.【F:main.py†L381-L388】
  - `500`: Unexpected login errors.【F:main.py†L419-L423】
- **Side effects**: Reads Milvus user, verifies password via `verify_password`, updates activity via `milvus_system_manager`.【F:main.py†L381-L403】【F:auth/authentication.py†L25-L29】【F:core/database.py†L456-L571】
- **Source**: `main.py`。【F:main.py†L377-L423】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/auth/login \
    -H 'Content-Type: application/json' \
    -d '{"email":"alice@example.com","password":"Secret123"}'
  ```

#### GET /api/auth/me
- **Summary**: Returns authenticated user info from JWT and Milvus lookup.【F:main.py†L425-L439】
- **Auth**: `Authorization: Bearer <token>` header processed by `get_current_user`.【F:auth/authentication.py†L68-L104】
- **Responses**:
  - `200 OK`: JSON with `user_id`, `name`, `email`, description, timestamps, preferences.【F:main.py†L431-L439】
  - `401 Unauthorized`: Missing/invalid token.【F:main.py†L428-L429】
- **Side effects**: None beyond Milvus fetch in dependency.【F:auth/authentication.py†L68-L104】
- **Source**: `main.py`。【F:main.py†L425-L439】
- **cURL**:
  ```bash
  curl -H 'Authorization: Bearer <JWT>' http://localhost:8000/api/auth/me
  ```

#### PUT /api/auth/profile
- **Summary**: Updates profile fields, with Milvus delete/recreate fallback when direct updates fail.【F:main.py†L442-L518】
- **Auth**: Bearer token required via dependency.【F:main.py†L443-L446】【F:auth/authentication.py†L68-L104】
- **Request**: JSON with `name`, `email`, optional `description`, `picture` etc.【F:main.py†L454-L508】
- **Responses**:
  - `200 OK`: Confirmation plus user payload; success path or cached fallback.【F:main.py†L501-L518】
  - `400/404`: Validation or missing user errors.【F:main.py†L458-L476】
  - `401`: Not authenticated.【F:main.py†L448-L450】
- **Side effects**: Attempts to delete/recreate Milvus user document; optional SQLite fallback (table `users`) is assumed but schema not defined here (TBD).【F:main.py†L470-L518】
- **Source**: `main.py`。【F:main.py†L442-L518】
- **cURL**:
  ```bash
  curl -X PUT http://localhost:8000/api/auth/profile \
    -H 'Authorization: Bearer <JWT>' \
    -H 'Content-Type: application/json' \
    -d '{"name":"Alice","email":"alice@example.com"}'
  ```

#### PUT /api/auth/change-password
- **Summary**: Validates current password, hashes the new password, and recreates the Milvus user record.【F:main.py†L536-L596】
- **Auth**: Bearer token required.【F:main.py†L537-L540】
- **Request**: JSON with `currentPassword` and `newPassword` (>=8 chars).【F:main.py†L545-L556】
- **Responses**:
  - `200 OK`: Message confirming update (real or simulated fallback).【F:main.py†L584-L596】
  - `400`: Missing/invalid passwords or incorrect current password.【F:main.py†L549-L562】
  - `401`: Missing auth.【F:main.py†L542-L544】
- **Side effects**: Deletes and recreates Milvus user with new hash; returns simulated success if Milvus update fails.【F:main.py†L563-L595】
- **Source**: `main.py`。【F:main.py†L536-L596】
- **cURL**:
  ```bash
  curl -X PUT http://localhost:8000/api/auth/change-password \
    -H 'Authorization: Bearer <JWT>' \
    -H 'Content-Type: application/json' \
    -d '{"currentPassword":"Secret123","newPassword":"NewSecret456"}'
  ```

#### POST /api/auth/upload-avatar
- **Summary**: Accepts an image upload, returns a data URI placeholder for avatar (no persistence backend).【F:main.py†L604-L637】
- **Auth**: Bearer token required.【F:main.py†L604-L612】
- **Request**: Multipart form with `file`.
- **Responses**:
  - `200 OK`: `{"message": "Avatar uploaded successfully", "avatar_url": "data:..."}`.【F:main.py†L634-L636】
  - `400`: Non-image or >5 MB file.【F:main.py†L615-L621】
- **Side effects**: Converts to base64 string; no storage beyond response.
- **Source**: `main.py`。【F:main.py†L604-L637】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/auth/upload-avatar \
    -H 'Authorization: Bearer <JWT>' \
    -F 'file=@avatar.png'
  ```

### User Management

#### DELETE /api/users/{user_id}
- **Summary**: Deletes a user from Milvus and optional SQLite `users` table; returns summary of deletion locations.【F:main.py†L645-L712】
- **Auth**: None enforced (administrative endpoint without guard).【F:main.py†L645-L712】
- **Responses**:
  - `200 OK`: `{ "status": "success", "deleted_from": ["milvus", "database"?], "user_id": ... }` when removed.【F:main.py†L692-L704】
  - `404`: User not found in either store.【F:main.py†L705-L706】
- **Side effects**: Calls `milvus_user_manager.delete_user` and attempts SQLite deletion using helpers that are assumed but not defined (TBD).【F:main.py†L654-L688】
- **Source**: `main.py`。【F:main.py†L645-L712】
- **cURL**:
  ```bash
  curl -X DELETE http://localhost:8000/api/users/1234
  ```

#### GET /api/users
- **Summary**: Lists users from Milvus with SQLite fallback if Milvus empty/unavailable.【F:main.py†L714-L764】
- **Auth**: None.
- **Responses**:
  - `200 OK`: `{ "users": [...], "total": N, "message": ... }` with user objects from whichever backend responded.【F:main.py†L756-L760】
  - `500`: Errors fetching users.【F:main.py†L762-L764】
- **Side effects**: Accesses Milvus user manager and, on failure, queries SQLite `users` table (schema undefined in repo).【F:main.py†L720-L753】
- **Source**: `main.py`。【F:main.py†L714-L764】
- **cURL**:
  ```bash
  curl -s http://localhost:8000/api/users | jq
  ```

#### POST /api/users
- **Summary**: Creates a user record with minimal fields and stores in Milvus and optional SQLite fallback; returns `UserResponse`.【F:main.py†L766-L827】【F:models/pydantic_models.py†L33-L42】
- **Auth**: None.
- **Request**: `UserDetails` (`name`, `email`, optional `description`).【F:models/pydantic_models.py†L24-L32】
- **Responses**:
  - `200 OK`: `{"message": "User created successfully...", "user_id": ..., "created_at": ...}`.【F:main.py†L822-L826】
  - `500`: Creation errors.【F:main.py†L828-L830】
- **Side effects**: Generates UUID, stores in Milvus via `create_user`, attempts SQLite insert (table schema missing).【F:main.py†L770-L812】
- **Source**: `main.py`。【F:main.py†L766-L827】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/users \
    -H 'Content-Type: application/json' \
    -d '{"name":"Bob","email":"bob@example.com"}'
  ```

#### GET /api/v1/users/{user_id}/chat-history
- **Summary**: Fetches chat history summaries stored in Milvus for a user.【F:main.py†L1880-L1892】
- **Auth**: None enforced.
- **Responses**:
  - `200 OK`: `{ "user_id": ..., "chat_history": [...], "total_entries": n, "limit": limit }`.【F:main.py†L1884-L1891】
  - `500`: Errors retrieving history.【F:main.py†L1892-L1894】
- **Side effects**: Calls `milvus_user_manager.get_user_chat_history`; implementation located in `core/database.py` (requires Milvus connection).【F:main.py†L1880-L1892】【F:core/database.py†L514-L542】
- **Source**: `main.py`。【F:main.py†L1880-L1892】
- **cURL**:
  ```bash
  curl http://localhost:8000/api/v1/users/1234/chat-history
  ```

### Question Answering & Search

#### POST /api/v1/ask
- **Summary**: Primary QA endpoint; delegates to `ask_question_handler` which invokes unified question processing pipeline.【F:main.py†L833-L836】【F:endpoints/api_endpoints.py†L39-L118】
- **Auth**: None.
- **Request**: `QuestionRequest` with `question`, optional `context`, `web_search_enabled`.【F:models/pydantic_models.py†L5-L16】
- **Responses**:
  - `200 OK`: Response dict with `answer`, `confidence`, `sources_used`, etc. (enforced by handler).【F:endpoints/api_endpoints.py†L40-L112】
  - `400/500`: Validation or processing errors bubbled from handler.【F:endpoints/api_endpoints.py†L60-L113】
- **Side effects**: Updates `system_stats`, clears embedding cache, may trigger Milvus retrieval, web search, or web scraping depending on context.【F:endpoints/api_endpoints.py†L40-L118】【F:processing/core_processing.py†L59-L200】
- **Source**: `main.py` & `endpoints/api_endpoints.py`.
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/ask \
    -H 'Content-Type: application/json' \
    -d '{"question":"What is the system status?"}'
  ```

#### GET /api/v1/stats
- **Summary**: Returns aggregated metrics (uptime, cache rate, Milvus status) from handler.【F:main.py†L839-L843】【F:endpoints/api_endpoints.py†L118-L235】
- **Auth**: None.
- **Responses**: JSON containing:
  - top-level metrics such as `total_questions_processed`, `average_response_time`, `cache_hit_rate`, `documents_count`, `system_health`, `status_message`, and Milvus flags.【F:endpoints/api_endpoints.py†L205-L223】
  - nested `detailed_stats` with `cache_hits`, `cache_misses`, `total_response_time_ms`, and current cache size.【F:endpoints/api_endpoints.py†L226-L233】
- **Side effects**: Calls Milvus store and `system_stats` aggregator; persists a `current_stats` snapshot via `milvus_system_manager` when Milvus is reachable.【F:endpoints/api_endpoints.py†L130-L204】
- **Source**: `main.py` & `endpoints/api_endpoints.py`.
- **cURL**:
  ```bash
  curl http://localhost:8000/api/v1/stats | jq
  ```

#### GET /api/v1/documents
- **Summary**: Lists stored documents through handler (Milvus preferred, SQLite/in-memory fallback).【F:main.py†L845-L849】【F:endpoints/api_endpoints.py†L242-L323】
- **Auth**: None.
- **Responses**: `{ "documents": [...], "total": N, "message": "..." }` where each document includes `id`, `title`, `content`, `summary`, parsed `metadata`, estimated size, and status flags.【F:endpoints/api_endpoints.py†L249-L265】【F:endpoints/api_endpoints.py†L295-L323】
- **Side effects**: Attempts Milvus query, then SQLite select on `documents`, then falls back to `in_memory_documents` if persistence unavailable.【F:endpoints/api_endpoints.py†L245-L323】
- **Source**: `main.py` & `endpoints/api_endpoints.py`.
- **cURL**:
  ```bash
  curl http://localhost:8000/api/v1/documents | jq
  ```

#### DELETE /api/v1/documents/{document_id}
- **Summary**: Removes a document via handler which attempts Milvus and SQLite deletes.【F:main.py†L851-L854】【F:endpoints/api_endpoints.py†L268-L323】
- **Auth**: None.
- **Responses**: `200 OK` with deletion message, or errors if ID not found.【F:endpoints/api_endpoints.py†L268-L323】
- **Side effects**: Invokes `delete_document_handler` referencing SQLite connection and Milvus utilities.【F:endpoints/api_endpoints.py†L268-L323】
- **Source**: `main.py` & `endpoints/api_endpoints.py`.
- **cURL**:
  ```bash
  curl -X DELETE http://localhost:8000/api/v1/documents/123
  ```

#### POST /api/v1/search
- **Summary**: Executes hybrid document search with chunk-based analysis and optional embeddings.【F:main.py†L857-L949】
- **Auth**: None.
- **Request**: JSON with `query`, optional `max_results`, `relevance_threshold`, `include_chunks`, `chunk_size`, `search_type`.【F:main.py†L861-L869】
- **Responses**:
  - `200 OK`: Search report containing `results` with `relevance_score`, `semantic_similarity`, `keyword_score`, excerpts, and optional `chunk_analysis.top_chunks` summary.【F:main.py†L920-L961】【F:main.py†L934-L952】
  - `400`: Missing query or invalid embedding results.【F:main.py†L861-L915】
- **Side effects**: Reads from SQLite `documents` table (expects `embedding` column not present in schema, flagged as TBD), may call OpenAI for embeddings per chunk, and increments similarity search counters.【F:main.py†L871-L939】【F:endpoints/api_endpoints.py†L39-L118】
- **Source**: `main.py`。【F:main.py†L857-L949】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/search \
    -H 'Content-Type: application/json' \
    -d '{"query":"vector search design"}' | jq
  ```

#### GET /api/v1/documents/{document_id}/content
- **Summary**: Retrieves document content from SQLite or in-memory fallback.【F:main.py†L1001-L1074】
- **Auth**: None.
- **Responses**:
  - `200 OK`: Document payload with metadata and timestamps.【F:main.py†L1044-L1071】
  - `404`: Document missing.【F:main.py†L1061-L1074】
- **Side effects**: Executes SQLite queries (assumes `metadata` JSON) with fallback to `in_memory_documents` list.【F:main.py†L1007-L1071】
- **Source**: `main.py`。【F:main.py†L1001-L1074】
- **cURL**:
  ```bash
  curl http://localhost:8000/api/v1/documents/1/content | jq
  ```

#### POST /api/v1/scrape-website
- **Summary**: Scrapes a single URL, generates embedding, and stores result via handler logic with Milvus/SQLite/memory fallbacks.【F:main.py†L1083-L1194】
- **Auth**: None.
- **Request**: `WebScrapingRequest` (`url`, optional `title`, `summary`, `metadata`).【F:models/pydantic_models.py†L61-L96】
- **Responses**: Success payload includes `document_id`, `title`, embedding dimension, storage flags, and metadata; errors raised for invalid URLs, embedding failures, or scraping issues.【F:main.py†L1110-L1189】
- **Side effects**: Calls `scrape_website_content`, `generate_openai_embedding`, increments `system_stats`, writes to Milvus and SQLite when available, and falls back to in-memory storage when persistence fails.【F:main.py†L1089-L1188】【F:web/scraping_helpers.py†L19-L198】
- **Source**: `main.py`。【F:main.py†L1083-L1194】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/scrape-website \
    -H 'Content-Type: application/json' \
    -d '{"url":"https://example.com"}'
  ```

#### POST /api/v1/bulk-scrape-website
- **Summary**: Invokes bulk scraping agent with optional metadata; similar persistence pattern.【F:main.py†L1197-L1438】
- **Auth**: None.
- **Request**: `BulkWebScrapingRequest` schema with `url`, `max_pages`, `crawl_internal_links`, etc.【F:models/pydantic_models.py†L98-L129】
- **Responses**: Complex result summarizing scraped documents and embeddings; falls back to memory when persistence fails.【F:main.py†L1233-L1438】
- **Side effects**: Uses `bulk_scrape_website`, `generate_openai_embedding`, `store_scraped_document`; relies on missing SQLite helpers for chat-aware storage (TBD).【F:main.py†L1212-L1438】【F:web/scraping_helpers.py†L169-L198】
- **Source**: `main.py`。【F:main.py†L1197-L1438】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/bulk-scrape-website \
    -H 'Content-Type: application/json' \
    -d '{"url":"https://example.com","max_pages":2}'
  ```

#### POST /api/v1/upload-file
- **Summary**: Accepts PDF/DOCX/TXT/JSON uploads, extracts content, generates embeddings, and persists via `store_scraped_document` or fallbacks.【F:main.py†L1682-L1858】
- **Auth**: None.
- **Request**: Multipart with `file`, optional `title`, `summary`, `metadata` JSON string.【F:main.py†L1683-L1735】
- **Responses**: Success payload includes storage location (Milvus, database, or memory), `embedding_dimensions`, metadata echo, and status flags; raises errors for unsupported MIME types, empty content, or embedding failures.【F:main.py†L1689-L1855】
- **Side effects**: Uses file extraction helpers (`files/file_processing.py`), OpenAI embeddings, updates `system_stats`, writes to SQLite `documents` (embedding column TBD), and appends to `in_memory_documents` when persistence unavailable.【F:main.py†L1690-L1855】【F:files/file_processing.py†L1-L200】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/upload-file \
    -F 'file=@sample.pdf'
  ```

#### PUT /api/v1/settings/system-prompt
- **Summary**: Stores a custom system prompt in Milvus user preferences via delete/recreate pattern.【F:main.py†L1551-L1600】
- **Auth**: Bearer token required.【F:main.py†L1552-L1558】
- **Request**: JSON with `system_prompt` string (<=2000 chars).【F:main.py†L1553-L1568】
- **Responses**: `200 OK` with prompt metadata or `400/401/404/500` on validation/missing user/errors.【F:main.py†L1592-L1604】
- **Side effects**: Updates Milvus user preferences; relies on `milvus_user_manager`.【F:main.py†L1570-L1599】【F:core/database.py†L456-L571】
- **cURL**:
  ```bash
  curl -X PUT http://localhost:8000/api/v1/settings/system-prompt \
    -H 'Authorization: Bearer <JWT>' \
    -H 'Content-Type: application/json' \
    -d '{"system_prompt":"Always respond politely."}'
  ```

#### GET /api/v1/settings/system-prompt
- **Summary**: Returns current system prompt and metadata from JWT user preferences.【F:main.py†L1612-L1628】
- **Auth**: Bearer token.【F:main.py†L1612-L1617】
- **Responses**: JSON with prompt, timestamps, boolean flags.【F:main.py†L1623-L1628】
- **Side effects**: None beyond dependency.
- **cURL**:
  ```bash
  curl -H 'Authorization: Bearer <JWT>' http://localhost:8000/api/v1/settings/system-prompt
  ```

#### DELETE /api/v1/settings/system-prompt
- **Summary**: Removes stored system prompt from Milvus preferences.【F:main.py†L1635-L1674】
- **Auth**: Bearer token.【F:main.py†L1635-L1640】
- **Responses**: `200 OK` message or errors if user missing or Milvus update fails.【F:main.py†L1668-L1674】
- **Side effects**: Milvus delete/recreate with prompt removed.【F:main.py†L1654-L1674】
- **cURL**:
  ```bash
  curl -X DELETE http://localhost:8000/api/v1/settings/system-prompt \
    -H 'Authorization: Bearer <JWT>'
  ```

### Health & Real-time

#### GET /health
- **Summary**: Composite health report covering API, Milvus, WebSocket connections, memory cache, and stats.【F:main.py†L1898-L1949】
- **Auth**: None.
- **Responses**: JSON `status`, `services`, `metrics`, optional Milvus diagnostics.【F:main.py†L1900-L1945】
- **Side effects**: Calls `ensure_milvus_connection`, lists Milvus collections, clears expired cache entries.【F:main.py†L1916-L1943】【F:core/database.py†L24-L200】
- **cURL**:
  ```bash
  curl http://localhost:8000/health | jq
  ```

#### WebSocket /ws
- **Summary**: Bidirectional channel for question answering with live progress updates; uses `ConnectionManager`.【F:main.py†L1955-L1961】【F:endpoints/websocket_handlers.py†L19-L170】
- **Auth**: None enforced.
- **Messages**:
  - Send `{"type":"question","question":"..."}` to trigger processing.
  - Receive initial `connected` confirmation, followed by `processing_update`, `answer`, `error`, and `pong` events.【F:endpoints/websocket_handlers.py†L84-L151】
- **Side effects**: Clears embedding cache per connection, increments `system_stats["websocket_connections"]`, delegates to `process_question_unified_with_updates`.【F:endpoints/websocket_handlers.py†L77-L143】【F:processing/core_processing.py†L44-L134】
- **Source**: `main.py`, `endpoints/websocket_handlers.py`.
- **Example (websocket-client Python)**:
  ```python
  import websockets, asyncio, json
  async def chat():
      async with websockets.connect('ws://localhost:8000/ws') as ws:
          await ws.send(json.dumps({"type":"question","question":"Hi"}))
          async for message in ws:
              print(message)
  asyncio.run(chat())
  ```

### Chat API (SSE and REST under /api)
Models for chat requests/responses live in `models/chat.py`, though imports such as `Enum` are currently missing (TBD).【F:models/chat.py†L1-L102】 Chat routes rely on SQLite helpers (`ensure_chat`, `append_message`, etc.) that are not implemented in `database/database_functions.py`, so persistence currently fails (marked TBD).【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】

#### POST /api/chat/ask
- **Summary**: Non-streaming chat completion storing user and assistant messages with citations.【F:api/chats_router.py†L194-L373】
- **Auth**: None.
- **Request**: `AskRequest` with `question`, optional `chat_id`, `user_id`, `system_prompt`, `context`, and web search flags.【F:models/chat.py†L66-L88】
- **Responses**: `AskResponse` containing `chat_id`, stored user/assistant messages, `model_used`, `tokens_used`, and truncation metadata (`input_truncated`, `original_question_length`, `used_question_length`).【F:api/chats_router.py†L330-L379】【F:models/chat.py†L90-L112】
- **Side effects**: Expects to ensure chat, append messages, run web search, call OpenAI completions, auto-title chat, and trim history. Missing SQLite helpers prevent actual storage (TBD).【F:api/chats_router.py†L214-L370】
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/api/chat/ask \
    -H 'Content-Type: application/json' \
    -d '{"question":"Hello"}'
  ```

#### POST /api/chat/ask/stream
- **Summary**: SSE streaming variant that emits start/meta/delta/done events during chat completion.【F:api/chats_router.py†L394-L727】
- **Auth**: None.
- **Request**: `AskRequest` JSON (same as non-streaming).【F:api/chats_router.py†L394-L422】【F:models/chat.py†L66-L88】
- **Responses**: Event stream where:
  - `start` announces `{chat_id, request_id, model_used?}` when chat bootstrap completes.【F:api/chats_router.py†L400-L405】【F:api/chats_router.py†L618-L626】
  - `meta` conveys stages such as `rag_gate`, `web_search`, `embeddings` with counts like `documents_attached` and preview web sources.【F:api/chats_router.py†L451-L505】
  - `saved` publishes persisted message IDs; `delta` emits incremental assistant text; `done` summarizes tokens, lengths, and citations; `error` reports failures.【F:api/chats_router.py†L618-L717】
- **Side effects**: Same as non-streaming plus incremental OpenAI streaming; depends on missing SQLite helpers (TBD).【F:api/chats_router.py†L427-L706】
- **cURL**:
  ```bash
  curl -N -X POST http://localhost:8000/api/chat/ask/stream \
    -H 'Content-Type: application/json' \
    -d '{"question":"Stream please"}'
  ```

#### GET /api/chat/{chat_id}/messages
- **Summary**: Returns ordered chat messages.【F:api/chats_router.py†L732-L745】
- **Auth**: None.
- **Responses**: `{ "chat_id": ..., "messages": [...], "count": n }` based on `_massage_message_row` helper.【F:api/chats_router.py†L732-L745】
- **Side effects**: Calls `fetch_chat_messages` (missing implementation).【F:api/chats_router.py†L735-L745】

#### GET /api/chats
- **Summary**: Lists chats with pagination and optional user filter.【F:api/chats_router.py†L752-L779】
- **Auth**: None.
- **Query params**: `limit` (1-200), `offset` (>=0), optional `user_id`.【F:api/chats_router.py†L753-L768】
- **Responses**: `{ "items": [...], "total": N, "limit": limit, "offset": offset }` from SQLite helpers.【F:api/chats_router.py†L770-L779】
- **Side effects**: Calls `list_chats` (missing) and raw count query.【F:api/chats_router.py†L760-L779】

#### DELETE /api/chat/{chat_id}
- **Summary**: Deletes a chat via SQLite helper.【F:api/chats_router.py†L787-L796】
- **Auth**: None.
- **Responses**: `{"deleted": true, "chat_id": ...}` or `404` if not found.【F:api/chats_router.py†L792-L795】

#### POST /api/chat/{chat_id}/auto-title
- **Summary**: Forces deterministic title generation for an untitled chat by inspecting first user message.【F:api/chats_router.py†L803-L876】
- **Auth**: None.
- **Responses**: `{ "chat_id": ..., "title": ... }` or `404` if chat missing.【F:api/chats_router.py†L823-L876】

#### GET /api/chat/{chat_id}/export
- **Summary**: Exports chat messages as JSON payload for download.【F:api/chats_router.py†L877-L915】
- **Auth**: None.
- **Responses**: `{ "chat_id": ..., "messages": [...], "exported_at": ... }` (requires helper functions).【F:api/chats_router.py†L877-L915】

#### GET /api/my/chats
- **Summary**: Convenience wrapper for listing chats belonging to current user (based on query parameter).【F:api/chats_router.py†L917-L964】
- **Auth**: None enforced (user_id expected in query).【F:api/chats_router.py†L917-L964】

#### POST /api/chat/{chat_id}/title
- **Summary**: Sets a manual title for a chat.【F:api/chats_router.py†L966-L996】
- **Auth**: None.
- **Request**: JSON `{"title": "..."}` (non-empty).【F:api/chats_router.py†L969-L994】
- **Responses**: Confirmation payload or validation errors.【F:api/chats_router.py†L969-L996】

#### GET /api/chat/{chat_id}/messages/paged
- **Summary**: Returns chat messages with pagination metadata.【F:api/chats_router.py†L999-L1091】
- **Auth**: None.
- **Query params**: `page`, `page_size`.【F:api/chats_router.py†L1000-L1091】
- **Responses**: `{ "chat_id": ..., "messages": [...], "pagination": {...} }` using helpers (missing).【F:api/chats_router.py†L1035-L1087】

#### POST /api/documents (chat scope)
- **Summary**: Stores a document associated with a chat, returning `CreateDocumentResponse` (requires missing persistence helpers).【F:api/chats_router.py†L1096-L1153】【F:models/chat.py†L114-L139】
- **Auth**: None.
- **Request**: `RequestDocument` schema supporting legacy alias keys (Document_Title, etc.).【F:models/chat.py†L118-L152】
- **Responses**: `CreateDocumentResponse` with generated UUID and chat association.【F:api/chats_router.py†L1134-L1153】【F:models/chat.py†L154-L162】

#### GET /api/documents/by-chat/{chat_id}
- **Summary**: Lists documents attached to a chat via helper queries (missing).【F:api/chats_router.py†L1155-L1203】

#### GET /api/documents/{doc_id}/content (chat scope)
- **Summary**: Retrieves chat-specific document content; overlaps with global `/api/v1/documents/...` but scoped to chat IDs.【F:api/chats_router.py†L1204-L1204】

> **TBD**: All chat endpoints rely on SQLite chat schema helpers (`init_chat_schema`, `ensure_chat`, `append_message`, `fetch_chat_messages`, `list_chats`, `delete_chat`, `trim_chat_history`, `has_documents_for_chat`, `count_documents_for_chat`) which are referenced but not defined in `database/database_functions.py`. Until implemented, chat read/write operations will raise errors. Expected implementations should live in `database/database_functions.py` or a dedicated chat module.【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】

### Background Jobs / Schedulers
No cron jobs, Celery workers, or other background schedulers are defined in the repository. Startup logic is limited to a FastAPI `lifespan` context manager that initializes Milvus connections and chat schema (the latter also depends on missing helpers).【F:lifecycle/app_lifecycle.py†L1-L26】【F:main.py†L1862-L1880】

### Domain Services / Use Cases
- **process_question_unified / process_question_unified_with_updates**: Central question-answering orchestrators handling context merging, web scraping, web search, retrieval, and synthesis, returning structured answers and emitting progress callbacks.【F:processing/core_processing.py†L39-L200】
- **SynthesisAgent**: Ranks documents, calculates confidence, and prepares quality metrics using embeddings and cosine similarity.【F:synthesis/synthesis_agent.py†L20-L193】
- **Retrieval service (`services/retreival.py`)**: Adds documents to SQLite, deduplicates chunks, generates embeddings, and queries embeddings with cosine fallback when Milvus unavailable. Note synchronous calls to async OpenAI embedding are currently inconsistent (TBD).【F:services/retreival.py†L38-L146】
- **QueryOrchestrator**: Higher-level pipeline combining analysis, routing, multi-source querying, caching, and synthesis; depends on LangChain and custom managers for vector, graph, search, and cache data sources.【F:api/orchestrator.py†L1-L200】

### External Clients / Wrappers
- **OpenAI Integration**: `ai/openai_integration.py` encapsulates embedding creation, chat completions, streaming responses, caching, and retry logic; requires `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`.【F:ai/openai_integration.py†L13-L206】【F:ai/openai_integration.py†L236-L384】
- **Milvus Manager**: `core/database.py` handles connecting to Milvus, ensuring collections, performing similarity search, storing documents, users, chats, and system stats.【F:core/database.py†L16-L571】
- **Web Scraping**: `web/scraping_helpers.py` provides HTML parsing, metadata extraction, and internal link discovery used by scraping endpoints.【F:web/scraping_helpers.py†L19-L198】
- **Vector/Search Managers**: `data_sources/vector_manager.py`, `data_sources/search_manager.py`, and `data_sources/cache_manager.py` define interfaces for orchestrated pipelines, though some dependencies (`vector_store`, graph DB) are placeholders pending implementation (TBD).【F:data_sources/vector_manager.py†L1-L127】【F:data_sources/search_manager.py†L1-L200】

## Background Notes
- Multiple endpoints attempt SQLite writes to tables/columns (`users`, `documents.embedding`) not created by existing schema helpers; data persistence will fail until migrations are added (TBD).【F:database/database_functions.py†L38-L103】【F:main.py†L871-L1188】
- Chat APIs require unimplemented SQLite helpers before they can operate correctly (TBD).【F:api/chats_router.py†L14-L1204】
