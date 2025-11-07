# Data Models

## Databases & Storage Engines
- **SQLite**: Default relational store accessed via `database/database_functions.py`; `get_db_connection()` reads `SQLITE_DB_PATH` (default `app.db`). Tables defined include `documents`, `chunks`, and `chunk_embeddings` with foreign keys and dedupe constraints.【F:database/database_functions.py†L14-L112】
- **Milvus (vector database)**: Connection managed by `core/database.py`. Collections created during initialization include `Documents`, `Users`, `UserChats`, `SystemStats`, and `DocumentsV2`, each with defined schema and indices for embeddings.【F:core/database.py†L16-L210】【F:core/database.py†L214-L351】

## SQLite Schema
| Table | Fields | Notes |
| --- | --- | --- |
| `documents` | `id TEXT PRIMARY KEY`, `title TEXT NOT NULL`, `content TEXT`, `summary TEXT`, `metadata TEXT`, `source_path TEXT`, `created_at TEXT NOT NULL`, `updated_at TEXT NOT NULL` | `metadata` stored as JSON string; `source_path` optional.【F:database/database_functions.py†L38-L64】 |
| `chunks` | `id TEXT PRIMARY KEY`, `document_id TEXT NOT NULL`, `chunk_index INTEGER NOT NULL`, `text TEXT NOT NULL`, `chunk_hash TEXT NOT NULL UNIQUE`, `start_char INTEGER`, `end_char INTEGER`, `created_at TEXT NOT NULL` | Foreign key `document_id` references `documents` with cascade delete; `chunk_hash` deduplicates inserts.【F:database/database_functions.py†L66-L87】 |
| `chunk_embeddings` | `chunk_id TEXT PRIMARY KEY`, `embedding BLOB NOT NULL`, `dim INTEGER NOT NULL` | Stores embeddings serialized as JSON blobs; references `chunks` on delete cascade.【F:database/database_functions.py†L89-L103】 |

### Helper Behaviors
- `upsert_document` JSON-serializes metadata, preserving existing `created_at` on replace; returns document ID.【F:database/database_functions.py†L45-L60】
- `insert_chunk` hashes normalized text to avoid duplicates, returning metadata indicating deduped status.【F:database/database_functions.py†L66-L83】
- `upsert_chunk_embedding` persists embeddings and dimension, with conversion helpers `_to_blob` / `_from_blob`.【F:database/database_functions.py†L95-L119】
- Bulk fetch helpers provide embeddings for cosine similarity fallback (`fetch_all_embeddings`).【F:database/database_functions.py†L124-L139】

### Missing / TBD
- Chat-specific helpers (`ensure_chat`, `append_message`, `fetch_chat_messages`, `list_chats`, `delete_chat`, `trim_chat_history`, `has_documents_for_chat`, `count_documents_for_chat`, etc.) referenced by chat routes are not implemented in this module, leaving chat persistence incomplete.【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】
- Multiple API endpoints expect `documents` to include an `embedding` column and a `users` table for profile storage, but the schema does not define these structures. Migrations or schema updates are required to align with usage in `main.py` search, upload, and user endpoints (TBD).【F:main.py†L875-L1188】【F:main.py†L766-L827】【F:database/database_functions.py†L38-L103】

## Milvus Collections
| Collection | Fields | Purpose |
| --- | --- | --- |
| `Documents` | Auto ID primary key, `doc_id`, `title`, `content`, `summary`, `metadata`, `embedding`, `created_at`, `file_size`, `content_type` | Legacy full-document storage with IVF_FLAT cosine index.【F:core/database.py†L32-L118】 |
| `Users` | `user_id` (PK), `name`, `email`, `password_hash`, `description`, `created_at`, `last_active`, `preferences`, `dummy_vector` | Stores user profiles and auth metadata; `dummy_vector` enables vector operations despite non-embedding use.【F:core/database.py†L118-L158】 |
| `UserChats` | Auto ID, `chat_id`, `user_id`, `question`, `answer`, `confidence`, `sources_used`, `reasoning`, `processing_time`, `timestamp`, `session_id`, `dummy_vector` | Intended for chat history archiving; recreated on schema mismatch.【F:core/database.py†L158-L198】 |
| `SystemStats` | Auto ID, `stat_key`, `stat_value`, `stat_type`, `timestamp`, `metadata`, `dummy_vector` | Persists metrics snapshots and cache metadata.【F:core/database.py†L198-L210】【F:core/database.py†L394-L424】 |
| `DocumentsV2` | `doc_id` (PK), `chat_id`, `title`, `summary`, `metadata_json`, `created_at`, `embedding` (FloatVector) | Preferred lightweight vector store for chat-associated documents, indexed with HNSW inner product.【F:core/database.py†L214-L295】 |

### Operations
- **Upserts**: `MilvusStore.upsert_document_v2` ensures collection/index, then upserts or inserts with manual delete fallback; flushes after each operation.【F:core/database.py†L214-L275】
- **Similarity Search**: `similarity_search` (legacy) and `similarity_search_v2` use cosine/IP metrics with optional chat filters, converting results to dictionaries with metadata.【F:core/database.py†L276-L366】
- **User Manager**: `MilvusUserManager` handles create/get/delete operations via Milvus queries and deletes; operations fail gracefully if `milvus_connected` is false.【F:core/database.py†L456-L571】
- **System Manager**: Stores stats snapshots and user activity in Milvus or in-memory cache.【F:core/database.py†L394-L424】【F:core/database.py†L571-L690】

### Missing / TBD
- Several Milvus utility references in `retreival/milvus_store.py` expect helper functions such as `upsert_embeddings`, `fetch_chunk_texts_by_ids`, and `search_embeddings` to exist in `database/database_functions.py`, but these functions are not defined. Confirm intended implementation or remove unused imports (TBD).【F:retreival/milvus_store.py†L1-L206】【F:database/database_functions.py†L14-L196】
- Vector store managers under `data_sources/` reference `vector_store` methods (`create_embedding_table`, `similarity_search`, `insert_embedding`, etc.) that are not implemented in the provided codebase, blocking orchestrator flows (TBD).【F:data_sources/vector_manager.py†L31-L461】【F:core/database.py†L16-L351】

## Pydantic Schemas
| Schema | Fields | Used by |
| --- | --- | --- |
| `QuestionRequest` | `question: str`, optional `context: Dict[str, Any]`, `web_search_enabled: bool` | `/api/v1/ask` question submission.【F:models/pydantic_models.py†L5-L16】 |
| `UserSignup`, `UserLogin`, `UserDetails`, `UserResponse`, `AuthResponse` | Email/password validation, hashed password requirements, response payloads | Auth endpoints under `/api/auth/*`.【F:models/pydantic_models.py†L18-L59】 |
| `WebScrapingRequest`, `BulkWebScrapingRequest` | URL validation, crawl options | Scraping endpoints `/api/v1/scrape-website`, `/api/v1/bulk-scrape-website`.【F:models/pydantic_models.py†L61-L129】 |
| `AskRequest`, `AskResponse`, `Message`, `Chat`, `ChatListResponse`, `RequestDocument`, `CreateDocumentResponse` | Chat-related payloads for REST/SSE endpoints (requires `Enum` import fix). | `/api/chat/*` routes.【F:models/chat.py†L1-L162】 |

Validation uses `field_validator` for non-empty strings, URL normalization, and ranges (e.g., `max_pages` between 1 and 100).【F:models/pydantic_models.py†L5-L129】

## In-Memory Structures
- `system_stats`: Tracks processed questions, cache hits, uptime, uploaded documents, API calls, and WebSocket connections for diagnostics endpoints.【F:main.py†L106-L205】【F:endpoints/api_endpoints.py†L39-L214】
- `in_memory_documents`: List storing fallback documents when SQLite or Milvus is unavailable; read by search and content endpoints.【F:main.py†L106-L118】【F:main.py†L891-L1071】
- `embedding_cache`: Request-scoped dictionary preventing duplicate OpenAI embedding calls during processing.【F:ai/openai_integration.py†L72-L104】【F:processing/core_processing.py†L59-L110】

## Example Records
- **Document (SQLite)**: After scraping, expected record resembles `{ "id": "<uuid>", "title": "Example", "content": "...", "summary": "...", "metadata": "{...}", "source_path": null, "created_at": "2025-01-01T00:00:00+00:00", "updated_at": "..." }` based on insertion logic in `store_scraped_document` fallback; embeddings stored either in `chunk_embeddings` or `documents.embedding` (once schema updated).【F:main.py†L1112-L1171】【F:database/database_functions.py†L45-L103】
- **Milvus DocumentV2 row**: Upsert writes `[doc_id, chat_id, title, summary, json.dumps(metadata), created_at, embedding]` to the collection; the embedding dimension defaults to 1536 (OpenAI text-embedding-3-small).【F:core/database.py†L214-L275】【F:ai/openai_integration.py†L44-L102】

## Data Retention & Deletion
- SQLite cascades delete chunks/embeddings when `documents` entries are removed.【F:database/database_functions.py†L66-L103】
- Milvus deletion uses `coll.delete(expr=...)` or partition-scoped deletes for document embeddings; user updates frequently delete and recreate records due to lack of update support.【F:core/database.py†L214-L351】【F:core/database.py†L456-L571】
- No soft-delete flags or archival logic are implemented.

## Vector Embeddings
- Embeddings generated via OpenAI are cached in-memory per request (`embedding_cache`) and stored either in SQLite blob (`chunk_embeddings`) or Milvus float vectors.【F:ai/openai_integration.py†L13-L120】【F:database/database_functions.py†L95-L139】【F:core/database.py†L214-L366】
- Vector dimension defaults to 1536 via environment variable `EMBEDDING_DIM` if provided.【F:retreival/index_milvus.py†L19-L52】【F:retreival/milvus_store.py†L17-L103】

## Migrations & Tooling
- No Alembic or migration framework is present. Schema creation occurs at runtime through helper functions such as `ensure_tables`. Developers must manage schema evolution manually (TBD).【F:database/database_functions.py†L38-L103】
- `scripts/init_databases.py` references PostgreSQL, Redis, and Elasticsearch initialization but these stores are not integrated with application code; treat as legacy script until confirmed.【F:scripts/init_databases.py†L1-L200】
