# Agentic RAG System

## Project Summary
FastAPI application providing question answering, chat (REST, SSE, WebSocket), document ingestion, and user management backed by Milvus vector search with SQLite fallbacks and OpenAI completions.【F:main.py†L123-L213】【F:processing/core_processing.py†L39-L200】【F:core/database.py†L16-L366】 Chat persistence helpers are referenced but not implemented, so chat features currently require additional database work (see Troubleshooting).【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】

## Tech Stack
- FastAPI with CORS, SSE, and WebSocket support.【F:main.py†L123-L727】【F:endpoints/websocket_handlers.py†L19-L170】
- Milvus (Documents, DocumentsV2, Users, UserChats, SystemStats) for vector storage; SQLite for fallback relational storage.【F:core/database.py†L16-L351】【F:database/database_functions.py†L38-L112】
- OpenAI embeddings and chat completions with retry logic and streaming support.【F:ai/openai_integration.py†L13-L384】
- Web scraping utilities (requests + BeautifulSoup) for content ingestion; file extraction for PDF/DOCX/TXT/JSON uploads.【F:web/scraping_helpers.py†L19-L198】【F:files/file_processing.py†L1-L200】
- JWT authentication with bcrypt hashing and optional Google OAuth login.【F:auth/authentication.py†L16-L92】【F:main.py†L221-L423】

## Getting Started
```bash
# Optional: bring up Milvus dependencies
docker compose up -d

# Install Python dependencies (create virtualenv first)
pip install fastapi uvicorn pymilvus openai passlib[bcrypt] google-auth google-auth-oauthlib \
            structlog aiohttp python-docx2txt PyPDF2 beautifulsoup4 requests python-dotenv
# (add additional packages as imports indicate; requirements file is TBD)

# Configure environment
export OPENAI_API_KEY=sk-...
export JWT_SECRET_KEY=change-me
# Optional overrides
export SQLITE_DB_PATH=agentic_rag.db
export MILVUS_HOST=localhost
export MILVUS_PORT=19530

# Run the API
uvicorn main:app --reload
```
- API docs: http://localhost:8000/docs
- SSE stream endpoint: POST http://localhost:8000/api/chat/ask/stream
- WebSocket endpoint: ws://localhost:8000/ws

## Configuration
| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for embeddings and completions.【F:ai/openai_integration.py†L44-L78】 | none |
| `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` | Override chat/embedding models (default `gpt-5`, `text-embedding-3-small`).【F:ai/openai_integration.py†L44-L68】 | as shown |
| `SQLITE_DB_PATH` | SQLite file path for document storage and chat (once helpers exist).【F:database/database_functions.py†L14-L24】 | `app.db` |
| `MILVUS_HOST`, `MILVUS_PORT` | Milvus connection info.【F:config/settings.py†L10-L17】 | `localhost`, `19530` |
| `JWT_SECRET_KEY` | HS256 signing secret for tokens.【F:auth/authentication.py†L16-L24】 | `your-secret-key-change-this-in-production` |
| `GOOGLE_CLIENT_ID` | Required for Google OAuth login.【F:main.py†L221-L312】 | none |
| `PORT` | Override app port (defaults to 8000).【F:main.py†L1958-L1961】 | `8000` |
| `EMBEDDING_DIM`, `MILVUS_COLLECTION`, etc. | Customize Milvus client behavior.【F:retreival/index_milvus.py†L9-L52】【F:retreival/milvus_store.py†L17-L103】 | see code |
| `.env.example` optional keys (Stripe, Redis, Kafka, alt DBs) | Documented for broader platform integrations; unused unless corresponding services implemented.【F:.env.example†L1-L154】 | none |

## Useful Commands & Scripts
- `docker compose up -d` – start Milvus standalone, etcd, MinIO, and Attu explorer for local testing.【F:docker-compose.yml†L11-L98】
- `uvicorn main:app --reload` – run development server with live reload.【F:main.py†L1958-L1961】
- `python scripts/init_databases.py` – legacy setup script that attempts to initialize PostgreSQL, Redis, and Elasticsearch; those services are not wired into this codebase (TBD).【F:scripts/init_databases.py†L1-L200】

## API Quick Peek
```bash
# Ask a question
curl -X POST http://localhost:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many documents are loaded?"}'

# Retrieve stats snapshot
curl http://localhost:8000/api/v1/stats | jq

# Upload a text file
curl -X POST http://localhost:8000/api/v1/upload-file \
  -F 'file=@notes.txt'

# Check service health
curl http://localhost:8000/health | jq

# Create a user via signup
curl -X POST http://localhost:8000/api/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"name":"Alice","email":"alice@example.com","password":"Secret123"}'

# Stream chat response (SSE)
curl -N -X POST http://localhost:8000/api/chat/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"Hello"}'
```
(See `services.md` for exhaustive endpoint documentation.)

## Troubleshooting
- **Chat endpoints fail**: `api/chats_router.py` expects SQLite helpers (`ensure_chat`, `append_message`, etc.) that are not implemented in `database/database_functions.py`. Add missing functions or disable chat routes before deploying.【F:api/chats_router.py†L14-L1204】【F:database/database_functions.py†L14-L196】
- **Search/upload errors referencing `embedding` column**: The SQLite schema lacks the `embedding` column used in search and upload endpoints; add migrations or adjust queries.【F:main.py†L875-L1855】【F:database/database_functions.py†L38-L103】
- **Missing dependencies**: No requirements file is included; ensure all imported packages are installed (FastAPI, Uvicorn, OpenAI SDK, pymilvus, passlib, docx2txt, PyPDF2, BeautifulSoup, structlog, google-auth, etc.).【F:main.py†L18-L118】【F:ai/openai_integration.py†L13-L120】【F:web/scraping_helpers.py†L8-L16】
- **Milvus unavailable**: `/health` reports `milvus: disconnected` and `/api/v1/stats` returns `limited` status when Milvus cannot be reached; confirm compose stack is running or provide fallback documents in memory.【F:main.py†L1898-L1949】【F:endpoints/api_endpoints.py†L118-L214】
- **OpenAI rate limits**: The OpenAI client logs warnings and retries on rate limits; ensure API key is valid and monitor usage.【F:ai/openai_integration.py†L13-L120】
