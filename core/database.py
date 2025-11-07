# core/database.py
"""Database connection managers for Milvus-only architecture."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import structlog
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

from config.settings import settings

logger = structlog.get_logger()


class DatabaseManager:
    """Manages Milvus database connection and collections."""

    def __init__(self):
        self.milvus_connected: bool = False
        self.cache: Dict[str, Dict[str, Any]] = {}  # simple in-memory cache

    async def initialize(self) -> None:
        """Initialize Milvus database connection and ensure base collections."""
        await self._init_milvus()
        logger.info("Milvus database connection initialized")

    async def _init_milvus(self) -> None:
        """Initialize Milvus connection and ensure baseline collections."""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self.milvus_connected = True
            await self._create_milvus_collections()
            logger.info("Milvus connection initialized")
        except Exception as e:
            logger.error("Failed to initialize Milvus", error=str(e))
            self.milvus_connected = False

    async def _create_milvus_collections(self) -> None:
        """Create baseline collections (Documents, Users, UserChats, SystemStats)."""
        try:
            # Documents (legacy)
            if not utility.has_collection("Documents"):
                doc_fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=10000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="file_size", dtype=DataType.INT64),
                    FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=100),
                ]
                doc_schema = CollectionSchema(doc_fields, "Document storage with embeddings and metadata")
                doc_collection = Collection("Documents", doc_schema)
                doc_collection.create_index(
                    "embedding",
                    {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
                )
                logger.info("Created Documents collection in Milvus")

            # Users (recreate if schema mismatch)
            if utility.has_collection("Users"):
                try:
                    existing = Collection("Users")
                    if len(existing.schema.fields) != 9:
                        utility.drop_collection("Users")
                        logger.info("Dropped Users due to schema mismatch; recreating")
                except Exception:
                    utility.drop_collection("Users")

            if not utility.has_collection("Users"):
                user_fields = [
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="password_hash", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="last_active", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="preferences", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1),
                ]
                user_schema = CollectionSchema(user_fields, "User information and preferences with authentication")
                user_collection = Collection("Users", user_schema)
                user_collection.create_index("dummy_vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
                logger.info("Created Users collection in Milvus")

            # UserChats (recreate if schema mismatch)
            if utility.has_collection("UserChats"):
                try:
                    existing = Collection("UserChats")
                    if len(existing.schema.fields) != 12:
                        utility.drop_collection("UserChats")
                        logger.info("Dropped UserChats due to schema mismatch; recreating")
                except Exception:
                    utility.drop_collection("UserChats")

            if not utility.has_collection("UserChats"):
                chat_fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="chat_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=5000),
                    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=10000),
                    FieldSchema(name="confidence", dtype=DataType.FLOAT),
                    FieldSchema(name="sources_used", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="reasoning", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="processing_time", dtype=DataType.FLOAT),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1),
                ]
                chat_schema = CollectionSchema(chat_fields, "Chat history and conversations")
                chat_collection = Collection("UserChats", chat_schema)
                chat_collection.create_index("dummy_vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
                logger.info("Created UserChats collection in Milvus")

            # SystemStats
            if not utility.has_collection("SystemStats"):
                stats_fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="stat_key", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="stat_value", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="stat_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1),
                ]
                stats_schema = CollectionSchema(stats_fields, "System statistics and cache")
                stats_collection = Collection("SystemStats", stats_schema)
                stats_collection.create_index("dummy_vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
                logger.info("Created SystemStats collection in Milvus")

        except Exception as e:
            logger.error("Failed to create Milvus collections", error=str(e))

    # inside class DatabaseManager:
    async def ensure_documents_v2(self, dim: int = 1536) -> None:
        """
        Ensure the new DocumentsV2 collection exists with:
          - doc_id (PK, VarChar <= 64)
          - chat_id (VarChar 64)
          - title (VarChar 512)
          - summary (VarChar 2048)
          - metadata_json (VarChar 8192)
          - created_at (VarChar 64)
          - embedding (FloatVector dim)
        """
        coll_name = "DocumentsV2"
        if not utility.has_collection(coll_name):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="doc_id",       dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                    FieldSchema(name="chat_id",      dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="title",        dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="summary",      dtype=DataType.VARCHAR, max_length=2048),
                    FieldSchema(name="metadata_json",dtype=DataType.VARCHAR, max_length=8192),
                    FieldSchema(name="created_at",   dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="embedding",    dtype=DataType.FLOAT_VECTOR, dim=dim),
                ],
                description="New document vectors (PK = doc_id) with lightweight metadata"
            )
            coll = Collection(name=coll_name, schema=schema)
            index_params = {
                "metric_type": "IP",          # or "L2"
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            coll.create_index(field_name="embedding", index_params=index_params)
            coll.load()
            logger.info("Created DocumentsV2 collection in Milvus")
        else:
            coll = Collection(name=coll_name)
            # optional: validate index; create if missing
            try:
                coll.indexes  # access to ensure created
            except Exception:
                index_params = {
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200}
                }
                coll.create_index(field_name="embedding", index_params=index_params)
            coll.load()


    async def set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        self.cache[key] = {"value": value, "expires_at": datetime.now(timezone.utc).timestamp() + ttl}

    async def clear_expired_cache(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        for k in list(self.cache.keys()):
            if self.cache[k].get("expires_at", 0) < now:
                self.cache.pop(k, None)

    async def close(self) -> None:
        if self.milvus_connected:
            connections.disconnect("default")
        logger.info("Milvus connection closed")


class MilvusStore:
    """Vector/document operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def upsert_document_v2(
        self,
        *,
        document_id: str,
        chat_id: str,
        title: str,
        summary: str,
        metadata: Dict[str, Any],
        embedding: List[float],
        created_at: str,
        collection_name: str = "DocumentsV2",
    ) -> None:
        """Upsert into DocumentsV2 with PK = document_id and attached chat_id."""
        try:
            if not self.db_manager.milvus_connected:
                raise Exception("Milvus not connected")

            dim = len(embedding)
            await self.db_manager.ensure_documents_v2(dim=dim)

            coll = Collection(collection_name)
            coll.load()

            row = [
                [document_id],
                [chat_id or ""],
                [title or ""],
                [summary or ""],
                [json.dumps(metadata or {}, ensure_ascii=False)],
                [created_at],
                [embedding],
            ]

            if hasattr(coll, "upsert"):
                coll.upsert(row)
            else:
                try:
                    coll.delete(expr=f"doc_id == '{document_id}'")
                except Exception:
                    pass
                coll.insert(row)
            coll.flush()                 # <--- add this line

            logger.info("Upserted document into DocumentsV2", doc_id=document_id, chat_id=chat_id)
        except Exception as e:
            logger.error("Failed to upsert document into DocumentsV2", error=str(e))
            raise

    async def insert_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        summary: str,
        metadata: Dict[str, Any],
        embedding: List[float],
        created_at: str,
    ) -> None:
        """(legacy) Insert full document into Documents collection (with content)."""
        try:
            if not self.db_manager.milvus_connected:
                raise Exception("Milvus not connected")

            collection = Collection("Documents")
            collection.load()

            file_size = len(content.encode("utf-8"))
            content_type = metadata.get("content_type", "text/plain")

            data = [
                [doc_id],
                [title],
                [content],
                [summary],
                [json.dumps(metadata)],
                [embedding],
                [created_at],
                [file_size],
                [content_type],
            ]
            collection.insert(data)
            collection.flush()
            logger.info("Inserted document into Documents", doc_id=doc_id)
        except Exception as e:
            logger.error("Failed to insert document into Documents", error=str(e))
            raise

    async def similarity_search(
        self, query_embedding: List[float], limit: int = 10, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Search in legacy Documents."""
        try:
            if not self.db_manager.milvus_connected:
                return []
            collection = Collection("Documents")
            collection.load()
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                output_fields=["doc_id", "title", "content", "summary", "metadata", "created_at"],
            )
            out: List[Dict[str, Any]] = []
            for hits in results:
                for hit in hits:
                    if hit.score >= threshold:
                        out.append(
                            {
                                "id": hit.entity.get("doc_id"),
                                "title": hit.entity.get("title"),
                                "content": hit.entity.get("content"),
                                "summary": hit.entity.get("summary"),
                                "metadata": hit.entity.get("metadata"),
                                "created_at": hit.entity.get("created_at"),
                                "similarity": hit.score,
                            }
                        )
            return out
        except Exception as e:
            logger.error("Milvus similarity search failed", error=str(e))
            return []

    async def similarity_search_v2(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.8,
        chat_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search in DocumentsV2 and optionally filter by chat_id."""
        try:
            if not self.db_manager.milvus_connected:
                return []
            await self.db_manager.ensure_documents_v2(dim=len(query_embedding))
            coll = Collection("DocumentsV2")
            coll.load()

            results = coll.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"ef": 80}},
                limit=limit,
                output_fields=["doc_id", "chat_id", "title", "summary", "metadata_json", "created_at"],
            )

            docs: List[Dict[str, Any]] = []
            for hits in results:
                for h in hits:
                    if h.score < threshold:
                        continue
                    row = {k: h.entity.get(k) for k in ["doc_id", "chat_id", "title", "summary", "metadata_json", "created_at"]}
                    if chat_id and row.get("chat_id") != chat_id:
                        continue
                    try:
                        meta = json.loads(row.get("metadata_json") or "{}")
                    except Exception:
                        meta = {}
                    docs.append(
                        {
                            "id": row.get("doc_id"),
                            "chat_id": row.get("chat_id"),
                            "title": row.get("title"),
                            "summary": row.get("summary"),
                            "metadata": meta,
                            "created_at": row.get("created_at"),
                            "similarity": h.score,
                        }
                    )
            return docs
        except Exception as e:
            logger.error("DocumentsV2 similarity search failed", error=str(e))
            return []

    async def list_all_documents(self) -> List[Dict[str, Any]]:
        """(legacy) List from Documents collection."""
        try:
            if not self.db_manager.milvus_connected:
                return []
            collection = Collection("Documents")
            collection.load()
            results = collection.query(
                expr="id >= 0",
                output_fields=["doc_id", "title", "content", "summary", "metadata", "created_at"],
                limit=1000,
            )
            docs: List[Dict[str, Any]] = []
            for r in results:
                metadata_str = r.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except Exception:
                    metadata = {}
                docs.append(
                    {
                        "id": r.get("doc_id"),
                        "title": r.get("title"),
                        "content": r.get("content"),
                        "summary": r.get("summary"),
                        "metadata": metadata,
                        "created_at": r.get("created_at"),
                        "file_size": len((r.get("content") or "").encode("utf-8")),
                        "content_type": metadata.get("content_type", "text/plain"),
                    }
                )
            return docs
        except Exception as e:
            logger.error("Failed to list documents from Milvus", error=str(e))
            return []

    async def delete_document(self, doc_id: str) -> bool:
        try:
            if not self.db_manager.milvus_connected:
                return False
            collection = Collection("Documents")
            collection.load()
            results = collection.query(expr=f'doc_id == "{doc_id}"', output_fields=["doc_id"], limit=1)
            if not results:
                return False
            collection.delete(f'doc_id == "{doc_id}"')
            collection.flush()
            return True
        except Exception as e:
            logger.error("Failed to delete document", error=str(e))
            return False

    async def clear_collection(self, collection_name: str) -> None:
        if not self.db_manager.milvus_connected:
            raise Exception("Milvus not connected")
        collection = Collection(collection_name)
        collection.load()
        collection.delete("id >= 0")
        collection.flush()
        logger.info("Cleared collection", collection=collection_name)


class MilvusUserManager:
    """User operations in Milvus."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def create_user(
        self, user_id: str, name: str, email: str, password_hash: str = "", description: str = "", preferences: Dict[str, Any] | None = None
    ) -> str:
        if not self.db_manager.milvus_connected:
            raise Exception("Milvus not connected")
        collection = Collection("Users")
        collection.load()
        created_at = datetime.now(timezone.utc).isoformat()
        last_active = created_at
        preferences_json = json.dumps(preferences or {})
        dummy_vector = [0.0]
        data = [
            [user_id],
            [name],
            [email],
            [password_hash],
            [description],
            [created_at],
            [last_active],
            [preferences_json],
            [dummy_vector],
        ]
        collection.insert(data)
        collection.flush()
        return user_id

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        if not self.db_manager.milvus_connected:
            return None
        collection = Collection("Users")
        collection.load()
        results = collection.query(
            expr=f'email == "{email}"',
            output_fields=["user_id", "name", "email", "password_hash", "description", "created_at", "last_active", "preferences"],
            limit=1,
        )
        if not results:
            return None
        user = results[0]
        try:
            preferences = json.loads(user.get("preferences", "{}"))
        except Exception:
            preferences = {}
        return {
            "userId": user.get("user_id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "password_hash": user.get("password_hash"),
            "description": user.get("description"),
            "createdAt": user.get("created_at"),
            "lastActive": user.get("last_active"),
            "preferences": preferences,
        }

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.db_manager.milvus_connected:
            return None
        collection = Collection("Users")
        collection.load()
        results = collection.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["user_id", "name", "email", "password_hash", "description", "created_at", "last_active", "preferences"],
            limit=1,
        )
        if not results:
            return None
        user = results[0]
        try:
            preferences = json.loads(user.get("preferences", "{}"))
        except Exception:
            preferences = {}
        return {
            "userId": user.get("user_id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "password_hash": user.get("password_hash"),
            "description": user.get("description"),
            "createdAt": user.get("created_at"),
            "lastActive": user.get("last_active"),
            "preferences": preferences,
        }

    async def get_all_users(self) -> List[Dict[str, Any]]:
        if not self.db_manager.milvus_connected:
            return []
        collection = Collection("Users")
        collection.load()
        results = collection.query(
            expr="user_id != ''",
            output_fields=["user_id", "name", "email", "description", "created_at", "last_active", "preferences"],
            limit=10000,
        )
        users: List[Dict[str, Any]] = []
        for u in results:
            try:
                preferences = json.loads(u.get("preferences", "{}"))
            except Exception:
                preferences = {}
            users.append(
                {
                    "userId": u.get("user_id"),
                    "name": u.get("name"),
                    "email": u.get("email"),
                    "description": u.get("description"),
                    "createdAt": u.get("created_at"),
                    "lastActive": u.get("last_active"),
                    "preferences": preferences,
                }
            )
        return users

    async def delete_user(self, user_id: str) -> bool:
        if not self.db_manager.milvus_connected:
            return False
        collection = Collection("Users")
        collection.load()
        collection.delete(f'user_id == "{user_id}"')
        collection.flush()
        return True

    async def create_chat_entry(
        self,
        user_id: str,
        question: str,
        answer: str,
        confidence: float,
        sources_used: List[str],
        reasoning: str,
        processing_time: float,
        session_id: str | None = None,
    ) -> str:
        if not self.db_manager.milvus_connected:
            raise Exception("Milvus not connected")
        import uuid as _uuid

        collection = Collection("UserChats")
        collection.load()
        chat_id = str(_uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        sources_str = ",".join(sources_used) if sources_used else ""
        session_id = session_id or str(_uuid.uuid4())
        dummy_vector = [0.0]
        data = [
            [chat_id],
            [user_id],
            [question],
            [answer],
            [confidence],
            [sources_str],
            [reasoning],
            [processing_time],
            [timestamp],
            [session_id],
            [dummy_vector],
        ]
        collection.insert(data)
        collection.flush() 
        return chat_id

    async def get_user_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.db_manager.milvus_connected:
            return []
        collection = Collection("UserChats")
        collection.load()
        results = collection.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["chat_id", "question", "answer", "confidence", "sources_used", "reasoning", "processing_time", "timestamp", "session_id"],
            limit=limit,
        )
        chats: List[Dict[str, Any]] = []
        for r in results:
            sources_used = r.get("sources_used", "").split(",") if r.get("sources_used") else []
            chats.append(
                {
                    "chatId": r.get("chat_id"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "confidence": r.get("confidence"),
                    "sourcesUsed": sources_used,
                    "reasoning": r.get("reasoning"),
                    "processingTime": r.get("processing_time"),
                    "timestamp": r.get("timestamp"),
                    "sessionId": r.get("session_id"),
                }
            )
        chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return chats


class MilvusSystemManager:
    """System stats and small cache helpers."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def store_system_stat(self, key: str, value: Any, stat_type: str = "general", metadata: Dict[str, Any] | None = None) -> None:
        if not self.db_manager.milvus_connected:
            return
        collection = Collection("SystemStats")
        collection.load()
        timestamp = datetime.now(timezone.utc).isoformat()
        value_str = json.dumps(value) if not isinstance(value, str) else value
        metadata_str = json.dumps(metadata or {})
        dummy_vector = [0.0]
        data = [[key], [value_str], [stat_type], [timestamp], [metadata_str], [dummy_vector]]
        collection.insert(data)
        collection.flush()

    async def get_system_stat(self, key: str) -> Optional[Any]:
        if not self.db_manager.milvus_connected:
            return None
        collection = Collection("SystemStats")
        collection.load()
        results = collection.query(
            expr=f'stat_key == "{key}"',
            output_fields=["stat_value", "stat_type", "timestamp", "metadata"],
            limit=1,
        )
        if not results:
            return None
        value_str = results[0].get("stat_value", "")
        try:
            return json.loads(value_str)
        except Exception:
            return value_str

    async def update_user_activity(self, user_id: str) -> None:
        if not self.db_manager.milvus_connected:
            return
        # store in memory cache for now
        ts = datetime.now(timezone.utc).isoformat()
        await self.db_manager.set_cache(f"user_activity_{user_id}", ts)


# Global singletons
db_manager = DatabaseManager()
milvus_store = MilvusStore(db_manager)
milvus_user_manager = MilvusUserManager(db_manager)
milvus_system_manager = MilvusSystemManager(db_manager)
