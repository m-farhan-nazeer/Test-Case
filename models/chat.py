# ---------------------------pydantic model fo chat ----------------------
# app/schemas/chat.py
"""
Pydantic schemas for chat & message payloads.
Designed to work with sqlite3-backed repository functions.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, constr
from typing import Optional, Dict, Any  # add this


# --- Enums ---

class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


# --- Core Entities ---

class Message(BaseModel):
    id: int = Field(..., description="Autoincrement DB id")
    chat_id: str = Field(..., description="Conversation id; all messages in a chat share this")
    role: MessageRole
    content: str
    model_used: Optional[str] = None
    tokens_used: Optional[int] = Field(None, ge=0)
    metadata: Optional[str] = Field(
        None, description="Optional JSON string for tool calls, citations, etc."
    )
    created_at: str


class MessageCreate(BaseModel):
    # chat_id is optional: if absent -> server creates/ensures one
    chat_id: Optional[constr(min_length=1, max_length=64)] = Field(
        None,
        description="Provide to continue an existing chat; if omitted a new chat_id will be created."
    )
    role: MessageRole = MessageRole.user
    content: constr(min_length=1)  # non-empty user content
    # For assistant messages created server-side, these are ignored if provided:
    model_used: Optional[str] = None
    tokens_used: Optional[int] = Field(None, ge=0)
    metadata: Optional[str] = None


class Chat(BaseModel):
    id: str = Field(..., description="chat_id")
    title: Optional[str] = None
    user_id: Optional[str] = None
    created_at: str
    updated_at: str


class ChatCreate(BaseModel):
    # Optional initial title; many UIs set it after first message
    title: Optional[constr(strip_whitespace=True, max_length=200)] = None
    user_id: Optional[str] = None
    # Optional seed system prompt if you want to start a chat with system message
    system_prompt: Optional[constr(strip_whitespace=True, max_length=8000)] = None


# --- Response shapes ---

class ChatWithMessages(BaseModel):
    chat: Chat
    messages: List[Message]


class ChatListItem(BaseModel):
    id: str
    title: Optional[str] = None
    updated_at: str


class ChatListResponse(BaseModel):
    items: List[ChatListItem]
    total: int
    limit: int
    offset: int


# models/chat.py

from typing import Optional, List, Dict, Any  # add Dict, Any

class AskRequest(BaseModel):
    question: constr(min_length=1)
    chat_id: Optional[constr(min_length=1, max_length=64)] = None
    user_id: Optional[str] = None
    system_prompt: Optional[str] = None
    context: Optional[str] = None

    # NEW
    web_search_enabled: Optional[bool] = False
    web_search_settings: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    chat_id: str
    question_message: Message
    answer_message: Message
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
# -----------------------------------------------------document implementation---------------------------

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, constr, ConfigDict

class RequestDocument(BaseModel):
    # allow sending/receiving with the original keys (Document_Title, Content, etc.)
    model_config = ConfigDict(populate_by_name=True)

    document_title: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        alias="Document_Title",
        description="Required title (non-empty).",
    )
    content: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        alias="Content",
        description="Required content/body (non-empty).",
    )
    summary: Optional[str] = Field(
        None,
        alias="Summary",
        description="Optional short summary.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        alias="Meta data",
        description="Optional JSON object with arbitrary key/values.",
    )
    chat_id: Optional[constr(min_length=1, max_length=64, strip_whitespace=True)] = Field(
        None,
        alias="chat_id",
        description="Optional chat identifier (1–64 chars).",
    )

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, constr

class CreateDocumentResponse(BaseModel):
    document_id: UUID = Field(..., description="Unique identifier of the stored document")
    chat_id: Optional[constr(min_length=1, max_length=64, strip_whitespace=True)] = Field(
        None, description="Related chat id, if any"
    )
