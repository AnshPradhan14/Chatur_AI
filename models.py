from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# =============================================================
#                    USER MODELS
# =============================================================

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    id: str
    name: str
    email: EmailStr


# =============================================================
#                    CHAT MODELS
# =============================================================

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"


class Chat(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime


# =============================================================
#                    MESSAGE MODELS
# =============================================================

class Message(BaseModel):
    id: str
    chat_id: str
    role: str        # "user" or "assistant"
    content: str
    timestamp: datetime


# =============================================================
#                    DOCUMENT MODELS
# =============================================================

# ✔ Input model for upload (used in /documents/upload)
class DocumentUpload(BaseModel):
    title: str
    doc_type: str      # "pdf" | "url" | "text"


# ✔ Database model (Mongo stored)
class Document(BaseModel):
    id: str
    user_id: str
    title: str
    doc_type: str
    uploaded_at: datetime
    chunk_count: int


# =============================================================
#                    EMBEDDING MODELS
# =============================================================

class Embedding(BaseModel):
    id: str
    user_id: str
    doc_id: str
    chunk_text: str
    embedding: List[float]
