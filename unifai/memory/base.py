from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class MemoryRole(Enum):
    USER = "user"
    SYSTEM = "system"

class ToolInfo(BaseModel):
    name: str
    description: str

class MemoryType(Enum):
    INTERACTION = "interaction"
    FACT = "fact"
    GOAL = "goal"
    EVALUATION = "evaluation"
    CUSTOM = "custom"

class Memory(BaseModel):
    id: UUID
    user_id: UUID
    agent_id: UUID
    content: Dict[str, Any]
    memory_type: MemoryType = MemoryType.INTERACTION
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role: MemoryRole = MemoryRole.USER
    tools: Optional[List[ToolInfo]] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    unique: bool = False
    similarity: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class StorageType(Enum):
    HTTP = "http"
    PERSISTENT = "persistent"

@dataclass
class ChromaConfig:
    storage_type: StorageType = StorageType.HTTP
    host: str = "localhost"
    port: int = 8000
    dimensions: int = 1536
    collection_name: str = "memories"
    distance_metric: str = "cosine"
    embedding_function: Optional[str] = None
    persist_directory: Optional[str] = "./chroma_db"