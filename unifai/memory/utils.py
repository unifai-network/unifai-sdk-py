from typing import List, Any, Dict, Optional
from datetime import datetime
from uuid import UUID
import json
from enum import Enum
from .base import Memory, MemoryRole, ToolInfo, MemoryType
from pydantic import TypeAdapter
import uuid

def serialize_memory(memory: Memory) -> Dict[str, Any]:
    """Serialize memory to metadata format for Chroma storage"""
    metadata = {
        "user_id": str(memory.user_id),
        "agent_id": str(memory.agent_id),
        "memory_type": memory.memory_type.value,
        "role": memory.role.value,
        "content_text": memory.content["text"],
        "created_at": memory.created_at.isoformat(),
        "unique": memory.unique,
        **{k: str(v) for k, v in memory.metadata.items()},
        **({"tools": json.dumps([t.model_dump() for t in memory.tools])} if memory.tools else {})
    }

    metadata["content"] = json.dumps(memory.content)
    return metadata

def deserialize_memory(
    memory_id: str,
    metadata: Dict[str, Any],
    embedding: Optional[List[float]] = None,
    similarity: Optional[float] = None
) -> Memory:
    """Deserialize metadata from Chroma storage to Memory object"""
    def to_uuid(value: str) -> UUID:
        try:
            return UUID(value)
        except ValueError:
            return uuid.uuid5(uuid.NAMESPACE_DNS, str(value))

    tools = None
    if "tools" in metadata:
        try:
            tools = json.loads(metadata.pop("tools"))
        except json.JSONDecodeError:
            tools = None

    try:
        content = json.loads(metadata["content"])
    except (json.JSONDecodeError, KeyError):
        content = {
            "text": metadata["content_text"]
        }

    memory_data = {
        "id": to_uuid(memory_id),
        "user_id": to_uuid(metadata.get("user_id", metadata.get("chat_id"))),
        "agent_id": to_uuid(metadata["agent_id"]),
        "memory_type": metadata.get("memory_type", metadata.get("type", "interaction")),
        "role": metadata.get("role", "system"),
        "content": content,
        "created_at": datetime.fromisoformat(metadata.get("created_at", metadata.get("timestamp"))),
        "unique": metadata.get("unique", False),
        "tools": tools,
        "embedding": embedding,
        "similarity": similarity,
        "metadata": {
            k: v for k, v in metadata.items()
            if k not in {
                "user_id", "agent_id", "memory_type", "role", 
                "content_text", "created_at", "unique", "tools",
                "chat_id", "type", "content"
            }
        }
    }

    return TypeAdapter(Memory).validate_python(memory_data)