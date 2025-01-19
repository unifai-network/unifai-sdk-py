from typing import List, Any, Dict, Optional
from datetime import datetime
from uuid import UUID
import json
from .base import Memory, MemoryRole, ToolInfo

def serialize_memory(memory: Memory) -> Dict[str, Any]:
    return {
        "content_text": memory.content.get("text", ""),
        "user_id": str(memory.user_id),
        "agent_id": str(memory.agent_id),
        "role": memory.role.value,
        "tools": json.dumps([
            {
                "name": tool.name,
                "description": tool.description
            } for tool in (memory.tools or [])
        ]),
        "created_at": memory.created_at.isoformat(),
        "unique": memory.unique
    }

def deserialize_memory(
    memory_id: str,
    metadata: Dict[str, Any],
    embedding: Optional[List[float]] = None,
    similarity: Optional[float] = None
) -> Memory:
    try:
        role = MemoryRole(metadata.get("role", MemoryRole.USER.value))
    except (ValueError, KeyError):
        role = MemoryRole.USER
        
    tools = None
    tools_str = metadata.get("tools", "[]")
    if tools_str:
        try:
            tools_data = json.loads(tools_str)
            tools = [
                ToolInfo(
                    name=tool["name"],
                    description=tool["description"]
                ) for tool in tools_data
            ]
        except (json.JSONDecodeError, KeyError, ValueError):
            tools = None
    
    return Memory(
        id=UUID(memory_id),
        content={"text": metadata["content_text"]},
        user_id=UUID(metadata["user_id"]),
        agent_id=UUID(metadata["agent_id"]),
        role=role,
        tools=tools,
        embedding=embedding,
        created_at=datetime.fromisoformat(metadata["created_at"]),
        unique=metadata["unique"],
        similarity=similarity
    )