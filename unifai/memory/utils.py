from typing import List, Any, Dict, Optional
from datetime import datetime
from uuid import UUID
import json
from enum import Enum
from .base import Memory, MemoryRole, ToolInfo, MemoryType
from pydantic import TypeAdapter

def serialize_memory(memory: Memory) -> Dict[str, Any]:
    memory_dict = memory.model_dump(
        mode="json",
        exclude={"embedding", "similarity"}
    )
    
    memory_dict["content_text"] = memory_dict["content"]["text"]
    del memory_dict["content"]
    
    for key, value in memory_dict.items():
        if isinstance(value, (dict, list, tuple)):
            memory_dict[key] = json.dumps(value)
        elif isinstance(value, Enum):
            memory_dict[key] = value.value
    
    return memory_dict

def deserialize_memory(
    memory_id: str,
    metadata: Dict[str, Any],
    embedding: Optional[List[float]] = None,
    similarity: Optional[float] = None
) -> Memory:

    for key in ["metadata", "tools"]:
        if isinstance(metadata.get(key), str):
            try:
                metadata[key] = json.loads(metadata[key])
            except json.JSONDecodeError:
                metadata[key] = {} if key == "metadata" else None

    memory_data = {
        **metadata,
        "id": memory_id,
        "content": {"text": metadata.pop("content_text")},
        **({"embedding": embedding} if embedding is not None else {}),
        **({"similarity": similarity} if similarity is not None else {})
    }
    
    return TypeAdapter(Memory).validate_python(memory_data)