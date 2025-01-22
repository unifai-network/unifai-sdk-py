from typing import List, Any, Dict, Optional
from datetime import datetime
from uuid import UUID
import json
from .base import Memory, MemoryRole, ToolInfo
from pydantic import TypeAdapter

def serialize_memory(memory: Memory) -> Dict[str, Any]:
    if memory.embedding is not None:
        if hasattr(memory.embedding, 'tolist'):
            memory.embedding = memory.embedding.tolist()
        elif not isinstance(memory.embedding, list):
            memory.embedding = list(memory.embedding)
    
    memory_dict = memory.model_dump(
        mode="json",
        exclude={"embedding", "similarity"}
    )
    
    memory_dict["content_text"] = memory_dict["content"]["text"]
    del memory_dict["content"]
    
    if memory_dict.get("tools"):
        memory_dict["tools"] = json.dumps(memory_dict["tools"])
    
    return memory_dict

def deserialize_memory(
    memory_id: str,
    metadata: Dict[str, Any],
    embedding: Optional[List[float]] = None,
    similarity: Optional[float] = None
) -> Memory:
    metadata = {
        **metadata,
        "id": memory_id,
        "content": {"text": metadata.pop("content_text")},
        **({"embedding": embedding} if embedding is not None else {}),
        **({"similarity": similarity} if similarity is not None else {})
    }
    
    if isinstance(metadata.get("tools"), str):
        metadata["tools"] = json.loads(metadata["tools"])
    
    return TypeAdapter(Memory).validate_json(json.dumps(metadata))