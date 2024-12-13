from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
import numpy as np
import aiofiles
import asyncio
from collections import deque
import logging
from .base import Memory

logger = logging.getLogger(__name__)

class LocalStorage:
    def __init__(self, persist_directory: str, max_batch_size: int = 100):
        self.persist_directory = persist_directory
        self.max_batch_size = max_batch_size
        self._write_queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._memory_cache = {}
        self._write_batch = deque()
        os.makedirs(persist_directory, exist_ok=True)
        
    async def _start_write_worker(self):
        while True:
            try:
                memory = await self._write_queue.get()
                self._write_batch.append(memory)
                
                if len(self._write_batch) >= self.max_batch_size or self._write_queue.empty():
                    async with self._write_lock:
                        batch = list(self._write_batch)
                        self._write_batch.clear()
                        await self._batch_write(batch)
            except Exception as e:
                logger.error(f"Error in write worker: {e}")
                await asyncio.sleep(1)
                
    async def _batch_write(self, memories: List[Memory]):
        for memory in memories:
            memory_dict = await memory.to_dict()
            serialized_dict = self._serialize_memory(memory_dict)
            path = self._get_memory_path(memory.id)
            async with aiofiles.open(path, 'w') as f:
                await f.write(json.dumps(serialized_dict))
            self._memory_cache[memory.id] = memory

    def _get_memory_path(self, memory_id: str) -> str:
        return os.path.join(self.persist_directory, f"{memory_id}.json")

    def _serialize_memory(self, memory_dict: Dict[str, Any]) -> Dict[str, Any]:
        serialized = memory_dict.copy()
        serialized["created_at"] = memory_dict["created_at"].isoformat()
        serialized["last_accessed"] = memory_dict["last_accessed"].isoformat()
        if memory_dict["embedding"] is not None:
            if isinstance(memory_dict["embedding"], np.ndarray):
                serialized["embedding"] = memory_dict["embedding"].tolist()
            else:
                serialized["embedding"] = memory_dict["embedding"]
        return serialized

    def _deserialize_memory(self, memory_dict: Dict[str, Any]) -> Dict[str, Any]:
        deserialized = memory_dict.copy()
        deserialized["created_at"] = datetime.fromisoformat(memory_dict["created_at"])
        deserialized["last_accessed"] = datetime.fromisoformat(memory_dict["last_accessed"])
        if memory_dict["embedding"] is not None:
            deserialized["embedding"] = np.array(memory_dict["embedding"])
        return deserialized

    async def save_memory(self, memory: Memory) -> None:
        await self._write_queue.put(memory)

    async def load_memory(self, memory_id: str) -> Optional[Memory]:
        if memory_id in self._memory_cache:
            return self._memory_cache[memory_id]
            
        memory_path = self._get_memory_path(memory_id)
        if not os.path.exists(memory_path):
            return None
            
        async with aiofiles.open(memory_path, 'r') as f:
            content = await f.read()
            memory_dict = json.loads(content)
            
        deserialized_dict = self._deserialize_memory(memory_dict)
        memory = await Memory.from_dict(deserialized_dict)
        self._memory_cache[memory_id] = memory
        return memory

    async def delete_memory(self, memory_id: str) -> bool:
        memory_path = self._get_memory_path(memory_id)
        if os.path.exists(memory_path):
            os.remove(memory_path)
            self._memory_cache.pop(memory_id, None)
            return True
        return False

    async def list_all_memories(self) -> List[Memory]:
        tasks = []
        for filename in os.listdir(self.persist_directory):
            if filename.endswith('.json'):
                memory_id = filename[:-5]
                tasks.append(self.load_memory(memory_id))
                
        memories = await asyncio.gather(*tasks)
        return [m for m in memories if m is not None]

    async def clear_all(self) -> None:
        for filename in os.listdir(self.persist_directory):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.persist_directory, filename))
        self._memory_cache.clear()