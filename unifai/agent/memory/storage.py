from typing import Dict, Any, Optional, List, Deque
import json
import os
from datetime import datetime
import numpy as np
import aiofiles
import asyncio
from collections import deque
from .base import Memory


class LocalStorage:
    def __init__(self, persist_directory: str, max_batch_size: int = 100, max_open_files: int = 256):
        self.persist_directory = persist_directory
        self.max_batch_size = max_batch_size
        self._write_queue: asyncio.Queue[Memory] = asyncio.Queue()
        self._write_lock : asyncio.Lock = asyncio.Lock()
        self._memory_cache : Dict[str, Memory] = {}
        self._write_batch : Deque[Memory] = deque()
        self._write_task: Optional[asyncio.Task] = None
        os.makedirs(persist_directory, exist_ok=True)
        self._file_semaphore = asyncio.Semaphore(max_open_files)

    async def initialize(self):
        if self._write_task is None:
            self._write_task = asyncio.create_task(self._start_write_worker())

    async def _start_write_worker(self):
        while True:
            try:
                memory = await self._write_queue.get()
                self._write_batch.append(memory)

                if (
                    len(self._write_batch) >= self.max_batch_size
                    or self._write_queue.empty()
                ):
                    async with self._write_lock:
                        batch = list(self._write_batch)
                        self._write_batch.clear()
                        await self._batch_write(batch)
            except Exception as e:
                print(f"Error in write worker: {e}")
                await asyncio.sleep(1)

    async def _batch_write(self, memories: List[Memory]):
        for memory in memories:
            memory_dict = await memory.to_dict()
            serialized_dict = self._serialize_memory(memory_dict)
            path = self._get_memory_path(memory.id)
            async with self._file_semaphore:
                async with aiofiles.open(path, "w") as f:
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
        deserialized["last_accessed"] = datetime.fromisoformat(
            memory_dict["last_accessed"]
        )
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

        async with self._file_semaphore:
            async with aiofiles.open(memory_path, "r") as f:
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
            if filename.endswith(".json"):
                memory_id = filename[:-5]
                tasks.append(self.load_memory(memory_id))

        memories = await asyncio.gather(*tasks)
        return [m for m in memories if m is not None]

    async def clear_all(self) -> None:
        for filename in os.listdir(self.persist_directory):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.persist_directory, filename))
        self._memory_cache.clear()

    async def save_memory_immediately(self, memory: Memory) -> None:
        memory_dict = await memory.to_dict()
        serialized_dict = self._serialize_memory(memory_dict)
        memory_path = self._get_memory_path(memory.id)
        
        async with self._file_semaphore:
            async with aiofiles.open(memory_path, "w") as f:
                await f.write(json.dumps(serialized_dict))
        
        self._memory_cache[memory.id] = memory
