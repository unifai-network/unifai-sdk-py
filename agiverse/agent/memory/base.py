from typing import List, Dict, Optional, Set
from datetime import datetime
import numpy as np
import uuid
import asyncio
from functools import lru_cache
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Memory:
    __slots__ = (
        "id",
        "content",
        "type",
        "associated_agents",
        "metadata",
        "embedding",
        "importance_score",
        "created_at",
        "last_accessed",
    )

    def __init__(
    self,
    content: str,
    type: str = "observation",
    associated_agents: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    embedding: Optional[np.ndarray] = None,
    importance_score: float = 0.0,
    id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    last_accessed: Optional[datetime] = None,
):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.type = type
        self.associated_agents = associated_agents or []
        self.metadata = metadata or {}
        self.embedding = embedding
        self.importance_score = importance_score
        self.created_at = created_at or datetime.now()
        self.last_accessed = last_accessed or datetime.now()

    @lru_cache(maxsize=1000)
    async def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "associated_agents": self.associated_agents,
            "metadata": self.metadata,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "importance_score": self.importance_score,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    async def from_dict(cls, data: Dict):
        if data.get("embedding") is not None:
            data["embedding"] = np.array(data["embedding"])
        return cls(**data)


class MemoryStream:
    def __init__(self, storage=None, cache_size: int = 1000):
        self.memories: OrderedDict[str, Memory] = OrderedDict()
        self._cache: OrderedDict[str, Memory] = OrderedDict()
        self._save_queue: asyncio.Queue[Memory] = asyncio.Queue()
        self.storage = storage
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._cache_size = cache_size
        self._batch_size = 100
        self._save_task = None

    async def _start_save_worker(self):
        while True:
            batch = []
            try:
                while len(batch) < self._batch_size:
                    memory = await self._save_queue.get()
                    batch.append(memory)
                    if self._save_queue.empty():
                        break
                if batch:
                    await self._batch_save(batch)
            except Exception as e:
                logger.error(f"Error in save worker: {e}")
                await asyncio.sleep(1)

    async def _batch_save(self, memories: List[Memory]):
        if self.storage:
            tasks = [self.storage.save_memory(memory) for memory in memories]
            await asyncio.gather(*tasks)

    def _update_cache(self, memory_id: str, memory: Memory):
        self._cache[memory_id] = memory
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def ensure_initialized(self):
        async with self._initialization_lock:
            if not self._initialized:
                if self.storage:
                    await self.storage.initialize()
                await self.initialize()
                self._initialized = True
                if self.storage:
                    self._save_task = asyncio.create_task(self._start_save_worker())

    async def initialize(self):
        if self.storage:
            try:
                memories = await self.storage.list_all_memories()

                tasks = []
                for memory in memories:
                    tasks.append(self._process_memory(memory))
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.error(f"Failed to initialize memory stream: {e}")
                self.memories = OrderedDict()

    async def _process_memory(self, memory: Memory):
        try:
            memory_dict = await memory.to_dict()
            if isinstance(memory_dict, dict):
                self.memories[memory.id] = memory
                self._update_cache(memory.id, memory)
        except Exception as e:
            logger.error(f"Failed to process memory: {e}")

    async def add_memory(self, memory: Memory) -> str:
        await self.ensure_initialized()
        self.memories[memory.id] = memory
        self._update_cache(memory.id, memory)
        if self.storage:
            await self._save_queue.put(memory)
        return memory.id

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        if memory_id in self._cache:
            memory: Optional[Memory] = self._cache[memory_id]
        else:
            memory = self.memories.get(memory_id)
            if memory is None and self.storage:
                memory = await self.storage.load_memory(memory_id)
                if memory:
                    self.memories[memory_id] = memory
                    self._update_cache(memory_id, memory)
        if memory:
            memory.last_accessed = datetime.now()
            if self.storage:
                await self._save_queue.put(memory)
        return memory

    async def update_memory(self, memory: Memory) -> None:
        if memory.id in self.memories:
            self.memories[memory.id] = memory
            self._update_cache(memory.id, memory)
            if self.storage:
                await self.storage.save_memory(memory)

    async def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._cache.pop(memory_id, None)
            if self.storage:
                await self.storage.delete_memory(memory_id)
            return True
        return False

    async def clear(self) -> None:
        self.memories.clear()
        self._cache.clear()
        if self.storage:
            await self.storage.clear_all()

    async def get_all_memories(self):
        await self.ensure_initialized()
        return list(self.memories.values())

    async def get_memories_by_type(self, memory_type: str) -> List[Memory]:
        await self.ensure_initialized()
        return [
            memory for memory in self.memories.values()
            if memory.type == memory_type
        ]
    
    async def get_memories_by_associated_agent(self, agent_id: str) -> List[Memory]:
        await self.ensure_initialized()
        return [
            memory for memory in self.memories.values()
            if agent_id in (memory.associated_agents or [])
        ]

    async def update_memory_immediately(self, memory: Memory) -> None:
        if memory.id in self.memories:
            self.memories[memory.id] = memory
            self._update_cache(memory.id, memory)
            if self.storage:
                await self.storage.save_memory_immediately(memory)
