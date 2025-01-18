from typing import List, Optional, Protocol, runtime_checkable
from uuid import UUID
from .base import Memory

@runtime_checkable
class MemoryManager(Protocol):
    async def add_embedding_to_memory(self, memory: Memory) -> Memory:
        raise NotImplementedError

    async def create_memory(self, memory: Memory) -> None:
        raise NotImplementedError

    async def search_memories_by_embedding(
        self,
        embedding: List[float],
        match_threshold: float = 0.8,
        count: int = 10,
        unique: bool = False
    ) -> List[Memory]:
        raise NotImplementedError

    async def get_memories(
        self,
        count: Optional[int] = None,
        unique: bool = False,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Memory]:
        raise NotImplementedError

    async def remove_memory(self, memory_id: UUID) -> None:
        raise NotImplementedError

    async def remove_all_memories(self) -> None:
        raise NotImplementedError

    async def get_memory_by_id(self, memory_id: UUID) -> Optional[Memory]:
        raise NotImplementedError

    async def update_memory(self, memory: Memory) -> None:
        raise NotImplementedError