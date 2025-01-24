from typing import List, Optional, Protocol, runtime_checkable
from uuid import UUID
from .base import Memory
from .plugin import MemoryRankPlugin

@runtime_checkable
class MemoryManager(Protocol):
    plugins: List[MemoryRankPlugin]
    
    def add_plugin(self, plugin: MemoryRankPlugin) -> None:
        """Add a ranking plugin"""
        raise NotImplementedError
        
    def remove_plugin(self, plugin_name: str) -> None:
        """Remove a plugin by name"""
        raise NotImplementedError
        
    def get_plugin(self, plugin_name: str) -> Optional[MemoryRankPlugin]:
        """Get a plugin by name"""
        raise NotImplementedError
        
    def list_plugins(self) -> List[str]:
        """List all registered plugin names"""
        raise NotImplementedError

    async def add_embedding_to_memory(self, memory: Memory) -> Memory:
        raise NotImplementedError

    async def create_memory(self, memory: Memory) -> None:
        raise NotImplementedError

    async def get_memories(
        self,
        content: str,
        count: int = 5,
        threshold: float = 0.0,
        **kwargs
    ) -> List[Memory]:
        """Get relevant memories based on content"""
        raise NotImplementedError

    async def remove_memory(self, memory_id: UUID) -> None:
        raise NotImplementedError

    async def remove_all_memories(self) -> None:
        raise NotImplementedError

    async def get_memory_by_id(self, memory_id: UUID) -> Optional[Memory]:
        raise NotImplementedError

    async def update_memory(self, memory: Memory) -> None:
        raise NotImplementedError