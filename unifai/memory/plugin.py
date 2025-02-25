from abc import ABC, abstractmethod
from typing import (
    List, 
    Optional, 
    Protocol, 
    TypeVar, 
    Generic, 
    Dict, 
    Any,
    runtime_checkable
)
from dataclasses import dataclass, field
from unifai.memory.base import Memory

T = TypeVar('T')
PluginConfig = TypeVar('PluginConfig')

@runtime_checkable
class PluginContext(Protocol):
    """Protocol for plugin context data"""
    content: str
    count: int
    threshold: float = 0.0
    
    @property
    def extra_args(self) -> Dict[str, Any]:
        """Additional arguments passed to the plugin"""
        return {}

@dataclass
class RankingResult:
    """Result of a ranking operation"""
    memories: List[Memory]
    scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryContext:
    """Concrete implementation of PluginContext"""
    content: str
    count: int
    threshold: float = 0.0
    _extra_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def extra_args(self) -> Dict[str, Any]:
        return self._extra_args or {}

class MemoryRankPlugin(Generic[PluginConfig], ABC):
    """Base class for memory ranking plugins with type-safe configuration"""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config
        self._weight: float = 1.0
        self._enabled: bool = True
        
    @property
    def name(self) -> str:
        """Plugin name, defaults to class name"""
        return self.__class__.__name__
        
    @property
    def weight(self) -> float:
        """Plugin weight in ranking calculations"""
        return self._weight
    
    @weight.setter
    def weight(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("Weight must be between 0 and 1")
        self._weight = value
        
    @property
    def enabled(self) -> bool:
        """Whether the plugin is enabled"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    @abstractmethod
    async def calculate_scores(
        self, 
        memories: List[Memory],
        context: PluginContext
    ) -> Dict[str, float]:
        """Calculate scores for memories"""
        pass
        
    async def rerank(
        self,
        memories: List[Memory],
        context: PluginContext
    ) -> RankingResult:
        """Rerank memories using plugin logic"""
        if not self.enabled or not memories:
            return RankingResult(
                memories=memories[:context.count],
                scores={str(m.id): 0.0 for m in memories[:context.count]}
            )
            
        scores = await self.calculate_scores(memories, context)
        # Update memory similarities
        for memory in memories:
            memory_id = str(memory.id)
            if memory_id in scores:
                current_score = memory.similarity or 0.0
                memory.similarity = current_score * (1 - self.weight) + scores[memory_id] * self.weight
                
        # Sort by updated similarity
        memories.sort(key=lambda x: x.similarity or 0.0, reverse=True)
        
        return RankingResult(
            memories=memories[:context.count],
            scores=scores,
            metadata={"plugin_name": self.name, "weight": self.weight}
        )

    def __repr__(self) -> str:
        return f"{self.name}(weight={self.weight}, enabled={self.enabled})"