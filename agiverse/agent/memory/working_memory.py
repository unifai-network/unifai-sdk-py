from typing import List, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from .base import Memory
from .reflection import MemoryReflection
from .manager import MemoryManager

if TYPE_CHECKING:
    from ..agent import Agent

@dataclass
class MemoryStep:
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_string(self) -> str:
        parts = []
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.action_input:
            parts.append(f"Action Input: {self.action_input}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)

    async def to_memory(self) -> Memory:
        return Memory(
            content=self.to_string(),
            type="working_memory",
            created_at=self.timestamp,
            metadata=self.metadata
        )

class WorkingMemory:
    def __init__(self, max_size: int = 10, agent: Optional["Agent"] = None, memory_manager: MemoryManager = None):
        self.max_size = max_size
        self.steps = []
        self.memory_manager = memory_manager
        self.agent = agent
        self.memory_reflection = MemoryReflection(agent=agent)

    def __len__(self) -> int:
        return len(self.steps)

    async def _step_to_memory(self, step: MemoryStep) -> Memory:
        memory = await step.to_memory() 
        if self.memory_reflection:
            memory.content = self.memory_reflection._format_memory_content(memory.content)
        return memory

    async def add_step(self, thought: Optional[str] = None, 
                      action: Optional[str] = None,
                      action_input: Optional[str] = None,
                      observation: Optional[str] = None,
                      metadata: Dict[str, Any] = None) -> MemoryStep:
        step = MemoryStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            metadata=metadata
        )
        
        self.steps.append(step)
        
        return step
    # To be used
    def get_recent_steps(self, n: Optional[int] = None) -> List[MemoryStep]:
        if n is None:
            return self.steps.copy()
        return self.steps[-n:]
    # To be used
    def get_last_step(self) -> Optional[MemoryStep]:
        return self.steps[-1] if self.steps else None
  
    def clear(self) -> None:
        self.steps.clear()

    async def _compress_steps(self) -> None:
        if len(self.steps) <= self.max_size:
            return

        # will enable long term memory in the future
        self.steps.pop(0)
        return

        memories_to_compress = []
        for step in self.steps[-self.max_size:]:
            memory = await self._step_to_memory(step)
            memories_to_compress.append(memory)
        
        for memory in memories_to_compress:
            compressed_content = await self.memory_reflection.compress_memory_content(memory)
            metadata = memory.metadata or {}
            metadata.update({
                "compressed": True,
                "original_content": memory.content,
                "original_type": memory.type
            })
            await self.memory_manager.add_memory(
                content=compressed_content,
                memory_type="compressed_memory",
                associated_agents=memory.associated_agents or [],
                metadata=metadata
            )
        self.steps = []
        self.clear()

    def steps_to_string(self) -> str:
        if not self.steps:
            return "No steps recorded."
        
        formatted_steps = []
        for i, step in enumerate(self.steps, 1):
            step_str = f"Step {i}:\n{step.to_string()}"
            formatted_steps.append(step_str)
            
        return "\n\n".join(formatted_steps)

    