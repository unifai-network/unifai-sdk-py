from typing import List, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from .base import Memory
from .reflection import MemoryReflection
from .manager import MemoryManager
import json

if TYPE_CHECKING:
    from ..agent import Agent

@dataclass
class MemoryStep:
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    action_result: Optional[str] = None  
    observation: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_string(self) -> str:
        content_dict = {
            "thought": self.thought if self.thought else "",
            "action": self.action if self.action else "",
            "actionInput": self.action_input if self.action_input else "",
            "actionResult": self.action_result if self.action_result else "",
            "observation": self.observation if self.observation else ""
        }
        return json.dumps(content_dict)

    async def to_memory(self) -> Memory:
        metadata_copy = {}
        if self.metadata is not None:
            metadata_copy = self.metadata.copy()
            metadata_copy.pop('type', None)
            memory_type = self.metadata.get('type', '')
        else:
            memory_type = ''  # Default value when metadata is None
            
        return Memory(
            content=self.to_string(),
            type=memory_type,
            created_at=self.timestamp,
            metadata=metadata_copy,
        )


class WorkingMemory:
    def __init__(
        self,
        max_size: int = 10,
        agent: Optional["Agent"] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.max_size = max_size
        self.steps : List[MemoryStep] = []
        self.memory_manager = memory_manager
        self.agent = agent
        self.memory_reflection = MemoryReflection(agent=agent)

    def __len__(self) -> int:
        return len(self.steps)

    async def _step_to_memory(self, step: MemoryStep) -> Memory:
        metadata = {}
        if hasattr(step, 'metadata') and isinstance(step.metadata, dict):
            metadata = step.metadata
        
        return Memory(
            content=step.to_string(),
            type="working_memory",
            metadata=metadata,
            created_at=datetime.now()
        )

    async def add_step(
        self,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        action_input: Optional[str] = None,
        action_result: Optional[str] = None,  
        observation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryStep:
        step = MemoryStep(
            thought=thought,
            action=action,
            action_input=action_input,
            action_result=action_result, 
            observation=observation,
            metadata=metadata,
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
        if len(self.steps) <= self.max_size or not self.memory_manager:
            return

        memories_to_compress = []
        for step in self.steps[-self.max_size :]:
            memory = await self._step_to_memory(step)
            memories_to_compress.append(memory)

        for memory in memories_to_compress:
            compressed_content = await self.memory_reflection.compress_memory_content(
                memory
            )
            metadata = memory.metadata or {}
            metadata.update(
                {
                    "compressed": True,
                    "original_content": memory.content,
                }
            )
            await self.memory_manager.add_memory(
                content=compressed_content,
                memory_type=memory.type,
                associated_agents=memory.associated_agents or [],
                metadata=metadata,
            )
        self.steps = []

    def steps_to_string(self) -> str:
        if not self.steps:
            return "No steps recorded."

        formatted_steps = []
        for i, step in enumerate(self.steps, 1):
            step_str = f"Step {i}:\n{step.to_string()}"
            formatted_steps.append(step_str)

        return "\n\n".join(formatted_steps)
