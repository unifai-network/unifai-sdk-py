from typing import List, Optional, Dict
import numpy as np
from .base import Memory, MemoryStream
from .storage import LocalStorage

class MemoryManager:
    def __init__(self, importance_calculator, embedding_generator, agent):
        self.importance_calculator = importance_calculator
        self.embedding_generator = embedding_generator
        self.storage = LocalStorage(persist_directory=f"{agent.data_dir}/memories")
        self.memory_stream = MemoryStream(storage=self.storage)
        self.agent = agent

    async def add_memory(self, content: str, memory_type: str,
                        associated_agents: Optional[List[str]] = None,
                        metadata: Optional[Dict] = None) -> Memory:
        embedding = np.array(await self.embedding_generator.get_embedding(content, self.agent.get_model('embedding')))
        memory = Memory(content=content, type=memory_type,
                       associated_agents=associated_agents,
                       metadata=metadata, embedding=embedding)
        
        importance = await self.importance_calculator.calculate_memory_importance(
            memory)
        memory.importance_score = importance
        await self.memory_stream.add_memory(memory)
        return memory
    # To be used
    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        return await self.memory_stream.get_memory(memory_id)
    # To be used
    async def get_memories_by_type(self, memory_type: str) -> List[Memory]:
        return await self.memory_stream.get_memories_by_type(memory_type)

    async def get_memories_by_agent(self, agent_id: str) -> List[Memory]:
        return await self.memory_stream.get_memories_by_agent(agent_id)
    
    async def get_memories_by_associated_agent(self, agent_id: str) -> List[Memory]:
        return await self.memory_stream.get_memories_by_agent(agent_id)
        


    