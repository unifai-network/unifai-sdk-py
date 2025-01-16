from typing import List, Union
from datetime import datetime
import numpy as np
from .base import Memory
import logging

logger = logging.getLogger(__name__)

class ImportanceCalculator:
    def __init__(self, agent):
        self.time_weight = 0.3
        self.relevance_weight = 0.7
        self.agent = agent

    async def calculate_relevance(self, memory: Memory, current_time: datetime,
                                related_memories: List[Memory]) -> tuple[List[float], List[float]]:
        memory_times = [m.created_at for m in related_memories]
        time_factor = self._calculate_time_decay(memory.created_at, memory_times)
        relevance_factor = await self._calculate_relevance(memory, related_memories)
        return time_factor, relevance_factor

    async def calculate_memory_importance(self, memory: Memory) -> float:
        try:
            response = await self.agent.get_model_response(
                'agent.importance',
                memory_content=memory.content,
            )

            importance_score = response.get("importance_score")
            if importance_score is None:
                logger.error("Missing importance_score in response")
                return 0.05

            try:
                importance_score = float(importance_score)
            except (ValueError, TypeError):
                logger.error("Invalid importance_score format")
                return 0.05

            if "reasoning" in response:
                logger.debug(f"Memory importance calculation: {response['reasoning']}")
            return max(0.0, min(1.0, importance_score))
        
        except Exception as e:
            logger.error(f"Error calculating memory importance: {str(e)}")
            return 0.05

    def _calculate_time_decay(self, memory_created_at: datetime, 
                         reference_times: Union[datetime, List[datetime]]) -> List[float]:
        if isinstance(reference_times, datetime):
            time_diff = (reference_times - memory_created_at).total_seconds()
            return [abs(1.0 / (1.0 + time_diff / (24 * 3600)))]
        
        time_decays = []
        for ref_time in reference_times:
            time_diff = (ref_time - memory_created_at).total_seconds()
            decay = abs(1.0 / (1.0 + time_diff / (24 * 3600)))
            time_decays.append(decay)
            
        return time_decays

    async def _calculate_relevance(self, memory: Memory, related_memories: List[Memory]) -> List[float]:
        if not related_memories or memory.embedding is None:
            return [0.1] 
        similarities = []
        for related_memory in related_memories:
            if related_memory.embedding is not None:
                similarity = np.dot(memory.embedding, related_memory.embedding)
                similarities.append(similarity)
        
        return similarities if similarities else [0.05] * len(related_memories)
