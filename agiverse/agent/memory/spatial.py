from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from .base import Memory

class SpatialMemory(Memory):
    def __init__(self, content: str, coordinates: Tuple[float, float, float],
                 location_type: str = None, address: str = None,
                 type: str = "spatial", associated_agents: List[str] = None,
                 metadata: Dict = None, embedding: Optional[np.ndarray] = None,
                 importance_score: float = 0.0, id: str = None,
                 created_at: datetime = None, last_accessed: datetime = None):
        super().__init__(content, type, associated_agents, metadata,
                        embedding, importance_score, id, created_at, last_accessed)
        self.coordinates = coordinates
        self.location_type = location_type
        self.address = address

    async def to_dict(self) -> Dict:
        memory_dict = await super().to_dict()
        memory_dict.update({
            "coordinates": self.coordinates,
            "location_type": self.location_type,
            "address": self.address
        })
        return memory_dict

    @classmethod
    async def from_dict(cls, data: Dict):
        return cls(**data)

    def get_distance_to(self, other_memory: 'SpatialMemory') -> float:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(self.coordinates, other_memory.coordinates)))

    def is_nearby(self, coordinates: Tuple[float, float, float], threshold: float = 100.0) -> bool:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(self.coordinates, coordinates))) <= threshold 