from typing import List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_manager):
        self.dimension = 1536  
        self.model_manager = model_manager

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        try:
            response = await self.model_manager.embedding(
                model=model,
                input=text,
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    