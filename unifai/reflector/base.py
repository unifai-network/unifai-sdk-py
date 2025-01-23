from abc import ABC
from typing import List, Any
from .types import ReflectionExample, ReflectionResult

class BaseReflector(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        similes: List[str],
        prompt_template: str
    ):
        self.name = name
        self.description = description
        self.similes = similes
        self.examples: List[ReflectionExample] = []
        self.always_run: bool = False
        self.prompt_template = prompt_template

    async def reflect(self, content: Any) -> ReflectionResult:
        try:
            return await self.process_reflection(content)
        except Exception as e:
            return ReflectionResult(success=False, data=None, reason=str(e))

    async def process_reflection(self, content: str) -> ReflectionResult:
        raise NotImplementedError