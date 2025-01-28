from dataclasses import dataclass
from typing import List, Optional, Any, Dict, TypeVar, Generic
from enum import Enum

T = TypeVar('T')

class ReflectionType(Enum):
    FACT = "fact"
    GOAL = "goal"
    TRUST = "trust"
    EMOTION = "emotion"
    INTENT = "intent"

@dataclass
class ReflectionExample:
    context: str
    input: str
    output: str

@dataclass
class ReflectionResult(Generic[T]):
    success: bool
    data: Optional[T]
    reason: Optional[str] = None