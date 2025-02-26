from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
import litellm

Message = litellm.Message

@dataclass
class MessageContext(ABC):
    chat_id: str
    user_id: str
    message: str
    extra: Dict[str, Any]

class BaseClient(ABC):
    @property
    @abstractmethod
    def client_id(self) -> str:
        """Get client ID"""
        raise NotImplementedError("client_id is not implemented")

    @abstractmethod
    async def start(self):
        """Start the client"""
        raise NotImplementedError("start is not implemented")

    @abstractmethod
    async def stop(self):
        """Stop the client"""
        raise NotImplementedError("stop is not implemented")

    @abstractmethod
    async def receive_message(self) -> Optional[MessageContext]:
        """Receive a message from the queue"""
        raise NotImplementedError("receive_message is not implemented")

    @abstractmethod
    async def send_message(self, ctx: MessageContext, reply_messages: List[Message]):
        """Send a message using the context"""
        raise NotImplementedError("send_message is not implemented")
