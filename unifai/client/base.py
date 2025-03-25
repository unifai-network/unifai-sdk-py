from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from litellm.types.utils import Message as LitellmMessage

class Message(LitellmMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@dataclass
class MessageContext(ABC):
    chat_id: str
    user_id: str
    message: str
    progress_report: bool
    cost: float

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
