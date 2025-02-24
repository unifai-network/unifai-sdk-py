from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class MessageContext(ABC):
    @property
    @abstractmethod
    def chat_id(self) -> str:
        """Get chat ID"""
        pass
        
    @property
    @abstractmethod
    def user_id(self) -> str:
        """Get user ID"""
        pass
        
    @property
    @abstractmethod
    def message(self) -> str:
        """Get message content"""
        pass
        
    @property
    @abstractmethod
    def extra(self) -> Dict[str, Any]:
        """Get extra data"""
        pass

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
    async def send_message(self, ctx: MessageContext, reply: str):
        """Send a message using the context"""
        raise NotImplementedError("send_message is not implemented")
