from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)

def ensure_started(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self._started:
            raise RuntimeError(f"Client {self.client_id} not started")
        return await func(self, *args, **kwargs)
    return wrapper

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
    def __init__(self, client_id: str):
        self._client_id = client_id
        self._started = False
        self._message_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

    @property
    def client_id(self) -> str:
        return self._client_id

    @abstractmethod
    async def start(self):
        """Start the client"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the client"""
        pass

    @abstractmethod
    @ensure_started
    async def receive_message(self) -> Optional[MessageContext]:
        """Receive a message from the queue"""
        pass

    @abstractmethod
    @ensure_started
    async def send_message(self, ctx: MessageContext, reply: str):
        """Send a message using the context"""
        pass
