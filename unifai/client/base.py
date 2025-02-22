from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

@dataclass
class MessageContext:
    chat_id: str
    user_id: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)

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
    async def _connect(self):
        """Internal connection logic"""
        pass

    @abstractmethod
    async def _disconnect(self):
        """Internal disconnection logic"""
        pass

    @abstractmethod
    async def _send_message_impl(self, ctx: MessageContext, reply: str):
        """Implementation of sending message"""
        pass

    async def start(self):
        """Start the client"""
        if self._started:
            return
        await self._connect()
        self._started = True
        self._stop_event.clear()

    async def stop(self):
        """Stop the client"""
        if not self._started:
            return
        self._stop_event.set()
        await self._disconnect()
        self._started = False

    @ensure_started
    async def receive_message(self) -> Optional[MessageContext]:
        """Receive a message from the queue"""
        try:
            return await self._message_queue.get()
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    @ensure_started
    async def send_message(self, ctx: MessageContext, reply: str):
        """Send a message using the context"""
        await self._send_message_impl(ctx, reply) 