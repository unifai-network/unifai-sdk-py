import asyncio
import logging
from functools import wraps
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from telegram import Update, LinkPreviewOptions
from telegram.ext import ApplicationBuilder, ContextTypes, filters, MessageHandler
from .base import BaseClient, MessageContext, Message

logger = logging.getLogger(__name__)

def ensure_started(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self._started:
            raise RuntimeError(f"Client {self.client_id} not started")
        return await func(self, *args, **kwargs)
    return wrapper

@dataclass
class TelegramMessageContext(MessageContext):
    chat_id: str
    user_id: str
    message_id: int
    message: str
    extra: Dict[str, Any]

class TelegramClient(BaseClient):
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.bot_name = ""
        self._application = None
        self._started = False
        self._message_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

    @property
    def client_id(self) -> str:
        return f"telegram-{self.bot_name}"

    async def start(self):
        """Start the client"""
        if self._started:
            return
            
        self._application = ApplicationBuilder().token(self.bot_token).build()

        await self._application.initialize()

        self.bot_name = self._application.bot.username

        logger.info(f"Bot name: {self.bot_name}")

        filter_conditions = (
            (filters.CaptionRegex(f"@{self.bot_name}") & (~filters.COMMAND)) |
            (filters.Mention(self.bot_name) & (~filters.COMMAND)) |
            (filters.ChatType.PRIVATE & (~filters.COMMAND))
        )
        
        self._application.add_handler(MessageHandler(
            filter_conditions,
            self._handle_telegram_update,
        ))

        if not self._application.updater:
            raise RuntimeError("Updater is not initialized")

        await self._application.start()
        await self._application.updater.start_polling()
        self._started = True
        self._stop_event.clear()

    async def stop(self):
        """Stop the client"""
        if not self._started:
            return
            
        if self._application:
            if self._application.updater:
                await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
            
        self._stop_event.set()
        self._started = False

    @ensure_started
    async def receive_message(self) -> Optional[TelegramMessageContext]:
        """Receive a message from the queue"""
        try:
            return await self._message_queue.get()
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    @ensure_started
    async def send_message(self, ctx: TelegramMessageContext, reply_messages: List[Message]):
        """Send a message using the context"""
        if not self._application:
            raise RuntimeError("Application not initialized")

        reply_text = "Failed to generate response."
        if reply_messages and reply_messages[-1].content:
            reply_text = reply_messages[-1].content 

        MAX_MESSAGE_LENGTH = 4000
        messages = [reply_text[i:i + MAX_MESSAGE_LENGTH] 
                   for i in range(0, len(reply_text), MAX_MESSAGE_LENGTH)]
        
        for msg in messages:
            await self._application.bot.send_message(
                chat_id=ctx.chat_id,
                text=msg,
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )

    async def _handle_telegram_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming Telegram update"""
        if not update.message or not update.message.from_user or not update.effective_chat:
            return

        message = update.message.text or update.message.caption
        if not message:
            return

        ctx = TelegramMessageContext(
            chat_id=str(update.effective_chat.id),
            user_id=str(update.message.from_user.id),
            message_id=update.message.message_id,
            message=message,
            extra={
                "is_private": update.effective_chat.type == "private",
                "has_media": bool(update.message.photo or update.message.video or update.message.document),
                "telegram_user": update.message.from_user,
                "telegram_chat": update.effective_chat
            }
        )
        await self._message_queue.put(ctx) 