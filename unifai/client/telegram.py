from telegram import Update, Bot, LinkPreviewOptions
from telegram.ext import ApplicationBuilder, ContextTypes, filters, MessageHandler
import logging
from .base import BaseClient, MessageContext, ensure_started

logger = logging.getLogger(__name__)

class TelegramClient(BaseClient):
    def __init__(self, bot_token: str, bot_name: str):
        super().__init__(client_id=f"telegram-{bot_name}")
        self.bot_token = bot_token
        self.bot_name = bot_name
        self._application = None

    async def _connect(self):
        """Connect to Telegram"""
        self._application = ApplicationBuilder().token(self.bot_token).build()
        
        if not self._application.updater:
            raise RuntimeError("Updater is not initialized")
        
        filter_conditions = (
            (filters.CaptionRegex(f"@{self.bot_name}") & (~filters.COMMAND)) |
            (filters.Mention(self.bot_name) & (~filters.COMMAND)) |
            (filters.ChatType.PRIVATE & (~filters.COMMAND))
        )
        
        self._application.add_handler(MessageHandler(
            filter_conditions,
            self._handle_telegram_update,
            block=False
        ))

        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling()

    async def _disconnect(self):
        """Disconnect from Telegram"""
        if self._application:
            if self._application.updater:
                await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

    async def _handle_telegram_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming Telegram update"""
        if not update.message or not update.message.from_user or not update.effective_chat:
            return

        message = update.message.text or update.message.caption
        if not message:
            return

        ctx = MessageContext(
            chat_id=str(update.effective_chat.id),
            user_id=str(update.message.from_user.id),
            message=message,
            extra={
                "is_private": update.effective_chat.type == "private",
                "has_media": bool(update.message.photo or update.message.video or update.message.document),
                "telegram_user": update.message.from_user,
                "telegram_chat": update.effective_chat
            }
        )
        await self._message_queue.put(ctx)

    async def _send_message_impl(self, ctx: MessageContext, reply: str):
        """Implementation of sending message for Telegram"""
        if not self._application:
            raise RuntimeError("Application not initialized")

        MAX_MESSAGE_LENGTH = 4000
        messages = [reply[i:i + MAX_MESSAGE_LENGTH] 
                   for i in range(0, len(reply), MAX_MESSAGE_LENGTH)]
        
        for msg in messages:
            await self._application.bot.send_message(
                chat_id=ctx.chat_id,
                text=msg,
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            ) 