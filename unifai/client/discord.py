import asyncio
import logging
from functools import wraps
from dataclasses import dataclass
from typing import List, Optional, Union
import discord
from discord import Message as DiscordMessage, User, Guild, abc
from discord.ext import commands
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
class DiscordMessageContext(MessageContext):
    channel: abc.Messageable
    chat_id: str
    user: Union[User, discord.Member]
    user_id: str
    message: str
    message_id: int
    original_message: DiscordMessage
    guild: Optional[Guild] = None

class DiscordClient(BaseClient):
    def __init__(self, bot_token: str, command_prefix: str = "!"):
        intents = discord.Intents.default()
        intents.message_content = True

        self.bot_token = bot_token
        self.bot_name = ""
        self.command_prefix = command_prefix
        self._bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        self._started = False
        self._message_queue: asyncio.Queue[DiscordMessageContext] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        
        @self._bot.event
        async def on_ready():
            self.bot_name = self._bot.user.name if self._bot.user else "Unknown"
            logger.info(f"Logged in as {self.bot_name}")
            
        @self._bot.event
        async def on_message(message):
            if message.author == self._bot.user:
                return
                
            await self._bot.process_commands(message)
            
            should_respond = (
                self._bot.user in message.mentions or 
                isinstance(message.channel, discord.DMChannel)
            )
            
            if should_respond:
                await self._handle_discord_message(message)

    @property
    def client_id(self) -> str:
        return f"discord-{self.bot_name}"

    async def start(self):
        """Start the client"""
        if self._started:
            return
            
        self._stop_event.clear()
        self._started = True
    
        asyncio.create_task(self._bot.start(self.bot_token))

        while not self._bot.is_ready():
            await asyncio.sleep(0.1)

    async def stop(self):
        """Stop the client"""
        if not self._started:
            return
            
        await self._bot.close()
        self._stop_event.set()
        self._started = False

    @ensure_started
    async def receive_message(self) -> Optional[DiscordMessageContext]:
        """Receive a message from the queue"""
        try:
            return await self._message_queue.get()
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    @ensure_started
    async def send_message(self, ctx: DiscordMessageContext, reply_messages: List[Message]):
        """Send a message using the context"""
        reply_text = "Failed to generate response."
        if reply_messages and reply_messages[-1].content:
            reply_text = reply_messages[-1].content 

        MAX_MESSAGE_LENGTH = 2000  # Discord's max message length
        messages = [reply_text[i:i + MAX_MESSAGE_LENGTH] 
                   for i in range(0, len(reply_text), MAX_MESSAGE_LENGTH)]
        
        channel = ctx.channel
        for msg in messages:
            if len(messages) == 1:
                await channel.send(msg, reference=ctx.original_message)
            else:
                await channel.send(msg)

    async def _handle_discord_message(self, message: DiscordMessage):
        """Handle incoming Discord message"""
        if not message.content:
            return

        ctx = DiscordMessageContext(
            channel=message.channel,
            chat_id=str(message.channel.id),
            user=message.author,
            user_id=str(message.author.id),
            message=message.content,
            message_id=message.id,
            original_message=message,
            guild=message.guild,
            progress_report=True,
            cost=0.0,
        )
        await self._message_queue.put(ctx)
