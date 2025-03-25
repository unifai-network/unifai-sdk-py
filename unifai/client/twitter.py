import asyncio
import logging
import datetime
import time
import tweepy
from dataclasses import dataclass
from functools import wraps
from typing import List, Optional

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
class TwitterMessageContext(MessageContext):
    tweet_id: str
    chat_id: str
    user_id: str
    username: str
    message: str
    author_name: str

class TwitterClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_secret: str,
        bearer_token: str,
        bot_screen_name: str,
        poll_interval: int = 20,
        max_message_length: int = 280,
        respond_to_mentions: bool = True,
        respond_to_replies: bool = False,
        search_query: str | None = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.bearer_token = bearer_token
        self.bot_screen_name = bot_screen_name
        self.poll_interval = poll_interval
        self.max_message_length = max_message_length
        self.respond_to_mentions = respond_to_mentions
        self.respond_to_replies = respond_to_replies
        self.search_query = search_query
        self._started = False
        self._message_queue: asyncio.Queue[TwitterMessageContext] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._polling_task = None
        
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_secret,
            wait_on_rate_limit = True
        )

    @property
    def client_id(self) -> str:
        return f"twitter-{self.bot_screen_name}"

    async def start(self):
        """Start the client"""
        if self._started:
            return
        
        self._started = True
        self._stop_event.clear()
        self._polling_task = asyncio.create_task(self._poll_mentions())
        logger.info(f"Twitter client {self.client_id} started")

    async def stop(self):
        """Stop the client"""
        if not self._started:
            return
            
        self._stop_event.set()
        if self._polling_task:
            await self._polling_task
            self._polling_task = None
        
        self._started = False
        logger.info(f"Twitter client {self.client_id} stopped")

    @ensure_started
    async def receive_message(self) -> Optional[TwitterMessageContext]:
        """Receive a message from the queue"""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=None)
        except asyncio.TimeoutError:
            return None

    @ensure_started
    async def send_message(self, ctx: TwitterMessageContext, reply_messages: List[Message]):
        """Send a message using the context"""
        if reply_messages and reply_messages[-1].content:
            reply_text = reply_messages[-1].content
        else:
            logger.warning(f"No content in reply_messages for tweet {ctx.tweet_id}")
            return

        if len(reply_text) > self.max_message_length:
            reply_text = reply_text[:self.max_message_length-3] + "..."

        try:
            self.client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=ctx.tweet_id
            )
            logger.info(f"Replied to tweet ID={ctx.tweet_id}")
        except Exception as e:
            logger.error(f"Error replying to tweet {ctx.tweet_id}: {e}")

    async def _poll_mentions(self):
        """Poll for mentions and add them to the queue"""
        start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
        since_id = None

        query = self.search_query
        if not query:
            query_parts = []
            if self.respond_to_mentions:
                query_parts.append(f"@{self.bot_screen_name}")
            if self.respond_to_replies:
                query_parts.append(f"to:{self.bot_screen_name}")
            if query_parts:
                query = f"({' OR '.join(query_parts)}) -from:{self.bot_screen_name} -is:retweet -is:quote"
            else:
                return
        
        base_wait_time = self.poll_interval
        max_wait_time = base_wait_time
        current_wait_time = base_wait_time

        while not self._stop_event.is_set():
            try:
                tweets_resp = self.client.search_recent_tweets(
                    query=query,
                    expansions=["author_id"],
                    tweet_fields=["author_id", "text", "conversation_id"],
                    user_fields=["username", "name", "description"],
                    start_time=start_time if not since_id else None,
                    since_id=since_id,
                    max_results=10,
                )
                
                current_wait_time = base_wait_time
                
                if isinstance(tweets_resp, tweepy.Response) and tweets_resp.data:
                    users = {user.id: user for user in tweets_resp.includes['users']}
                    
                    for tweet in reversed(tweets_resp.data):
                        since_id = max(since_id or 0, tweet.id)
                        if not tweet.author_id:
                            continue
                            
                        author = users.get(tweet.author_id)

                        if not author:
                            continue
                        
                        if author.username.lower() == self.bot_screen_name.lower():
                            continue
                            
                        logger.info(f"Processing tweet {tweet.id} by {author.username}")
                        
                        ctx = TwitterMessageContext(
                            tweet_id=str(tweet.id),
                            chat_id=str(tweet.conversation_id),
                            user_id=str(tweet.author_id),
                            username=author.username,
                            message=tweet.text,
                            author_name=author.name,
                            progress_report=False,
                            cost=0.0,
                        )
                        
                        await self._message_queue.put(ctx)
                        
            except Exception as e:
                logger.error(f"Error polling Twitter: {e}")
                
            current_wait_time = min(current_wait_time * 2, max_wait_time)
            logger.debug(f"Sleeping {current_wait_time} seconds.")
            await asyncio.sleep(current_wait_time)
