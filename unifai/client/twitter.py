import asyncio
import logging
import datetime
import os
import tempfile
import time
import tweepy
from dataclasses import dataclass
from functools import wraps
from typing import List, Dict, Any, Optional

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
        poll_interval: int = 20
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.bearer_token = bearer_token
        self.bot_screen_name = bot_screen_name
        self.poll_interval = poll_interval
        
        self._started = False
        self._message_queue = asyncio.Queue()
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
            return await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    @ensure_started
    async def send_message(self, ctx: TwitterMessageContext, reply_messages: List[Message]):
        """Send a message using the context"""
        if not reply_messages:
            return
            
        combined_text = "\n".join([msg.content for msg in reply_messages if msg.content])
        
        if len(combined_text) > 280:
            combined_text = combined_text[:277] + "..."
            
        try:
           
            self.client.create_tweet(
                text=combined_text,
                in_reply_to_tweet_id=ctx.tweet_id
            )
            logger.info(f"Replied to tweet ID={ctx.tweet_id}")
        except Exception as e:
            logger.error(f"Error replying to tweet {ctx.tweet_id}: {e}")

    async def _poll_mentions(self):
        """Poll for mentions and add them to the queue"""
        start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
        since_id = None
        query = f"(@{self.bot_screen_name} OR to:{self.bot_screen_name}) -is:retweet -is:quote"
        
        base_wait_time = self.poll_interval
        max_wait_time = 900  
        current_wait_time = base_wait_time

        while not self._stop_event.is_set():
            try:
                tweets_resp = self.client.search_recent_tweets(
                    query=query,
                    expansions=["author_id"],
                    tweet_fields=["author_id", "text"],
                    user_fields=["username", "name", "description"],
                    start_time=start_time if not since_id else None,
                    since_id=since_id,
                    max_results=10,
                )
                
                current_wait_time = base_wait_time
                
                if tweets_resp.data:
                    users = {user.id: user for user in tweets_resp.includes['users']}
                    
                    for tweet in reversed(tweets_resp.data):
                        since_id = max(since_id or 0, tweet.id)
                        if not tweet.author_id:
                            continue
                            
                        author = users.get(tweet.author_id)
                        
                        if author.username.lower() == self.bot_screen_name.lower():
                            continue
                            
                        logger.info(f"Processing tweet {tweet.id} by {author.username}")
                        
                        ctx = TwitterMessageContext(
                            tweet_id=str(tweet.id),
                            chat_id=str(tweet.id),  
                            user_id=str(tweet.author_id),
                            username=author.username,
                            message=tweet.text,
                            author_name=author.name
                        )
                        
                        await self._message_queue.put(ctx)
                        
            except Exception as e:
                logger.error(f"Error polling Twitter: {e}")
                
            current_wait_time = min(current_wait_time * 2, max_wait_time)
            logger.debug(f"Sleeping {current_wait_time} seconds.")
            time.sleep(current_wait_time)



