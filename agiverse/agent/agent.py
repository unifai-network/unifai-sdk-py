import asyncio
import logging
from .api import API
from .data import save_data
from .messaging import MessagingHandler
from .model import ModelManager
from .summary import Summarizer
from .utils import load_prompt, load_all_prompts
from ..common.const import DEFAULT_WS_ENDPOINT, DEFAULT_API_ENDPOINT

logger = logging.getLogger(__name__)

class Agent:
    vision_range_buildings = 16
    vision_range_players = 16
    min_num_buildings = 10
    min_num_players = 10
    post_story = True
    post_story_interval_hours = 24

    def __init__(self, api_key, name, data_dir='data'):
        self.api_key = api_key
        self.name = name
        self.data_dir = data_dir
        self._prompts = {}
        self._websocket = None
        self.messaging_handler = MessagingHandler(self)
        self.model_manager = ModelManager(self)
        self.summarizer = Summarizer(self)
        self.memory = []
        self.long_term_memory = None
        self.planning = None
        self.api = API(self.api_key)
        self.set_ws_endpoint(DEFAULT_WS_ENDPOINT)
        self.set_api_endpoint(DEFAULT_API_ENDPOINT)
        self._stop_event = asyncio.Event()
        self._tasks = []

    def set_ws_endpoint(self, endpoint):
        self.ws_uri = f"{endpoint}?type=player&api-key={self.api_key}&name={self.name}"

    def set_api_endpoint(self, endpoint):
        self.api.set_endpoint(endpoint)

    def set_chat_completion_function(self, f):
        self.model_manager.set_chat_completion_function(f)

    def set_image_generation_function(self, f):
        self.model_manager.set_image_generation_function(f)

    async def summarize(self, since_hours=24, min_responses=10, batch_size=256, concurrency=8, max_retries=4):
        return await self.summarizer.summarize(since_hours, min_responses, batch_size, concurrency, max_retries)

    def get_all_prompts(self):
        prompts = load_all_prompts()
        prompts.update(self._prompts)
        return prompts

    def get_prompt(self, prompt_path):
        prompt = self._prompts.get(prompt_path)
        if not prompt:
            prompt = load_prompt(prompt_path)
        return prompt

    def set_prompt(self, prompt_path, prompt):
        self._prompts[prompt_path] = prompt

    def set_prompts(self, prompts):
        self._prompts.update(prompts)

    async def _start_messaging(self):
        while not self._stop_event.is_set():
            try:
                async with await self.messaging_handler.connect(self.ws_uri) as websocket:
                    self._websocket = websocket
                    await self.messaging_handler.handle_messages(websocket)
            except asyncio.CancelledError:
                logger.info("Messaging task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _start_summarizer(self):
        if not self.post_story:
            return
        while not self._stop_event.is_set():
            await asyncio.sleep(60)
            try:
                time_since_last_post = self.summarizer.time_since_last_post()
                if time_since_last_post.total_seconds() >= self.post_story_interval_hours * 3600:
                    await self.summarizer.post_story(since_hours=self.post_story_interval_hours)
            except asyncio.CancelledError:
                logger.info("Summarizer task cancelled.")
                break
            except Exception as e:
                logger.error(f"Post story failed: {e}")

    def _save_data(self, data_type, data):
        save_data(self.data_dir, self.name, data_type, data)

    def run(self):
        """
        Synchronous entry point to run the agent.
        """
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            logger.info("Agent interrupted by user. Shutting down...")
            asyncio.run(self.stop())

    async def start(self):
        """
        Asynchronous entry point to run the agent.
        """
        self._tasks = [
            asyncio.create_task(self._start_messaging()),
            asyncio.create_task(self._start_summarizer())
        ]
        await self._stop_event.wait()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Agent has been stopped.")

    async def stop(self):
        logger.info("Stopping the agent...")
        self._stop_event.set()
