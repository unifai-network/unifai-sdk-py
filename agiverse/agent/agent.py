import asyncio
import logging
from .data import save_data
from .messaging import MessagingHandler
from .model import ModelManager
from .summary import Summarizer
from .utils import load_prompt, load_all_prompts
from ..common.const import DEFAULT_WS_ENDPOINT

logger = logging.getLogger(__name__)

class Agent:
    vision_range_buildings = 16
    vision_range_players = 16

    def __init__(self, api_key, name, data_dir='data', prompts={}):
        self.api_key = api_key
        self.name = name
        self.data_dir = data_dir
        self._prompts = prompts
        self.messaging_handler = MessagingHandler(self)
        self.model_manager = ModelManager(self)
        self.summarizer = Summarizer(self)
        self.memory = []
        self.long_term_memory = None
        self.planning = None
        self.set_ws_endpoint(DEFAULT_WS_ENDPOINT)

    def set_ws_endpoint(self, endpoint):
        self.ws_uri = f"{endpoint}?type=player&api-key={self.api_key}&name={self.name}"

    def run(self):
        asyncio.run(self._start())

    async def summarize(self, since_hours=24, batch_size=256, concurrency=10, max_retries=3):
        return await self.summarizer.summarize(since_hours, batch_size, concurrency, max_retries)

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

    async def _start(self):
        while True:
            try:
                async with await self.messaging_handler.connect(self.ws_uri) as websocket:
                    await self.messaging_handler.handle_messages(websocket)
            except Exception as e:
                logger.error(f"Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def _save_data(self, data_type, data):
        save_data(self.data_dir, self.name, data_type, data)
