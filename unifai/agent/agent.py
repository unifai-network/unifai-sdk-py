from __future__ import annotations
import asyncio
import logging
from .api import API
from .data import save_data, save_to_memory
from .messaging import MessagingHandler
from .model import ModelManager
from .summary import Summarizer
from datetime import datetime
from .utils import load_prompt, load_all_prompts
from ..common.const import FRONTEND_API_ENDPOINT, BACKEND_WS_ENDPOINT
from .memory.manager import MemoryManager
from .memory.importance import ImportanceCalculator
from .memory.embedding import EmbeddingGenerator
from .memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)

class Agent:
    vision_range_buildings = 16
    vision_range_players = 16
    min_num_buildings = 10
    min_num_players = 10
    post_story = True
    post_story_interval_hours = 24
    working_memory_max_size = 10

    def __init__(self, api_key, name, data_dir='data'):
        self.api_key = api_key
        self.name = name
        self.data_dir = data_dir
        self._prompts = {}
        self._models = {
            'default': 'gpt-4o-mini',
            'embedding': 'text-embedding-3-small',
        }
        self._websocket = None
        self.model_manager = ModelManager(self)
        self.importance_calculator = ImportanceCalculator(self)
        self.embedding_generator = EmbeddingGenerator(self.model_manager)
        self.memory_manager = MemoryManager(
            importance_calculator=self.importance_calculator,
            embedding_generator=self.embedding_generator,
            agent=self,
        )
        self.planning = None
        self.working_memory = WorkingMemory(
            memory_manager=self.memory_manager,
            max_size=self.working_memory_max_size,
            agent=self,
        )
        
        self.messaging_handler = MessagingHandler(self)
        self.summarizer = Summarizer(self)
        self.api = API(self.api_key)
        self.set_api_endpoint(FRONTEND_API_ENDPOINT)
        self.set_ws_endpoint(BACKEND_WS_ENDPOINT)
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
        """
        Get all prompts used by combining default prompts with any custom prompts.

        Returns:
            dict: Combined dictionary of all prompts
        """
        prompts = load_all_prompts()
        prompts.update(self._prompts)
        return prompts

    def get_prompt(self, prompt_key):
        """
        Get a specific prompt by key, first checking custom prompts then default prompts.

        Args:
            prompt_key (str): Key of the desired prompt

        Returns:
            str: The prompt text
        """
        prompt = self._prompts.get(prompt_key)
        if not prompt:
            prompt = load_prompt(prompt_key)
        return prompt

    def set_prompt(self, prompt_key, prompt):
        """
        Set a custom prompt. Set to None or empty string to use default prompt.
        The prompt will be immediately effective even if agent is already running.

        Args:
            prompt_key (str): Key of the prompt
            prompt (str): The prompt text to set
        """
        self._prompts[prompt_key] = prompt

    def set_prompts(self, prompts):
        """
        Replace all custom prompts with a new dictionary.
        Missing keys from the provided dictionary will be loaded from default prompts.
        The prompts will be immediately effective even if agent is already running.

        Args:
            prompts (dict): Dictionary of prompts to set
        """
        self._prompts = prompts

    def update_prompts(self, prompts):
        """
        Update custom prompts by merging with provided prompts.
        The prompts will be immediately effective even if agent is already running.

        Args:
            prompts (dict): Dictionary of prompts to merge with existing prompts
        """
        self._prompts.update(prompts)

    def get_model(self, prompt_key):
        """
        Get the model for a specific prompt key, falling back to default model if not found.

        Args:
            prompt_key (str): Key to look up model for

        Returns:
            str: The model name to use
        """
        return self._models.get(prompt_key, self._models.get('default'))

    def set_model(self, prompt_key, model):
        """
        Set the model to use for a specific prompt key.
        Set the model of key 'default' to set the default model.
        The model will be immediately effective even if agent is already running.

        Args:
            prompt_key (str): Key to set model for
            model (str): Name of the model to use
        """
        self._models[prompt_key] = model

    async def get_model_response(self, prompt_key, system_prompt=None, **kwargs):
        prompt = self.get_prompt(prompt_key).format(**kwargs)
        return await self.model_manager.get_model_response(prompt, prompt_key=prompt_key, system_prompt=system_prompt)

    def set_models(self, models):
        """
        Replace all custom models with a new dictionary.
        The model of key 'default' will be used as the default model.
        The models will be immediately effective even if agent is already running.

        Args:
            models (dict): Dictionary mapping prompt keys to model names
        """
        self._models = models

    def update_models(self, models):
        """
        Update custom models by merging with provided models.
        The model of key 'default' will be used as the default model.
        The models will be immediately effective even if agent is already running.

        Args:
            models (dict): Dictionary of models to merge with existing models
        """
        self._models.update(models)

    async def _start_messaging(self):
        while not self._stop_event.is_set():
            try:
                async with await self.messaging_handler.connect(self.ws_uri) as websocket:
                    self._websocket = websocket
                    await self.messaging_handler.handle_messages(websocket, self._stop_event)
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

    async def _save_data(self, data_type, data):
        save_data(self.data_dir, self.name, data_type, data)
        await save_to_memory(data, self.memory_manager)

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
        """
        Stop the agent.
        """
        logger.info("Stopping the agent...")
        self._stop_event.set()
        if self._websocket:
            await self._websocket.close()
        if hasattr(self, '_tasks'):
            await asyncio.gather(*self._tasks, return_exceptions=True)
