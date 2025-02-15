from __future__ import annotations
import asyncio
import logging
import os
from telegram.ext import ApplicationBuilder
from .model import ModelManager
from .utils import load_prompt, load_all_prompts
from ..common.const import BACKEND_WS_ENDPOINT

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, api_key):
        self.api_key = api_key
        self._prompts = {}
        self._models = {
            'default': 'gpt-4o',
        }
        self.model_manager = ModelManager()
        self.set_ws_endpoint(BACKEND_WS_ENDPOINT)
        self._stop_event = asyncio.Event()
        self._tasks = []

    def set_ws_endpoint(self, endpoint):
        self.ws_uri = f"{endpoint}?type=player&api-key={self.api_key}"

    def set_chat_completion_function(self, f):
        self.model_manager.set_chat_completion_function(f)

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

    def run(self):
        """
        Synchronous entry point to run the agent.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.start())
        except KeyboardInterrupt:
            logger.info("Agent interrupted by user. Shutting down...")
            loop.run_until_complete(self.stop())
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def start(self):
        """
        Asynchronous entry point to run the agent.
        """
        self._tasks = [
            asyncio.create_task(self._start_telegram()),
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
        if hasattr(self, '_tasks'):
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _start_telegram(self):
        while not self._stop_event.is_set():
            try:
                application = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN', '')).build()
                started = False
                await application.initialize()
                await application.start()
                await application.updater.start_polling() # type: ignore
                started = True
                await self._stop_event.wait()
            except asyncio.CancelledError:
                logger.info("Telegram task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            finally:
                if started:
                    await application.updater.stop() # type: ignore
                    await application.stop()
                    await application.shutdown()