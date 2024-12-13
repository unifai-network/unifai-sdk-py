from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent

import asyncio
import datetime
import json
import logging
import os
import litellm

logger = logging.getLogger(__name__)

litellm.drop_params = True

class ModelManager:
    agent: "Agent"

    def __init__(self, agent):
        self.agent = agent
        self.usage_history = []
        self.max_history_hours = 24
        self.chat_completion = litellm.acompletion
        self.image_generation = litellm.aimage_generation
        self.embedding = litellm.aembedding

    def set_chat_completion_function(self, f):
        self.chat_completion = f

    def set_embedding_function(self, f):
        self.embedding = f

    def set_image_generation_function(self, f):
        self.image_generation = f

    async def get_model_response(self, prompt, prompt_key='default', system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await asyncio.wait_for(
            self.chat_completion(
                model=self.agent.get_model(prompt_key),
                messages=messages,
                response_format={"type": "json_object"},
            ),
            timeout=os.getenv('MODEL_TIMEOUT', 60),
        )

        try:
            logger.info(f'Input tokens: {response.usage.prompt_tokens}, output tokens: {response.usage.completion_tokens}')
            current_time = datetime.datetime.now()
            self.usage_history.append((current_time, response.usage.prompt_tokens, response.usage.completion_tokens))
            cutoff_time = current_time - datetime.timedelta(hours=self.max_history_hours)
            self.usage_history = [stat for stat in self.usage_history if stat[0] > cutoff_time]
            stats = self.get_usage_stats(hours=1)
            logger.info(f'Last hour input tokens: {stats["input_tokens"]}, output tokens: {stats["output_tokens"]}')
            stats = self.get_usage_stats(hours=24)
            logger.info(f'Last 24 hours input tokens: {stats["input_tokens"]}, output tokens: {stats["output_tokens"]}')
        except Exception as e:
            logger.error(f'Error updating usage stats: {e}')
        
        content = response.choices[0].message.content
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3]
        return json.loads(content)
    
    def get_usage_stats(self, hours=None):
        """Get usage statistics for the specified number of hours."""
        if not self.usage_history:
            return {'input_tokens': 0, 'output_tokens': 0}

        if hours:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            filtered_stats = [stat for stat in self.usage_history if stat[0] > cutoff_time]
        else:
            filtered_stats = self.usage_history
        
        return {
            'input_tokens': sum(stat[1] for stat in filtered_stats),
            'output_tokens': sum(stat[2] for stat in filtered_stats)
        }

    async def generate_image(self, prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.image_generation(
                        model='dall-e-3',
                        quality='hd',
                        size='1792x1024',
                        prompt=prompt,
                    ),
                    timeout=os.getenv('IMAGE_GENERATION_TIMEOUT', 120),
                )
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.error(f"Attempt {attempt + 1} failed. Retrying... Error: {e}")
                await asyncio.sleep(5 * (2 ** attempt))
