from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent

import asyncio
import logging
from .data import DataTypes, load_data
from .utils import format_json

logger = logging.getLogger(__name__)

class Summarizer:
    agent: "Agent"

    def __init__(self, agent):
        self.agent = agent

    async def summarize(self, since_hours, batch_size, concurrency, max_retries):
        character_info = self.agent.get_prompt('character.info')
        responses = load_data(
            self.agent.data_dir,
            self.agent.name,
            data_type=DataTypes.MODEL_RESPONSE,
            since=since_hours
        )

        semaphore = asyncio.Semaphore(concurrency)
        batch_summaries = []

        async def process_batch(i, batch):
            async with semaphore:
                for attempt in range(max_retries):
                    try:
                        logger.debug(f'Processing batch {i // batch_size + 1}, attempt {attempt + 1}/{max_retries}')
                        prompt = self.agent.get_prompt('summarize.summarize_one_batch').format(
                            character_name=self.agent.name,
                            character_info=character_info,
                            data_stream=format_json(batch),
                        )
                        response = await self.agent.model_manager.get_model_response(prompt)
                        summary = response.get('summary', '')
                        logger.info(f'Finished processing batch {i // batch_size + 1}')
                        logger.debug(f'Batch {i // batch_size + 1} summary:\n{summary}\n--------\n')
                        return summary
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f'Batch {i // batch_size + 1} failed attempt {attempt + 1}: {str(e)}. Retrying...')
                            await asyncio.sleep(1 * (attempt + 1))
                        else:
                            logger.error(f'Batch {i // batch_size + 1} failed all {max_retries} attempts: {str(e)}')
                            return None

        tasks = []
        for i in range(0, len(responses), batch_size):
            batch = responses[i:i + batch_size]
            task = asyncio.create_task(process_batch(i, batch))
            tasks.append(task)

        logger.info(f'Splitting {len(responses)} responses into {len(tasks)} batches')

        batch_summaries = await asyncio.gather(*tasks, return_exceptions=True)

        successful_summaries = [s for s in batch_summaries if s is not None and not isinstance(s, Exception)]

        if not successful_summaries:
            raise Exception("All batches failed after retries. Cannot generate final summary.")

        final_prompt = self.agent.get_prompt('summarize.summarize_final').format(
            character_name=self.agent.name,
            character_info=character_info,
            data_stream=format_json(successful_summaries),
        )
        final_summary_response = await self.agent.model_manager.get_model_response(final_prompt)
        
        return final_summary_response
