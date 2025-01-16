from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent

import asyncio
import datetime
import json
import logging
from .data import DataTypes, load_data, load_file, save_file, save_image
from .utils import format_json

logger = logging.getLogger(__name__)

class Summarizer:
    agent: "Agent"

    def __init__(self, agent):
        self.agent = agent

    async def summarize(self, since_hours, min_responses, batch_size, concurrency, max_retries):
        character_info = self.agent.get_prompt('character.info')
        character_appearance = self.agent.get_prompt('character.appearance')
        image_style = self.agent.get_prompt('character.image_style')
        responses = load_data(
            self.agent.data_dir,
            self.agent.name,
            data_type=DataTypes.MODEL_RESPONSE,
            since=since_hours
        )

        if len(responses) < min_responses:
            raise Exception(f'Not enough responses to summarize ({len(responses)}/{min_responses}).')

        semaphore = asyncio.Semaphore(concurrency)
        batch_summaries = []

        async def process_batch(i, batch):
            async with semaphore:
                for attempt in range(max_retries):
                    try:
                        logger.debug(f'Processing batch {i // batch_size + 1}, attempt {attempt + 1}/{max_retries}')
                        response = await self.agent.get_model_response(
                            'summarize.summarize_one_batch',
                            character_name=self.agent.name,
                            character_info=character_info,
                            data_stream=format_json(batch),
                        )
                        summary = response.get('summary', '')
                        logger.info(f'Finished processing batch {i // batch_size + 1}')
                        logger.debug(f'Batch {i // batch_size + 1} summary:\n{summary}\n--------\n')
                        return summary
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f'Batch {i // batch_size + 1} failed attempt {attempt + 1}: {str(e)}. Retrying...')
                            await asyncio.sleep(5 * (2 ** attempt))
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

        for attempt in range(max_retries):
            try:
                logger.debug(f'Processing final summary, attempt {attempt + 1}/{max_retries}')
                final_summary_response = await self.agent.get_model_response(
                    'summarize.summarize_final',
                    character_name=self.agent.name,
                    character_info=character_info,
                    character_appearance=character_appearance,
                    image_style=image_style,
                    data_stream=format_json(successful_summaries),
                )
                logger.info('Finished processing final summary')
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f'Final summary failed attempt {attempt + 1}: {str(e)}. Retrying...')
                    await asyncio.sleep(5 * (2 ** attempt))
                else:
                    logger.error(f'Final summary failed all {max_retries} attempts: {str(e)}')
                    raise
        
        return final_summary_response

    async def post_story(self, since_hours):
        logger.info(f'Posting story for the last {since_hours} hours')

        logger.info(f'Summarizing data...')
        summary = await self.agent.summarize(since_hours=since_hours)

        logger.info(f'Generating image...')
        prompt = summary.get('image_prompt') + '\n' + self.agent.get_prompt('character.image_style')
        response = await self.agent.model_manager.generate_image(prompt)

        logger.info(f'Saving story...')
        summary_data = {
            'image_prompt': summary.get('image_prompt'),
            'revised_prompt': response.data[0]['revised_prompt'],
            'concise_summary': summary.get('concise_summary'),
        }
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f'{self.agent.data_dir}/{self.agent.name}_{now}'
        save_file(json.dumps(summary_data), f'{filename_base}.json')
        save_image(response.data[0]['url'], f'{filename_base}.png')

        logger.info(f'Posting story...')
        response = await self.agent.api.post_story(summary.get('concise_summary'), response.data[0]['url'])
        logger.info(f'Story posted: {response}')

        logger.info(f'Saving post time...')
        post_data = {
            'last_post_time': now,
        }
        save_file(json.dumps(post_data), self.get_story_filename())

        logger.info(f'Post story done.')

    def time_since_last_post(self):
        try:
            post_data = load_file(self.get_story_filename())
            return datetime.datetime.now() - datetime.datetime.fromisoformat(post_data.get('last_post_time'))
        except Exception as e:
            return datetime.timedelta(weeks=52*42)

    def get_story_filename(self):
        return f'{self.agent.data_dir}/{self.agent.name}_story.json'
