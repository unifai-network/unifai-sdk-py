import asyncio
import datetime
import logging
import litellm
from litellm.cost_calculator import completion_cost

logger = logging.getLogger(__name__)

litellm.drop_params = True

class ModelManager:
    def __init__(self):
        self.usage_history = []
        self.max_history_hours = 24
        self._chat_completion = litellm.acompletion
        self._completion_cost_calculator = completion_cost

    def set_chat_completion_function(self, f):
        self._chat_completion = f

    def set_completion_cost_calculator(self, f):
        self._completion_cost_calculator = f

    async def chat_completion(self, model, messages, timeout: float | None = None, retries=3, **kwargs):
        attempt = 0
        last_error = None
        
        while attempt < retries:
            try:
                response = await asyncio.wait_for(
                    self._chat_completion(
                        model=model,
                        messages=messages,
                        **kwargs
                    ),
                    timeout=timeout,
                )

                cached_tokens = response.usage.prompt_tokens_details.cached_tokens if response.usage.prompt_tokens_details else 0 # type: ignore
                input_tokens = response.usage.prompt_tokens # type: ignore
                output_tokens = response.usage.completion_tokens # type: ignore
                cost = 0
                try:
                    cost = self._completion_cost_calculator(response, model=model)
                except Exception as e:
                    logger.error(f'Error calculating cost: {e}')

                try:
                    logger.info(f'Cached tokens: {cached_tokens}, input tokens: {input_tokens}, output tokens: {output_tokens}, cost: {cost}')
                    current_time = datetime.datetime.now()
                    self.usage_history.append((current_time, cached_tokens, input_tokens, output_tokens, cost))
                    cutoff_time = current_time - datetime.timedelta(hours=self.max_history_hours)
                    self.usage_history = [stat for stat in self.usage_history if stat[0] > cutoff_time]
                    stats = self.get_usage_stats(hours=1)
                    logger.info(f'Last hour cached tokens: {stats["cached_tokens"]}, input tokens: {stats["input_tokens"]}, output tokens: {stats["output_tokens"]}, cost: {stats["cost"]}')
                    stats = self.get_usage_stats(hours=24)
                    logger.info(f'Last 24 hours cached tokens: {stats["cached_tokens"]}, input tokens: {stats["input_tokens"]}, output tokens: {stats["output_tokens"]}, cost: {stats["cost"]}')
                except Exception as e:
                    logger.error(f'Error updating usage stats: {e}')
                
                return response, cost
                
            except Exception as e:
                attempt += 1
                last_error = e
                if attempt < retries:
                    wait_time = 10 * 2 ** attempt
                    logger.warning(f"Attempt {attempt} failed with error: {e}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {retries} attempts failed. Last error: {e}")
                    raise last_error

        return None, 0

    def get_usage_stats(self, hours=None):
        """Get usage statistics for the specified number of hours."""
        if not self.usage_history:
            return {'cached_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0}

        if hours:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            filtered_stats = [stat for stat in self.usage_history if stat[0] > cutoff_time]
        else:
            filtered_stats = self.usage_history

        return {
            'cached_tokens': sum(stat[1] for stat in filtered_stats),
            'input_tokens': sum(stat[2] for stat in filtered_stats),
            'output_tokens': sum(stat[3] for stat in filtered_stats),
            'cost': sum(stat[4] for stat in filtered_stats),
        }
