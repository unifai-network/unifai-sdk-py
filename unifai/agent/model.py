import asyncio
import datetime
import logging
import litellm

logger = logging.getLogger(__name__)

litellm.drop_params = True

class ModelManager:
    def __init__(self):
        self.usage_history = []
        self.max_history_hours = 24
        self._chat_completion = litellm.acompletion

    def set_chat_completion_function(self, f):
        self._chat_completion = f

    async def chat_completion(self, model, messages, timeout=60, **kwargs):
        response = await asyncio.wait_for(
            self._chat_completion(
                model=model,
                messages=messages,
                **kwargs
            ),
            timeout=timeout,
        )

        try:
            logger.info(f'Input tokens: {response.usage.prompt_tokens}, output tokens: {response.usage.completion_tokens}')  # type: ignore
            current_time = datetime.datetime.now()
            self.usage_history.append((current_time, response.usage.prompt_tokens, response.usage.completion_tokens))  # type: ignore
            cutoff_time = current_time - datetime.timedelta(hours=self.max_history_hours)
            self.usage_history = [stat for stat in self.usage_history if stat[0] > cutoff_time]
            stats = self.get_usage_stats(hours=1)
            logger.info(f'Last hour input tokens: {stats["input_tokens"]}, output tokens: {stats["output_tokens"]}')
            stats = self.get_usage_stats(hours=24)
            logger.info(f'Last 24 hours input tokens: {stats["input_tokens"]}, output tokens: {stats["output_tokens"]}')
        except Exception as e:
            logger.error(f'Error updating usage stats: {e}')

        return response

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