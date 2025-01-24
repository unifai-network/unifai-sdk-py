from typing import Dict, Any, Callable, Awaitable, List
from .base import BaseReflector
from .types import ReflectionType, ReflectionResult
import json
import litellm

ChatCompletionFn = Callable[[List[Dict[str, Any]], Any], Awaitable[Any]]
async def default_chat_completion(
    messages: List[Dict[str, Any]], 
    tools: Any = None
) -> Any:
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    return response

class FactReflector(BaseReflector):
    def __init__(
        self, 
        chat_completion_fn: Callable[[Dict[str, Any]], Awaitable[Any]],
        model: str = "gpt-4o-mini",
    ):
        super().__init__(
            name="FACT_REFLECTOR",
            description="Extracts factual information and claims from messages",
            similes=["CLAIM_EXTRACTOR", "KNOWLEDGE_ANALYZER"],
            prompt_template="""
            TASK: Extract factual claims from the messages as JSON.
            Look for:
            - Verifiable facts
            - Stated preferences
            - Specific requirements
            - Numerical data

            Content:
            {content}

            Response format:
            ```json
            {
                "claims": [
                    {
                        "claim": "string",
                        "type": "fact|status|opinion",
                        "confidence": float
                    }
                ]
            }
            ```
            """
        )
        self.chat_completion_fn = chat_completion_fn
        self.model = model

    async def process_reflection(self, content: str) -> ReflectionResult[Dict[str, Any]]:
        if not content:
            return ReflectionResult(success=False, data=None, reason="Empty content")

        prompt = self.prompt_template.format(content=content)

        try:
            response = await self.chat_completion_fn(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
            )
            
            llm_response = response.choices[0].message.content
            llm_data = json.loads(llm_response)
            
            return ReflectionResult(success=True, data={
                "type": ReflectionType.FACT.value,
                "claims": llm_data["claims"]
            })
        except Exception as e:
            return ReflectionResult(success=False, data=None, reason=str(e))