from typing import Dict, Any
from .base import BaseReflector
from .types import ReflectionType, ReflectionResult
import json

class FactReflector(BaseReflector):
    def __init__(self, llm_client: Any):
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
            {{content}}

            Response format:
            ```json
            {
                "claims": [
                    {
                        "claim": "string",
                        "type": "fact|status|opinion",
                        "confidence": float,
                        "already_known": boolean
                    }
                ]
            }
            ```
            """
        )
        self.llm_client = llm_client

    async def process_reflection(self, content: str) -> ReflectionResult[Dict[str, Any]]:
        if not content:
            return ReflectionResult(success=False, data=None, reason="Empty content")

        prompt = self.prompt_template.replace("{{content}}", content)

        try:
            response = await self.llm_client.acompletion(
                model="openai/gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            llm_response = response.choices[0].message.content
            # Parse JSON string to dict
            llm_data = json.loads(llm_response)
            
            return ReflectionResult(success=True, data={
                "type": ReflectionType.FACT.value,
                "claims": llm_data["claims"]
            })
        except Exception as e:
            return ReflectionResult(success=False, data=None, reason=str(e))