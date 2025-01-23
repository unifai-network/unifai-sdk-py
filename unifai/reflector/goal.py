from typing import Dict, Any
from .base import BaseReflector
from .types import ReflectionType, ReflectionResult
import json

class GoalReflector(BaseReflector):
    def __init__(self, llm_client: Any):
        super().__init__(
            name="GOAL_REFLECTOR",
            description="Tracks progress and updates on stated goals",
            similes=["OBJECTIVE_TRACKER", "PROGRESS_MONITOR"],
            prompt_template="""
            TASK: Analyze the conversation for goal-related updates.
            Look for:
            - Progress on existing goals
            - New goals being set
            - Completion of objectives
            - Changes in priorities

            Content:
            {{content}}

            Response format:
            ```json
            {
                "goals": [
                    {
                        "description": "string",
                        "status": "NEW|IN_PROGRESS|COMPLETED|ABANDONED",
                        "progress": float,
                        "updates": []
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
                "type": ReflectionType.GOAL.value,
                "goals": llm_data["goals"]
            })
        except Exception as e:
            return ReflectionResult(success=False, data=None, reason=str(e))