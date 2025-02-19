from typing import List, Dict, Any, Callable, Awaitable
from .base import BaseReflector
from .types import ReflectionType, ReflectionResult
import json

class GoalReflector(BaseReflector):
    def __init__(
        self, 
        chat_completion_fn,
        model: str = "gpt-4o-mini",
    ):
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
            {content}

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
        self.chat_completion_fn = chat_completion_fn
        self.model = model

    async def process_reflection(self, content: str) -> ReflectionResult[Dict[str, Any]]:
        if not content:
            return ReflectionResult(success=False, data=None, reason="Empty content")

        messages = [
            {
                "role": "user", 
                "content": self.prompt_template.format(content=content)
            }
        ]

        try:
            response = await self.chat_completion_fn(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            llm_response = response.choices[0].message.content
            llm_data = json.loads(llm_response)
            
            return ReflectionResult(
                success=True,
                data=llm_data
            )
            
        except json.JSONDecodeError as e:
            return ReflectionResult(
                success=False,
                data=None,
                reason=f"Failed to parse JSON response: {str(e)}"
            )
        except Exception as e:
            return ReflectionResult(
                success=False,
                data=None,
                reason=f"Reflection failed: {str(e)}"
            )