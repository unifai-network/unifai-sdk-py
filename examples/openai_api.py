from dotenv import load_dotenv
load_dotenv()

from typing import Dict
import json
import time
import uuid
import litellm
import asyncio
import logging
import os
import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from unifai.agent import Agent
from unifai.tools import Tools

logger = logging.getLogger(__name__)

system_fingerprint = "fp_0000000001"

class OpenAIAPI():
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str = "",
        add_system_prompt: bool = False,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
    ):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.security = HTTPBearer(auto_error=False)
        self.response_queues: Dict[str, asyncio.Queue] = {}
        self.server = None
        self.server_task = None
        self.add_system_prompt = add_system_prompt
        self.model = model
        self.api_key = api_key

        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            if not await self.verify_credentials(credentials):
                raise HTTPException(status_code=401, detail="Unauthorized")

            try:
                request_data = await request.json()

                messages = []

                if self.add_system_prompt:
                    system_prompt = Agent("").get_prompt("agent.system")
                    messages.append({"content": system_prompt, "role": "system"})

                messages.extend(request_data.get("messages", []))

                tools = Tools(api_key=(credentials.credentials or self.api_key) if credentials else self.api_key)

                model = request_data.get("model", self.model)

                completion_id = f"chatcmpl-{uuid.uuid4()}"
                prompt_tokens = 0
                completion_tokens = 0
                finish_reason = ""

                while True:
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        tools=await tools.get_tools(),
                    )

                    try:
                        prompt_tokens += response.usage.prompt_tokens # type: ignore
                        completion_tokens += response.usage.completion_tokens # type: ignore
                        finish_reason = response.choices[0].finish_reason # type: ignore
                    except Exception as e:
                        logger.error(f"Error calculating cost: {e}")

                    message = response.choices[0].message # type: ignore
                    messages.append(message.model_dump(mode="json"))

                    if not message.tool_calls:
                        break

                    results = await tools.call_tools(message.tool_calls) # type: ignore

                    if len(results) == 0:
                        break

                    messages.extend(results)

                response_data = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": messages[-1],
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    "system_fingerprint": system_fingerprint,
                }

                return Response(content=json.dumps(response_data), media_type="application/json")
            except Exception as e:
                logger.error(f"Error in chat_completions: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def verify_credentials(self, credentials: HTTPAuthorizationCredentials):
        return True

    async def start(self):
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio"
        )
        self.server = uvicorn.Server(config)
        self.server.config.setup_event_loop()
        self.server_task = asyncio.create_task(self.server.serve())
        await self.server_task

    async def stop(self):
        if self.server is not None:
            self.server.should_exit = True

            if self.server_task is not None and not self.server_task.done():
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error during server shutdown: {e}")

            for request_id in list(self.response_queues.keys()):
                if request_id in self.response_queues:
                    await self.response_queues[request_id].put(None)
                    del self.response_queues[request_id]

if __name__ == "__main__":
    api = OpenAIAPI(add_system_prompt=True, api_key=os.getenv("UNIFAI_API_KEY", ""))
    asyncio.run(api.start())
