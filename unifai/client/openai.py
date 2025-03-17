from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import time
import uuid
import litellm
import asyncio
import logging
import os
from dataclasses import dataclass
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from dotenv import load_dotenv

from .base import BaseClient, Message, MessageContext
from unifai.tools import Tools

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class OpenAIMessageContext(MessageContext):
    chat_id: str
    user_id: str
    message: str
    progress_report: bool
    
    request_data: Dict[str, Any]
    
    use_agent_system_prompt: bool = False
    include_tool_history: bool = False

class OpenAIClient(BaseClient):
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, api_key: str = None):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.security = HTTPBearer(auto_error=False)
        self.message_queue: asyncio.Queue[OpenAIMessageContext] = asyncio.Queue()
        self.response_queues: Dict[str, asyncio.Queue] = {}
        self.server = None
        self.server_task = None
        
        self.api_key = api_key or os.getenv("UNIFAI_API_KEY")
        
        self._setup_routes()
    
    @property
    def client_id(self) -> str:
        return "openai"
    
    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            if not await self.verify_credentials(credentials):
                raise HTTPException(status_code=401, detail="Unauthorized")

            try:
                request_data = await request.json()
                
                use_agent_system_prompt = request_data.get("use_agent_system_prompt", False)
                include_tool_history = request_data.get("include_tool_history", False)
                progress_report = request_data.get("progress_report", False)
                
                request_id = str(uuid.uuid4())
                
                messages = request_data.get("messages", [])
                user_message = ""
                system_message = None
                
                for msg in messages:
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                    elif msg.get("role") == "system":
                        system_message = msg.get("content", "")
                
                model_messages = []
                
                if use_agent_system_prompt:
                    from unifai.agent import Agent
                    system_prompt = Agent("").get_prompt("agent.system")
                    model_messages.append({"content": system_prompt, "role": "system"})
                elif system_message:
                    model_messages.append({"content": system_message, "role": "system"})
                
                if user_message:
                    model_messages.append({"content": user_message, "role": "user"})
                
                tools_data = request_data.get("tools", [])
                
                ctx = OpenAIMessageContext(
                    chat_id=request_id,
                    user_id=request_id,
                    message=user_message,
                    progress_report=progress_report,
                    request_data=request_data,
                    use_agent_system_prompt=use_agent_system_prompt,
                    include_tool_history=include_tool_history,
                )
                
                tools = Tools(api_key=self.api_key)
                
                completion_id = f"chatcmpl-{uuid.uuid4()}"
                system_fingerprint = f"fp_{uuid.uuid4().hex[:11]}"
                model = request_data.get("model", "default")
                created_timestamp = int(time.time())
                
                all_messages = model_messages.copy()
                result_messages = []
                
                model_name = request_data.get("model", "anthropic/claude-3-7-sonnet-20250219")
                
                
                
                while True:
                    response = await litellm.acompletion(
                        model=model_name,
                        messages=all_messages,
                        tools=tools_data if tools_data else None,
                    )
                    
                    message = response.choices[0].message
                    result_messages.append(message)
                    
                    all_messages.append(message)
                    
                    if not message.tool_calls:
                        break
                    
                    if tools_data:
                        results = await tools.call_tools(message.tool_calls)
                        
                        if len(results) == 0:
                            break
                        
                        all_messages.extend(results)
                        result_messages.extend(results)
                
                final_messages = result_messages if include_tool_history else [result_messages[-1]]
                
                prompt_tokens = len(user_message) // 4
                completion_tokens = sum(len(msg.content or "") if hasattr(msg, "content") else len(msg.get("content", "")) for msg in final_messages) // 4
                total_tokens = prompt_tokens + completion_tokens
                
                choices = []
                for i, message in enumerate(final_messages):
                    content = message.content if hasattr(message, "content") else message.get("content", "")
                    
                    # Handle tool_calls serialization
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        # Convert tool_calls objects to dictionaries
                        serialized_tool_calls = []
                        for tool_call in message.tool_calls:
                            # Create a serializable dictionary from the tool_call object
                            tool_call_dict = {
                                "id": tool_call.id if hasattr(tool_call, "id") else str(uuid.uuid4()),
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name if hasattr(tool_call.function, "name") else "",
                                    "arguments": tool_call.function.arguments if hasattr(tool_call.function, "arguments") else "{}"
                                }
                            }
                            serialized_tool_calls.append(tool_call_dict)
                    elif isinstance(message, dict) and message.get("tool_calls"):
                        # If it's already a dictionary, use it directly
                        serialized_tool_calls = message.get("tool_calls")
                    else:
                        serialized_tool_calls = None
                    
                    choices.append({
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": content or "",
                            "function_call": None,
                            "tool_calls": serialized_tool_calls
                        },
                        "logprobs": None,
                        "finish_reason": "stop" if i == len(final_messages) - 1 else "tool_calls"
                    })
                
                response_data = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created_timestamp,
                    "model": model,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "system_fingerprint": system_fingerprint,
                    "warning": None
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
    
    async def receive_message(self) -> Optional[OpenAIMessageContext]:
        pass
    
    async def send_message(self, ctx: MessageContext, reply_messages: List[Message]):
        pass
