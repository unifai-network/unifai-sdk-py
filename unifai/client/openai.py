from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import time
import uuid
import asyncio
from dataclasses import dataclass
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .base import BaseClient, Message, MessageContext

@dataclass
class OpenAIMessageContext(MessageContext):
    request_data: Dict[str, Any]
    stream: bool = False

class OpenAIClient(BaseClient):
    def __init__(self, api_key: str = "", host: str = "0.0.0.0", port: int = 8000):
        self.api_key = api_key
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.security = HTTPBearer(auto_error=False)
        self.message_queue: asyncio.Queue[OpenAIMessageContext] = asyncio.Queue()
        self.response_queues: Dict[str, asyncio.Queue] = {}
        
        self._setup_routes()
    
    @property
    def client_id(self) -> str:
        return "openai"
    
    def _setup_routes(self):
        @self.app.post("/v1/completions")
        async def completions(request: Request, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            request_data = await request.json()
            stream = request_data.get("stream", False)
            
            request_id = str(uuid.uuid4())
            response_queue: asyncio.Queue = asyncio.Queue()
            self.response_queues[request_id] = response_queue
            
            ctx = OpenAIMessageContext(
                chat_id=request_id,
                user_id=request_id,
                message=request_data.get("prompt", ""),
                request_data=request_data,
                stream=stream
            )
            
            await self.message_queue.put(ctx)
            
            if stream:
                return StreamingResponse(
                    self._stream_response(request_id, request_data),
                    media_type="text/event-stream"
                )
            else:
                response_data = await response_queue.get()
                return Response(content=json.dumps(response_data), media_type="application/json")
    
    async def _stream_response(self, request_id: str, request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        response_queue = self.response_queues[request_id]
        
        try:
            while True:
                chunk = await response_queue.get()
                if chunk is None: 
                    break
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if request_id in self.response_queues:
                del self.response_queues[request_id]
    
    def _verify_api_key(self, credentials: HTTPAuthorizationCredentials):
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
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            print("The server is shutting down...")
        
        self.server_task = asyncio.create_task(self.server.serve())
    
    async def stop(self):
        if hasattr(self, 'server'):
            self.server.should_exit = True
            if hasattr(self, 'server_task') and not self.server_task.done():
                try:
                    await asyncio.wait_for(self.server_task, timeout=2.0)
                except asyncio.TimeoutError:
                    self.server_task.cancel()
                    try:
                        await self.server_task
                    except asyncio.CancelledError:
                        pass
    
    async def receive_message(self) -> Optional[OpenAIMessageContext]:
        """Receive a message from the queue"""
        try:
            return await self.message_queue.get()
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None
    
    async def send_message(self, ctx: MessageContext, reply_messages: List[Message]):
        """Send a response back to the client"""
        if not isinstance(ctx, OpenAIMessageContext):
            raise ValueError("Context must be an OpenAIMessageContext")
        
        request_id = ctx.chat_id
        if request_id not in self.response_queues:
            return
        
        response_queue = self.response_queues[request_id]
        
        if ctx.stream:
            for i, message in enumerate(reply_messages):
                chunk = {
                    "id": f"cmpl-{uuid.uuid4()}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": ctx.request_data.get("model", "default"),
                    "system_fingerprint": f"fp_{uuid.uuid4().hex[:11]}",
                    "choices": [
                        {
                            "text": message.content,
                            "index": i,
                            "logprobs": None,
                            "finish_reason": "stop" if i == len(reply_messages) - 1 else None
                        }
                    ]
                }
                await response_queue.put(chunk)
            
            await response_queue.put(None)
        else:
            combined_text = "".join([msg.content or "" for msg in reply_messages])
            
            response_data = {
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": ctx.request_data.get("model", "default"),
                "system_fingerprint": f"fp_{uuid.uuid4().hex[:11]}",
                "choices": [
                    {
                        "text": combined_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(ctx.message) // 4, 
                    "completion_tokens": len(combined_text) // 4, 
                    "total_tokens": (len(ctx.message) + len(combined_text)) // 4 
                }
            }
            
            await response_queue.put(response_data) 