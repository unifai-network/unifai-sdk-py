import asyncio
import json
import logging
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from ..common.const import BACKEND_API_ENDPOINT
from .api import ToolsAPI

logger = logging.getLogger(__name__)

class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str
    function: Function

class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str

class OpenAIToolCall(BaseModel):
    id: str
    type: str
    function: OpenAIFunctionCall

class OpenAIToolResult(BaseModel):
    role: str
    tool_call_id: str
    content: str

class FunctionName(Enum):
    SEARCH_TOOLS = "search_tools"
    CALL_TOOL = "call_tool"

function_list: List[Function] = [
    Function(
        name=FunctionName.SEARCH_TOOLS,
        description="Search for tools. The tools cover a wide range of domains include data source, API, SDK, etc. Try searching whenever you need to use a tool.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for tools, you can describe what you want to do or what tools you want to use"
                },
                "limit": {
                    "type": "number",
                    "description": "The maximum number of tools to return, must be between 1 and 100, default is 10, recommend at least 10"
                }
            },
            "required": ["query"],
        },
    ),
    Function(
        name=FunctionName.CALL_TOOL,
        description="Call a tool returned by search_tools",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The exact action you want to call in the search_tools result."
                },
                "payload": {
                    "type": "string",
                    "description": "Action payload, based on the payload schema in the search_tools result. You can pass either the json object directly or json encoded string of the object.",
                },
                "payment": {
                    "type": "number",
                    "description": "Amount to authorize in USD. Positive number means you will be charged no more than this amount, negative number means you are requesting to get paid for at least this amount.",
                }
            },
            "required": ["action", "payload"],
        },
    ),
]

tool_list: List[Tool] = [
    Tool(
        type="function",
        function=function
    ) for function in function_list
]

class Tools:
    _api_key: str
    _api: ToolsAPI

    def __init__(self, api_key: str):
        """
        A class to interact with the Unifai Tools API.

        :param api_key: the API key of your toolkit.
        """
        self._api_key = api_key
        self._api = ToolsAPI(api_key)
        self.set_api_endpoint(BACKEND_API_ENDPOINT)

    def set_api_endpoint(self, endpoint: str):
        self._api.set_endpoint(endpoint)

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get the list of tools in OpenAI API compatible format.

        :return: List of tools
        """
        return [tool.model_dump() for tool in tool_list]

    async def call_tool(self, name: str | FunctionName, arguments: dict | str) -> Any:
        """
        Call a tool based on the function name and arguments.

        :param name: The name of the function to call, either as a string or FunctionName enum
        :param arguments: The arguments to pass to the function, either a dictionary or a JSON encoded string
        :return: The result of the function call
        """
        name = name if isinstance(name, str) else name.value
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if name == FunctionName.SEARCH_TOOLS.value:
            return await self._api.search_tools(arguments)
        elif name == FunctionName.CALL_TOOL.value:
            return await self._api.call_tool(arguments)
        else:
            logger.warning(f"Unknown tool name: {name}")
            return None

    async def _sem_call_tool(self, name: str, arguments: dict | str, tool_call_id: str, semaphore: asyncio.Semaphore) -> Optional[OpenAIToolResult]:
        async with semaphore:
            try:
                result = await self.call_tool(name, arguments)
            except Exception as e:
                result = {"error": str(e)}
            if result is None:
                return None
            return OpenAIToolResult(
                role="tool",
                tool_call_id=tool_call_id,
                content=json.dumps(result),
            )

    async def call_tools(self, tool_calls: Optional[List[OpenAIToolCall]], concurrency: int = 1) -> List[OpenAIToolResult]:
        """
        Call multiple tools based on OpenAI tool call input/output format.

        :param tool_calls: List of OpenAI tool call objects containing function name and arguments
        :param concurrency: The maximum number of concurrent tool calls
        :return: List of results from each tool call
        """
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._sem_call_tool(
                tool_call.function.name,
                tool_call.function.arguments,
                tool_call.id,
                semaphore
            ) for tool_call in tool_calls or []
        ]
        results = await asyncio.gather(*tasks)
        return [result.model_dump() for result in results if result is not None]
