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
    SEARCH_TOOLS = "search_services"
    CALL_TOOL = "invoke_service"

function_list: List[Function] = [
    Function(
        name=FunctionName.SEARCH_TOOLS.value,
        description=(
            "Search for services. The services cover a wide range of domains include data source, API, SDK, etc. "
            "Try searching whenever you need to use a service. "
            f"Returned actions should ONLY be used in {FunctionName.CALL_TOOL.value}. "
            "Note that this function is not web search. If you need to search the web, try searching for web search services."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for services, you can describe what you want to do or what services you want to use"
                },
                "limit": {
                    "type": "number",
                    "description": "The maximum number of services to return, must be between 1 and 100, default is 10, recommend at least 10"
                }
            },
            "required": ["query"],
        },
    ),
    Function(
        name=FunctionName.CALL_TOOL.value,
        description=f"Call a service returned by {FunctionName.SEARCH_TOOLS.value}",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": f"The exact action you want to call in the {FunctionName.SEARCH_TOOLS.value} result."
                },
                "payload": {
                    "type": "string",
                    "description": (
                        f"Action payload, based on the payload schema in the {FunctionName.SEARCH_TOOLS.value} result. "
                        "You can pass either the json object directly or json encoded string of the object."
                    ),
                },
                "payment": {
                    "type": "number",
                    "description": (
                        "Amount to authorize in USD. "
                        "Positive number means you will be charged no more than this amount, "
                        "negative number means you are requesting to get paid for at least this amount. "
                        "Only include this field if the action you are calling includes payment information."
                    )
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

    async def _fetch_static_tools(self, staticToolkits: List[str] = None, staticActions: List[str] = None) -> List[Dict[str, Any]]:
        result_tools = []
        
        if staticToolkits:
            toolkit_tasks = [self._api.get_toolkit_actions(toolkit_id) for toolkit_id in staticToolkits]
            toolkit_results = await asyncio.gather(*toolkit_tasks, return_exceptions=True)
            
            for i, result in enumerate(toolkit_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch toolkit {staticToolkits[i]}: {result}")
                else:
                    for action in result.get("actions", []):
                        result_tools.append({
                            "type": "function",
                            "function": {
                                "name": action.get("id", "unknown"),
                                "description": action.get("description", ""),
                                "parameters": action.get("parameters", {})
                            }
                        })
        
        if staticActions:
            action_tasks = [self._api.get_action(action_id) for action_id in staticActions]
            action_results = await asyncio.gather(*action_tasks, return_exceptions=True)
            
            for i, result in enumerate(action_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch action {staticActions[i]}: {result}")
                else:
                    result_tools.append({
                        "type": "function",
                        "function": {
                            "name": result.get("id", "unknown"),
                            "description": result.get("description", ""),
                            "parameters": result.get("parameters", {})
                        }
                    })
        
        return result_tools

    def get_tools(self, dynamicTools: bool = True, staticToolkits: List[str] = None, staticActions: List[str] = None, cache_control: bool = False) -> List[Dict[str, Any]]:
        tools = []
        
        if dynamicTools:
            tools.extend([tool.model_dump(mode="json") for tool in tool_list])
        
        if staticToolkits or staticActions:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                static_tools = loop.run_until_complete(self._fetch_static_tools(staticToolkits, staticActions))
                loop.close()
                
                tools.extend(static_tools)
            except Exception as e:
                logger.error(f"Error fetching static tools: {e}")
        
        if cache_control and tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        
        return tools

    async def get_tools_async(self, dynamicTools: bool = True, staticToolkits: List[str] = None, staticActions: List[str] = None, cache_control: bool = False) -> List[Dict[str, Any]]:
        tools = []
        
        if dynamicTools:
            tools.extend([tool.model_dump(mode="json") for tool in tool_list])
        
        if staticToolkits or staticActions:
            static_tools = await self._fetch_static_tools(staticToolkits, staticActions)
            tools.extend(static_tools)
        
        if cache_control and tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}
        
        return tools

    async def call_tool(self, name: str | FunctionName, arguments: dict | str) -> Any:
        name = name if isinstance(name, str) else name.value
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        
        if name == FunctionName.SEARCH_TOOLS.value:
            return await self._api.search_tools(args)
        elif name == FunctionName.CALL_TOOL.value:
            return await self._api.call_tool(args)
        else:
            call_args = {
                "action": name,
                "payload": args
            }
            try:
                return await self._api.call_tool(call_args)
            except Exception as e:
                raise ValueError(f"Failed to call tool {name}: {e}")

    async def call_all_tools(self, name: str, arguments: dict | str) -> Any:
        """
        Call a tool by name, whether it's a dynamic tool or a static one.
        
        :param name: The name of the function/action to call
        :param arguments: The arguments to pass to the function
        :return: The result of the function call
        """
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        
        # First check if it's one of our built-in functions
        if name == FunctionName.SEARCH_TOOLS.value:
            return await self._api.search_tools(args)
        elif name == FunctionName.CALL_TOOL.value:
            return await self._api.call_tool(args)
        else:
            # It might be a static action, try to call it directly
            call_args = {
                "action": name,
                "payload": args
            }
            try:
                return await self._api.call_tool(call_args)
            except Exception as e:
                raise ValueError(f"Failed to call tool {name}: {e}")

    async def _sem_call_tool(self, name: str, arguments: dict | str, tool_call_id: str, semaphore: asyncio.Semaphore) -> Optional[OpenAIToolResult]:
        async with semaphore:
            try:
                result = await self.call_all_tools(name, arguments)
            except Exception as e:
                result = {"error": str(e)}
            if result is None:
                return None
            return OpenAIToolResult(
                role="tool",
                tool_call_id=tool_call_id,
                content=json.dumps(result),
            )

    async def call_tools(self, tool_calls: Optional[List[OpenAIToolCall]], concurrency: int = 1) -> List[dict[str, Any]]:
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
        return [result.model_dump(mode="json") for result in results if result is not None]
