from typing import Any, List
from ..common.api import API

class ToolsAPI(API):
    async def search_tools(self, arguments: dict) -> Any:
        return await self.request('GET', '/actions/search', params=arguments)

    async def call_tool(self, arguments: dict) -> Any:
        return await self.request('POST', '/actions/call', json=arguments, timeout=50)
    
    async def get_toolkit_actions(self, toolkit_id: str) -> Any:
        return await self.request('GET', f'/toolkits/{toolkit_id}/actions')
    
    async def get_action(self, action_id: str) -> Any:
        return await self.request('GET', f'/actions/{action_id}')
