from typing import Any
from ..common.api import API

class ToolsAPI(API):
    async def search_tools(self, arguments: dict) -> Any:
        return await self.request('GET', '/actions/search', params=arguments)

    async def call_tool(self, arguments: dict) -> Any:
        return await self.request('POST', '/actions/call', json=arguments, timeout=50)
