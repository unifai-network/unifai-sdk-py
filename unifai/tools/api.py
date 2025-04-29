from typing import Any, List, Dict, Optional
from ..common.api import API

class ToolsAPI(API):
    async def search_tools(
        self,
        arguments: dict,
        toolkit_ids: Optional[List[str]] = None,
        action_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Any:
        params = arguments.copy()

        if toolkit_ids:
            params['includeToolkits'] = toolkit_ids

        if action_ids:
            params['includeActions'] = action_ids

        if limit is not None and 'limit' not in params:
            params['limit'] = limit

        return await self.request('GET', '/actions/search', params=params)

    async def call_tool(self, arguments: dict) -> Any:
        return await self.request('POST', '/actions/call', json=arguments, timeout=50)
