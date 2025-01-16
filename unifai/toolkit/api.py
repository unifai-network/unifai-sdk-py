from typing import Optional
from ..common.api import API

class ToolkitAPI(API):
    async def update_toolkit(self, name: Optional[str] = None, description: Optional[str] = None):
        data = {k: v for k, v in {'name': name, 'description': description}.items() if v is not None}
        await self.request('POST', '/toolkits/fields/', json=data)
