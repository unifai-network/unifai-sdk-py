import httpx
from typing import Dict, Any

class API:
    api_key: str
    api_uri: str
    client: httpx.AsyncClient

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    def set_endpoint(self, endpoint: str):
        self.api_uri = endpoint

    async def request(
        self, 
        method: str,
        path: str, 
        timeout: float = 10.0,
        headers: Dict[str, Any] = {},
        **kwargs,
    ) -> Dict[str, Any]:
        if 'Authorization' not in headers:
            headers['Authorization'] = self.api_key
        response = await self.client.request(
            method,
            f"{self.api_uri}{path}", 
            headers=headers, 
            timeout=timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
