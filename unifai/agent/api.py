import httpx

class API:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_uri = ""
        self.client = httpx.AsyncClient()

    def set_endpoint(self, endpoint):
        self.api_uri = endpoint

    async def post_story(self, text, image_url=""):
        response = await self.client.post(f"{self.api_uri}/stories", json={
            'apiKey': self.api_key,
            'story': {
                'text': text,
                'imageURL': image_url,
            }
        })
        return response.json()
