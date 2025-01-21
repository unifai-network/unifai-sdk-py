import asyncio
import aiohttp
import json
import os
import importlib.metadata

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.websocket
import mcp.types as types
import mcp.server.stdio

from ..tools import Tools, function_list

API_KEY = os.getenv("UNIFAI_AGENT_API_KEY")
RAISE_EXCEPTIONS = os.getenv("RAISE_EXCEPTIONS", "true").lower() in ("true", "1")

SERVER_NAME = "unifai-tools"

try:
    package_name = __name__.split('.')[0]
    SERVER_VERSION = importlib.metadata.version(package_name)
except:
    SERVER_VERSION = ""

server = Server(SERVER_NAME, SERVER_VERSION)

tools = Tools(API_KEY)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name=function.name,
            description=function.description,
            inputSchema=function.parameters,
        )
        for function in function_list
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict = {}
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    result = await tools.call_tool(name, arguments)
    return [
        types.TextContent(
            type="text",
            text=json.dumps(result)
        )
    ]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
            raise_exceptions=RAISE_EXCEPTIONS,
        )

if __name__ == "__main__":
    asyncio.run(main())
