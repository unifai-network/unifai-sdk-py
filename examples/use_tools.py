import dotenv
dotenv.load_dotenv()

import asyncio
import os
import sys
import litellm
import unifai
from typing import List

async def run(msg: str, static_toolkits: List[str] | None = None, static_actions: List[str] | None = None):
    agent_api_key = os.getenv("UNIFAI_AGENT_API_KEY", "")
    tools = unifai.Tools(api_key=agent_api_key)
  
    available_tools = await tools.get_tools(
        dynamic_tools=True,
        static_toolkits=static_toolkits,
        static_actions=static_actions,
    )
    messages: List = [
        {"content": unifai.Agent("").get_prompt("agent.system"), "role": "system"},
        {"content": msg, "role": "user"},
    ]
    
    while True:
        response = await litellm.acompletion(
            model="anthropic/claude-3-7-sonnet-20250219",
            messages=messages,
            tools=available_tools,
        )

        message = response.choices[0].message # type: ignore

        if message.content:
            print(message.content)

        messages.append(message)

        if not message.tool_calls:
            break

        print(
            "Calling tools: ",
            [
                f"{tool_call.function.name}({tool_call.function.arguments})"
                for tool_call in message.tool_calls
            ]
        )

        results = await tools.call_tools(message.tool_calls) # type: ignore
        if len(results) == 0:
            break

        messages.extend(results)

if __name__ == "__main__":
    static_toolkits = None
    static_actions = None
    message_parts = []

    for arg in sys.argv[1:]:
        if arg.startswith("--toolkit="):
            static_toolkits = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--action="):
            static_actions = arg.split("=", 1)[1].split(",")
        else:
            message_parts.append(arg)

    msg = " ".join(message_parts) if message_parts else "What can you help me with?"

    if not message_parts and not static_toolkits and not static_actions and len(sys.argv) > 1:
        print("Usage: python use_tools.py [--toolkit=ID] [--action=ACTION] [your message here]")
        sys.exit(1)

    asyncio.run(run(msg, static_toolkits, static_actions))
