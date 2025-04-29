import dotenv
dotenv.load_dotenv()

import asyncio
import os
import sys
import litellm
import unifai
from typing import List

async def run(msg: str, toolkit_id=None, action=None):
    agent_api_key = os.getenv("UNIFAI_AGENT_API_KEY", "")
    tools = unifai.Tools(api_key=agent_api_key)
  
    available_tools = await tools.get_tools_async(
        dynamic_tools=True,
        static_toolkits=[toolkit_id] if toolkit_id else [],
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

        message = response.choices[0].message 

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
        print(results)
        if len(results) == 0:
            break

        messages.extend(results)
        

if __name__ == "__main__":
    toolkit_id = None
    action = None
    message_parts = []

    for arg in sys.argv[1:]:
        if arg.startswith("--toolkit="):
            toolkit_id = arg.split("=", 1)[1]
        elif arg.startswith("--action="):
            action = arg.split("=", 1)[1]
        else:
            message_parts.append(arg)

    msg = " ".join(message_parts) if message_parts else "What can you help me with?"

    if not message_parts and not toolkit_id and not action and len(sys.argv) > 1:
        print("Usage: python use_tools.py [--toolkit=ID] [--action=ACTION] [your message here]")
        sys.exit(1)

    asyncio.run(run(msg, toolkit_id, action))