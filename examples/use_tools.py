import asyncio
import os
import sys
import litellm
import unifai
from typing import List

import dotenv
dotenv.load_dotenv()

tools = unifai.Tools(api_key=os.getenv("UNIFAI_AGENT_API_KEY", ""))

async def run(msg: str):
    messages: List = [
        {"content": unifai.Agent("").get_prompt("agent.system"), "role": "system"},
        {"content": msg, "role": "user"},
    ]
    while True:
        response = await litellm.acompletion(
            model="anthropic/claude-3-7-sonnet-20250219",
            messages=messages,
            tools=tools.get_tools(),
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
    if len(sys.argv) < 2:
        print("Please provide a message")
        sys.exit(1)
    msg = " ".join(sys.argv[1:])
    asyncio.run(run(msg))
