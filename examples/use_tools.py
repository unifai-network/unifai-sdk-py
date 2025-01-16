import asyncio
import litellm
import unifai
import sys

tools = unifai.Tools(api_key="YOUR_AGENT_API_KEY")

system_prompt = """
You are a personal assistant capable of doing many things with your tools.
When you are given a task you cannot do (like something you don't know,
or requires you to take some action), try find appropriate tools to do it.
"""

async def run(msg: str):
    messages = [
        {"content": system_prompt, "role": "system"},
        {"content": msg, "role": "user"},
    ]
    while True:
        response = await litellm.acompletion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools.get_tools(),
        )

        if response.choices[0].message.content:
            print(response.choices[0].message.content)

        messages.append(response.choices[0].message)

        if not response.choices[0].message.tool_calls:
            break

        print(
            "Calling tools: ",
            [
                f"{tool_call.function.name}({tool_call.function.arguments})"
                for tool_call in response.choices[0].message.tool_calls
            ]
        )

        results = await tools.call_tools(response.choices[0].message.tool_calls)

        if len(results) == 0:
            break

        messages.extend(results)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a message")
        sys.exit(1)
    msg = " ".join(sys.argv[1:])
    asyncio.run(run(msg))
