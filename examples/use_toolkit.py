import asyncio
import os
import json
import unifai
import litellm
from dotenv import load_dotenv

load_dotenv()

async def use_toolkit_by_id(toolkit_id='10'):
    agent_api_key = os.getenv("AGENT_API_KEY")
    tools = unifai.Tools(api_key=agent_api_key)
    
    print(f"Using toolkit with ID: {toolkit_id}")
    try:
        toolkit_tools = await tools.get_tools_async(
            dynamicTools=True,
            staticToolkits=[toolkit_id]
        )
        
        if not toolkit_tools:
            print(f"No tools found for toolkit ID '{toolkit_id}'.")
            print("Note: Unifai may only support toolkit IDs, not names. Check the documentation.")
            return
        
        print(f"Successfully loaded {len(toolkit_tools)} tools from the toolkit")
    
        messages = [
            {"role": "system", "content": "You are an assistant that helps users with web search."},
            {"role": "user", "content": "What is the latest news todday?'"}
        ]
        
        litellm.modify_params = True
        
        print("Calling LLM with toolkit tools...")
        response = await litellm.acompletion(
            model="anthropic/claude-3-7-sonnet-20250219",
            messages=messages,
            tools=toolkit_tools
        )
        
        message = response.choices[0].message
        print("\nLLM initial response:")
        if message.content:
            print(message.content)
        else:
            print("(No content, LLM is using tools)")
        if message.tool_calls:
            print(f"\nExecuting {len(message.tool_calls)} tool calls...")
            for call in message.tool_calls:
                print(f"- Tool: {call.function.name}")
                print(f"  Args: {call.function.arguments}")
            
            results = await tools.call_tools(message.tool_calls)
            
            messages.append(message)
            messages.extend(results)
            
            print("\nTool execution results:")
            for result in results:
                print(f"- {result.get('content', 'No content')}")
            
            print("\nGetting final response...")
            final_response = await litellm.acompletion(
                model="anthropic/claude-3-7-sonnet-20250219",
                messages=messages
            )
            
            print("\nFinal result:")
            print(final_response.choices[0].message.content)
        else:
            print("\nNo tool calls were made.")
    except Exception as e:
        print(f"Error during toolkit use: {e}")
        print("Note: Unifai may only support toolkit IDs, not names. You might need to use toolkit IDs instead.")

async def main():
    await use_toolkit_by_id(10)
    print("Task completed")
    print("Exiting program")


if __name__ == "__main__":
    if not os.getenv("UNIFAI_TOOLKIT_API_KEY"):
        api_key = input("Enter your Unifai Toolkit API key: ")
        os.environ["UNIFAI_TOOLKIT_API_KEY"] = api_key
    
    asyncio.run(main())