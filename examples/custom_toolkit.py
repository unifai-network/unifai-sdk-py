import asyncio
import os
import unifai
import litellm
from dotenv import load_dotenv

load_dotenv()

async def use_toolkit_by_id(toolkit_id):
    agent_api_key = os.getenv("UNIFAI_AGENT_API_KEY", os.getenv("UNIFAI_TOOLKIT_API_KEY"))
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
            {"role": "system", "content": "You are an assistant that helps users with echoing messages."},
            {"role": "user", "content": "Echo back 'Hello, world!'"}
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
        
        if message.tool_calls and len(message.tool_calls) > 0:
            print(f"\nFound {len(message.tool_calls)} tool calls to execute...")
            
            for i, call in enumerate(message.tool_calls):
                print(f"Tool call {i+1}:")
                print(f"  Name: {call.function.name}")
                print(f"  Arguments: {call.function.arguments}")
            
            try:
                print("Executing tool calls...")
                results = await tools.call_tools(message.tool_calls)
                
                print(f"Got {len(results)} results from tool execution")
                
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                for result in results:
                    messages.append(result)
                
                print("\nTool execution results:")
                for result in results:
                    print(f"- Role: {result.get('role', 'unknown')}")
                    print(f"  Content: {result.get('content', 'No content')}")
                
                print("\nGetting final response with tool results...")
                final_response = await litellm.acompletion(
                    model="anthropic/claude-3-7-sonnet-20250219",
                    messages=messages
                )
                
                print("\nFinal result:")
                print(final_response.choices[0].message.content)
            except Exception as e:
                print(f"Error during tool execution: {e}")
                print("Tool execution failed. Check if the tools are properly configured.")
        else:
            print("\nNo tool calls were detected in the LLM response.")
    except Exception as e:
        print(f"Error during toolkit use: {e}")
        print("Note: Check if the toolkit ID is valid and you have proper permissions.")

async def main():
    toolkit_id = '10'  
    await use_toolkit_by_id(toolkit_id)
    print("Task completed")

if __name__ == "__main__":
    if not os.getenv("UNIFAI_TOOLKIT_API_KEY"):
        api_key = input("Enter your Unifai Toolkit API key: ")
        os.environ["UNIFAI_TOOLKIT_API_KEY"] = api_key
    
    asyncio.run(main())