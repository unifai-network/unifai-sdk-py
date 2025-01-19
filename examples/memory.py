import asyncio
import uuid
import litellm
import unifai
from datetime import datetime
from typing import List, Optional
from unifai.memory import (
    Memory, 
    ChromaConfig,
    ChromaMemoryManager,
    EmptyContentError,
    MemoryError,
    StorageType,
    MemoryRole,
    ToolInfo
)
import os
from dotenv import load_dotenv

load_dotenv()

tools = unifai.Tools(api_key=os.getenv("AGENT_API_KEY"))

system_prompt = """
You are a personal assistant with memory capabilities and access to various tools.
You can store and recall information from your memory system, and use tools to perform actions.
When you need to take actions or find information you don't know, try to use appropriate tools.
Use your memory to maintain context and provide more consistent responses.
"""

async def test_memory_features():
    persistent_config = ChromaConfig(
        storage_type=StorageType.PERSISTENT,
        persist_directory="./chroma_db",
        collection_name="assistant-memories"
    )
    
    memory_manager = ChromaMemoryManager(persistent_config)
    
    await memory_manager.remove_all_memories()
    
    async def process_message(user_message: str, previous_memories: List[Memory] = None):
        messages = [{"content": system_prompt, "role": "system"}]
        
        if previous_memories:
            memory_context = "Previous relevant information:\n"
            for mem in previous_memories:
                memory_context += f"- {mem.content['text']}\n"
            messages.append({"content": memory_context, "role": "system"})
        
        messages.append({"content": user_message, "role": "user"})
        
        interaction_content = []
        tool_infos_collection = []
        
        while True:
            response = await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=messages,
                tools=tools.get_tools(),
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.content:
                print(f"Assistant: {assistant_message.content}")
                interaction_content.append(f"Assistant: {assistant_message.content}")
            
            messages.append(assistant_message)
            
            if not assistant_message.tool_calls:
                break
            
            tool_infos = [
                ToolInfo(
                    name=tool_call.function.name,
                    description=tool_call.function.arguments
                )
                for tool_call in assistant_message.tool_calls
            ]
            tool_infos_collection.extend(tool_infos)
            
            print("Calling tools:", [
                f"{tool.name}({tool.description})"
                for tool in tool_infos
            ])
            
            results = await tools.call_tools(assistant_message.tool_calls)
            
            if len(results) == 0:
                break
            
            for result in results:
                interaction_content.append(f"Tool result: {result['content']}")
            
            messages.extend(results)
        

        memory = Memory(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            content={"text": "\n".join(interaction_content)},
            tools=tool_infos_collection,
            role=MemoryRole.SYSTEM,
            unique=True
        )
        await memory_manager.create_memory(memory)
        print(f"Stored interaction memory with ID: {memory.id}")

    print("\nTesting memory system with UnifAI tools...")
    
    await process_message("What's trending on Google today?")
    

    recent_memories = await memory_manager.get_memories(count=5)
    print(f"\nFound {len(recent_memories)} recent memories")
    
async def main():
    print("Starting memory system test with UnifAI tools...")
    await test_memory_features()
    print("Memory system test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())