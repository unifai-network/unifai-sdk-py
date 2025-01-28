import asyncio
import uuid
import litellm
import unifai
from typing import List
from unifai.memory import (
    Memory, 
    ChromaConfig,
    ChromaMemoryManager,
    StorageType,
    MemoryRole,
    ToolInfo
)
from unifai.memory.tool_plugin import ToolSimilarityPlugin
import os
from dotenv import load_dotenv

load_dotenv()

tools = unifai.Tools(api_key=os.getenv("AGENT_API_KEY"))
persistent_config = ChromaConfig(
    storage_type=StorageType.PERSISTENT,
    persist_directory="./chroma_db",
    collection_name="assistant-memories"
)
memory_manager = ChromaMemoryManager(persistent_config)

tool_plugin = ToolSimilarityPlugin(weight=0.5)
memory_manager.add_plugin(tool_plugin)

system_prompt = """
You are a personal assistant with memory capabilities and access to various tools.
You can store and recall information from your memory system, and use tools to perform actions.
When you need to take actions or find information you don't know, try to use appropriate tools.
Use your memory to maintain context and provide more consistent responses.
"""

async def test_memory_features():
    await memory_manager.remove_all_memories()
    
    async def process_message(user_message: str, previous_memories: List[Memory] = None):
        messages = [{"content": system_prompt, "role": "system"}]
        
        if previous_memories:
            memory_context = "Previous relevant information:\n"
            for mem in previous_memories:
                memory_context += f"- {mem.text}\n"
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
            tools=tool_infos_collection if tool_infos_collection else None,
            role=MemoryRole.SYSTEM,
            unique=True
        )
        await memory_manager.create_memory(memory)

    await process_message("What's trending on Google today?")
    
    recent_memories = await memory_manager.get_memories(
        content="What's trending on Google today?",
        count=5
    )
    print(f"\nFound {len(recent_memories)} recent memories")

if __name__ == "__main__":
    asyncio.run(test_memory_features())