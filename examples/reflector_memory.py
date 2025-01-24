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
    StorageType,
    MemoryRole,
    ToolInfo,
    MemoryType
)
from unifai.reflector import FactReflector, GoalReflector
import os
from dotenv import load_dotenv
from unifai.memory.tool_plugin import ToolSimilarityPlugin

load_dotenv()

tools = unifai.Tools(api_key=os.getenv("AGENT_API_KEY"))
persistent_config = ChromaConfig(
    storage_type=StorageType.PERSISTENT,
    persist_directory="./chroma_db",
    collection_name="assistant-memories"
)
memory_manager = ChromaMemoryManager(persistent_config)

fact_reflector = FactReflector(litellm)
goal_reflector = GoalReflector(litellm)

system_prompt = """
You are a personal assistant with memory capabilities and access to various tools.
You can store and recall information from your memory system, and use tools to perform actions.
When you need to take actions or find information you don't know, try to use appropriate tools.
Use your memory to maintain context and provide more consistent responses.
"""

tool_plugin = ToolSimilarityPlugin(weight=0.5)
memory_manager.add_plugin(tool_plugin)

print("Active plugins:", memory_manager.list_plugins())

tool_plugin = memory_manager.get_plugin("ToolSimilarityPlugin")
if tool_plugin:
    tool_plugin.weight = 0.5

async def process_interaction(user_message: str, previous_memories: List[Memory] = None):
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
        
        results = await tools.call_tools(assistant_message.tool_calls)
        
        if not results:
            break
        
        for result in results:
            interaction_content.append(f"Tool result: {result['content']}")
        
        messages.extend(results)

    full_interaction = "\n".join(interaction_content)

    fact_result = await fact_reflector.reflect(full_interaction)
    goal_result = await goal_reflector.reflect(full_interaction)
    if fact_result.success and fact_result.data.get('claims'):
        fact_memory = Memory(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            content={
                "text": "Extracted facts from conversation"
            },
            memory_type=MemoryType.FACT,
            metadata={
                "claims": fact_result.data['claims'],
                "source_interaction": full_interaction
            },
            role=MemoryRole.SYSTEM,
            tools=tool_infos_collection if tool_infos_collection else None,
            unique=True
        )
        await memory_manager.create_memory(fact_memory)
        print(f"Stored fact memory with ID: {fact_memory.id}")

    if goal_result.success and goal_result.data.get('goals'):
        goal_memory = Memory(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            content={
                "text": "Goals and progress tracking"
            },
            memory_type=MemoryType.GOAL,
            metadata={
                "goals": goal_result.data['goals'],
                "source_interaction": full_interaction
            },
            role=MemoryRole.SYSTEM,
            tools=tool_infos_collection if tool_infos_collection else None,
            unique=True
        )
        await memory_manager.create_memory(goal_memory)
        print(f"Stored goal memory with ID: {goal_memory.id}")

    interaction_memory = Memory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        content={
            "text": full_interaction
        },
        role=MemoryRole.SYSTEM,
        tools=tool_infos_collection if tool_infos_collection else None,
        unique=False
    )
    await memory_manager.create_memory(interaction_memory)
    print(f"Stored interaction memory with ID: {interaction_memory.id}")
    
async def test_reflector_memory():
    print("\nTesting reflector memory system...")
    
    await memory_manager.remove_all_memories()
    
    test_message = """
    I need to plan a trip to Japan next month. I want to visit Tokyo and Kyoto.
    My budget is $5000 and I'll be staying for 10 days.
    Can you help me research some options?
    """
    
    await process_interaction(test_message)
    
    recent_memories = await memory_manager.get_memories(
        content=test_message,
        count=5
    )
    print("\nRecent memories:")
    for memory in recent_memories:
        print(f"\nMemory ID: {memory.id}")
        print(f"Content: {memory.content['text']}")

async def main():
    print("Starting reflector memory system test...")
    await test_reflector_memory()
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 