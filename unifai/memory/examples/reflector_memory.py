import asyncio
import uuid
import litellm
import unifai
from datetime import datetime
from typing import List, Optional, Dict, Any
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

api_key = os.getenv("AGENT_API_KEY")
if api_key is None:
    raise ValueError("AGENT_API_KEY environment variable is not set")
    
tools = unifai.Tools(api_key=api_key) 

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

retrieved_plugin = memory_manager.get_plugin("ToolSimilarityPlugin")
if retrieved_plugin is not None and isinstance(retrieved_plugin, ToolSimilarityPlugin):
    retrieved_plugin.weight = 0.5

async def process_interaction(user_message: str, previous_memories: Optional[List[Memory]] = None):
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
            tools=await tools.get_tools(),
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
    if fact_result.success and fact_result.data is not None:
        facts = fact_result.data.get('claims', [])
        if facts:
            fact_memory = Memory(
                id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                agent_id=uuid.uuid4(),
                content={
                    "text": "Extracted facts from conversation",
                    "claims": facts
                },
                memory_type=MemoryType.FACT,
                metadata={
                    "source_interaction": full_interaction
                },
                role=MemoryRole.SYSTEM,
                tools=tool_infos_collection if tool_infos_collection else None,
                unique=True
            )
            await memory_manager.create_memory(fact_memory)
            print(f"Stored fact memory with ID: {fact_memory.id}")

    if goal_result.success and goal_result.data is not None:
        goals = goal_result.data.get('goals', [])
        if goals:
            goal_memory = Memory(
                id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                agent_id=uuid.uuid4(),
                content={
                    "text": "Goals and progress tracking",
                    "goals": goals
                },
                memory_type=MemoryType.GOAL,
                metadata={
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

async def consolidate_fact_memories(memory_manager, similarity_threshold=0.85):
    """Consolidate similar fact memories to reduce redundancy."""

    all_facts = await memory_manager.get_memories_by_type(
        memory_type=MemoryType.FACT,
        count=100  
    )
    
    consolidated = []
    to_remove = set()
    
    for i, fact1 in enumerate(all_facts):
        if i in to_remove:
            continue
            
        similar_facts = [fact1]
        
        for j, fact2 in enumerate(all_facts[i+1:], i+1):
            if j in to_remove:
                continue
                
            similarity = calculate_similarity(fact1.content["text"], fact2.content["text"])
            if similarity > similarity_threshold:
                similar_facts.append(fact2)
                to_remove.add(j)
        
        if len(similar_facts) > 1:
            consolidated_memory = await merge_memories(similar_facts)
            consolidated.append(consolidated_memory)
            

            for fact in similar_facts:
                await memory_manager.remove_memory(fact.id)
            
            await memory_manager.create_memory(consolidated_memory)
    
    return len(consolidated)

if __name__ == "__main__":
    asyncio.run(main()) 