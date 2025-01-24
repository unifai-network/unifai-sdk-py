import asyncio
from uuid import uuid4
from unifai.memory.base import Memory, ChromaConfig, StorageType, MemoryType, MemoryRole
from unifai.memory.chroma import ChromaMemoryManager

async def test_chroma_memory_manager():
    # Initialize configuration
    config = ChromaConfig(
        storage_type=StorageType.PERSISTENT,
        persist_directory="./test_db",
        collection_name="test_collection",
        dimensions=384,  # Default dimension for sentence-transformers
        distance_metric="cosine"
    )
    
    # Create memory manager
    manager = ChromaMemoryManager(config)
    
    # Clean up any existing memories
    await manager.remove_all_memories()
    
    # Generate test IDs
    test_user_id = uuid4()
    test_agent_id = uuid4()
    
    # Test creating memories with different types
    test_memories = []
    memory_types = [MemoryType.INTERACTION, MemoryType.FACT, MemoryType.GOAL]
    
    for i, memory_type in enumerate(memory_types):
        memory = Memory(
            id=uuid4(),
            user_id=test_user_id,
            agent_id=test_agent_id,
            content={"text": f"Test {memory_type.value} memory content {i}"},
            memory_type=memory_type,
            metadata={
                "test_key": f"test_value_{i}",
                "test_number": i,
                "test_bool": True
            },
            role=MemoryRole.SYSTEM,
            tools=[],
            unique=True
        )
        await manager.create_memory(memory)
        test_memories.append(memory)
        print(f"Created {memory_type.value} memory with ID: {memory.id}")
    
    # Test retrieving memories by type
    for memory_type in memory_types:
        type_memories = await manager.get_memories_by_type(
            memory_type=memory_type,
            count=5,
        )
        print(f"Found {len(type_memories)} memories of type {memory_type.value}")
        for mem in type_memories:
            print(f"- Memory content: {mem.content['text']}")
    
    # Test retrieving memories by content similarity
    similar_memories = await manager.get_memories(
        content="Test memory content",
        count=2,
        threshold=0.0
    )
    print(f"Found {len(similar_memories)} similar memories")
    
    # Test retrieving specific memory
    retrieved_memory = await manager.get_memory_by_id(test_memories[0].id)
    if retrieved_memory:
        print(f"Retrieved memory: {retrieved_memory.content['text']}")
    
    # Test updating memory
    test_memories[0].content["text"] = "Updated memory content"
    await manager.update_memory(test_memories[0])
    print("Updated memory successfully")
    
    # Test removing a memory
    await manager.remove_memory(test_memories[0].id)
    print(f"Removed memory: {test_memories[0].id}")
    
    # Clean up
    await manager.remove_all_memories()
    print("Cleaned up all memories")

if __name__ == "__main__":
    asyncio.run(test_chroma_memory_manager())