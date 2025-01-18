import asyncio
from typing import List, Optional, Dict, Any, Union, Sequence, cast, TypeVar, Mapping
from uuid import UUID
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection
from chromadb.types import Where, WhereDocument
import numpy as np
from numpy.typing import NDArray

from .base import Memory, ChromaConfig, StorageType
from .exceptions import EmptyContentError, CollectionError, ConnectionError, MemoryError
from .protocols import MemoryManager
from .utils import serialize_memory, deserialize_memory

# Type definitions
T = TypeVar('T')
EmbeddingType = Union[NDArray[np.float32], List[float], Sequence[float]]
WhereType = Dict[str, Any]
MetadataType = Dict[str, Union[str, int, float, bool]]
ChromaGetResult = Dict[str, Any]
ChromaQueryResult = Dict[str, Any]

def safe_cast(obj: Any, cls: type[T]) -> T:
    return cast(cls, obj)

class ChromaMemoryManager(MemoryManager):
    def __init__(self, config: ChromaConfig):
        self.config = config
        try:
            if config.storage_type == StorageType.HTTP:
                self.client = chromadb.HttpClient(
                    host=config.host,
                    port=config.port,
                    ssl=False,
                    headers=None,
                    settings=Settings(),
                )
            else:  # PERSISTENT
                self.client = chromadb.PersistentClient(
                    path=config.persist_directory,
                    settings=Settings(
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                
            # Add default embedding function if none provided
            self.embedding_function = self._get_embedding_function()
            self.collection = self._initialize_collection()
        except Exception as e:
            if config.storage_type == StorageType.HTTP:
                raise ConnectionError(
                    host=self.config.host,
                    port=self.config.port,
                    details=str(e)
                )
            else:
                raise ConnectionError(
                    host=f"persistent:{self.config.persist_directory}",
                    port=0,
                    details=str(e)
                )

    def _get_embedding_function(self):
        """Get embedding function based on config or use default"""
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            raise MemoryError(f"Failed to initialize embedding function: {str(e)}")

    def _initialize_collection(self) -> Collection:
        try:
            return self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "dimension": self.config.dimensions,
                    "distance_metric": self.config.distance_metric
                },
                embedding_function=self.embedding_function
            )
        except Exception as e:
            raise CollectionError(
                operation="initialize_collection",
                details=str(e)
            )

    async def add_embedding_to_memory(self, memory: Memory) -> Memory:
        if memory.embedding:
            return memory

        if not memory.content.get("text"):
            raise EmptyContentError()

        try:
            # Generate embedding using the embedding function
            embedding = self.embedding_function([memory.content["text"]])[0]
            memory.embedding = embedding
            
            # Store the memory with its embedding
            metadata = serialize_memory(memory)
            await asyncio.to_thread(
                self.collection.add,
                ids=[str(memory.id)],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[memory.content["text"]]
            )
            
            return memory
        except Exception as e:
            raise MemoryError(f"Failed to add embedding: {str(e)}")

    async def create_memory(self, memory: Memory) -> None:
        try:
            metadata = serialize_memory(memory)

            await asyncio.to_thread(
                self.collection.add,
                ids=[str(memory.id)],
                embeddings=[memory.embedding] if memory.embedding else None,
                metadatas=[metadata],
                documents=[memory.content["text"]]
            )
        except Exception as e:
            raise MemoryError(f"Failed to create memory: {str(e)}")

    async def search_memories_by_embedding(
        self,
        embedding: List[float],
        match_threshold: float = 0.8,
        count: int = 10,
        unique: bool = False
    ) -> List[Memory]:
        try:
            where = {"unique": True} if unique else None

            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[embedding],
                n_results=count,
                where=where,
                include=["metadatas", "embeddings", "distances"]
            )

            if not results["ids"][0]:
                return []

            return [
                deserialize_memory(
                    memory_id=results["ids"][0][idx],
                    metadata=metadata,
                    embedding=results["embeddings"][0][idx] if results.get("embeddings") else None,
                    similarity=1 - results["distances"][0][idx]
                )
                for idx, metadata in enumerate(results["metadatas"][0])
            ]
        except Exception as e:
            raise MemoryError(f"Failed to search memories: {str(e)}")

    async def get_memories(
        self,
        count: Optional[int] = None,
        unique: bool = False,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Memory]:
        try:
            where = {"unique": True} if unique else None
            
            results = await asyncio.to_thread(
                self.collection.get,
                where=where,
                limit=count,
                offset=start,
                include=["metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                return []
            
            memories = []
            for idx, memory_id in enumerate(results["ids"]):
                embedding = None
                if "embeddings" in results and isinstance(results["embeddings"], np.ndarray):
                    embedding = results["embeddings"][idx].tolist()
                elif "embeddings" in results and isinstance(results["embeddings"], list):
                    embedding = results["embeddings"][idx]
                    
                memory = deserialize_memory(
                    memory_id=memory_id,
                    metadata=results["metadatas"][idx],
                    embedding=embedding
                )
                memories.append(memory)
                
            return memories
        except Exception as e:
            raise MemoryError(f"Failed to get memories: {str(e)}")

    async def get_memory_by_id(self, memory_id: UUID) -> Optional[Memory]:
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                ids=[str(memory_id)],
                include=["metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                return None
            

            embedding = None
            if "embeddings" in results and isinstance(results["embeddings"], list) and results["embeddings"]:
                embedding_data = results["embeddings"][0]
                if hasattr(embedding_data, 'tolist'):
                    embedding = embedding_data.tolist()
                elif isinstance(embedding_data, (list, np.ndarray)):
                    embedding = list(embedding_data)
                
            return deserialize_memory(
                memory_id=results["ids"][0],
                metadata=results["metadatas"][0],
                embedding=embedding
            )
        except Exception as e:
            raise MemoryError(f"Failed to get memory by ID: {str(e)}")

    async def update_memory(self, memory: Memory) -> None:
        try:
            metadata = serialize_memory(memory)
            
            # Check if embedding exists using isinstance instead of truthiness
            if not isinstance(memory.embedding, (list, np.ndarray)) or len(memory.embedding) == 0:
                embedding = self.embedding_function([memory.content["text"]])[0]
                memory.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            # Ensure embedding is a list
            current_embedding = memory.embedding
            if hasattr(current_embedding, 'tolist'):
                current_embedding = current_embedding.tolist()
            elif not isinstance(current_embedding, list):
                current_embedding = list(current_embedding)
            
            await asyncio.to_thread(
                self.collection.update,
                ids=[str(memory.id)],
                embeddings=[current_embedding],
                metadatas=[metadata],
                documents=[memory.content["text"]]
            )
        except Exception as e:
            raise MemoryError(f"Failed to update memory: {str(e)}")

    async def remove_memory(self, memory_id: UUID) -> None:
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=[str(memory_id)]
            )
        except Exception as e:
            raise MemoryError(f"Failed to remove memory: {str(e)}")

    async def remove_all_memories(self) -> None:
        try:
            # First get all memory IDs
            results = await asyncio.to_thread(
                self.collection.get,
                include=["metadatas"]  # We only need metadata to get the IDs
            )
            
            if results["ids"]:
                # Then delete all memories by their IDs
                await asyncio.to_thread(
                    self.collection.delete,
                    ids=results["ids"]
                )
        except Exception as e:
            raise MemoryError(f"Failed to remove all memories: {str(e)}")