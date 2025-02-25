import asyncio
from typing import List, Optional, Dict, Any, Union, Sequence, cast, TypeVar, Mapping
from uuid import UUID
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection
from chromadb.types import Where, WhereDocument
import numpy as np
from numpy.typing import NDArray
import json

from .base import Memory, ChromaConfig, StorageType, MemoryType
from .exceptions import EmptyContentError, CollectionError, ConnectionError, MemoryError
from .protocols import MemoryManager
from .plugin import MemoryRankPlugin, PluginContext, MemoryContext
from .utils import serialize_memory, deserialize_memory

T = TypeVar('T')
EmbeddingType = Union[NDArray[np.float32], List[float], Sequence[float]]
WhereType = Dict[str, Any]
MetadataType = Dict[str, Union[str, int, float, bool]]
ChromaGetResult = Dict[str, Any]
ChromaQueryResult = Dict[str, Any]


class ChromaMemoryManager(MemoryManager):
    def __init__(self, config: ChromaConfig):
        self.config = config
        
        self.client = self._initialize_client()
        
        try:
            self.embedding_function = self._get_embedding_function()
            self.collection = self._initialize_collection()
        except Exception as e:
            raise MemoryError(f"Failed to initialize memory manager: {str(e)}")
        
        self.plugins: List[MemoryRankPlugin] = []

    def _initialize_client(self) -> Any:
        try:
            if self.config.storage_type == StorageType.PERSISTENT:
                if not self.config.persist_directory:
                    raise ValueError("persist_directory must be provided for PERSISTENT storage")
                
                return chromadb.PersistentClient(
                    path=self.config.persist_directory, 
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
            elif self.config.storage_type == StorageType.HTTP:
                return chromadb.HttpClient(
                    host=self.config.host or "localhost",
                    port=self.config.port or 8000,
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
            else:
                return chromadb.Client(
                    Settings(
                        anonymized_telemetry=False,
                        is_persistent=False
                    )
                )
        except Exception as e:
            raise ConnectionError(
            host=self.config.host,
            port=self.config.port,
               details=str(e)
        )

    def _get_embedding_function(self):
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            raise MemoryError(f"Failed to initialize embedding function: {str(e)}")

    def _initialize_collection(self) -> Any:
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
            raise ConnectionError(
            host=self.config.host,
            port=self.config.port,
               details=str(e)
        )

    def _convert_embedding_to_list(self, embedding: Any) -> List[float]:
        if embedding is None:
            return []
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        if isinstance(embedding, (list, np.ndarray)):
            return list(embedding)
        raise ValueError(f"Unsupported embedding type: {type(embedding)}")

    async def add_embedding_to_memory(self, memory: Memory) -> Memory:
        if memory.embedding:
            memory.embedding = self._convert_embedding_to_list(memory.embedding)
            return memory

        if not memory.content.get("text"):
            raise EmptyContentError()

        try:
            embedding = self.embedding_function([memory.content["text"]])[0]
            memory.embedding = self._convert_embedding_to_list(embedding)
            return memory
        except Exception as e:
            raise MemoryError(f"Failed to generate embedding: {str(e)}")

    async def create_memory(self, memory: Memory) -> None:
        try:
            if not memory.embedding:
                memory = await self.add_embedding_to_memory(memory)
            metadata = serialize_memory(memory)
            
            await asyncio.to_thread(
                self.collection.add,
                ids=[str(memory.id)],
                embeddings=[memory.embedding],
                metadatas=[metadata],
                documents=[memory.content["text"]]
            )
        except Exception as e:
            raise MemoryError(f"Failed to create memory: {str(e)}")

    async def _get_base_memories(
        self,
        content: str,
        count: int,
        threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        """Get base memories using content similarity and optional metadata filters"""
        try:
            embedding = self.embedding_function([content])[0]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            return await self._get_memories_with_filter(
                where=where,
                count=count,
                threshold=threshold,
                query_embedding=embedding if content.strip() else None
            )
        except Exception as e:
            raise MemoryError(f"Failed to get base memories: {str(e)}")

    async def get_memories(
        self,
        content: str,
        count: int = 5,
        threshold: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Memory]:
        """Get memories with content similarity and metadata filtering"""
        where = None
        if metadata:
            conditions = []
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    conditions.append({
                        key: {"$eq": str(value)}
                    })
                else:
                    conditions.append({
                        key: {"$eq": json.dumps(value)}
                    })
    
            if len(conditions) == 1:
                where = conditions[0]
    
            elif len(conditions) > 1:
                where = {"$and": conditions}
        
        base_memories = await self._get_base_memories(
            content=content, 
            count=count * 2, 
            threshold=threshold,
            where=where
        )

        if not self.plugins:
            return base_memories[:count]
        
        context = MemoryContext(
            content=content,
            count=count,
            threshold=threshold,
            _extra_args=kwargs
        )
        
        memories = base_memories
        for plugin in self.plugins:
            try:
                result = await plugin.rerank(memories, context)
                memories = result.memories
            except Exception as e:
                print(f"Plugin {plugin.name} failed: {str(e)}")
                continue
        return memories[:count]

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
            
            if not isinstance(memory.embedding, (list, np.ndarray)) or len(memory.embedding) == 0:
                embedding = self.embedding_function([memory.content["text"]])[0]
                memory.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
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
            results = await asyncio.to_thread(
                self.collection.get,
                include=["metadatas"]  
            )
            
            if results["ids"]:
                await asyncio.to_thread(
                    self.collection.delete,
                    ids=results["ids"]
                )
        except Exception as e:
            raise MemoryError(f"Failed to remove all memories: {str(e)}")

    def add_plugin(self, plugin: MemoryRankPlugin) -> None:
        if any(p.name == plugin.name for p in self.plugins):
            raise ValueError(f"Plugin with name {plugin.name} already exists")
        self.plugins.append(plugin)
        
    def remove_plugin(self, plugin_name: str) -> None:
        self.plugins = [p for p in self.plugins if p.name != plugin_name]
        
    def get_plugin(self, plugin_name: str) -> Optional[MemoryRankPlugin]:
        return next((p for p in self.plugins if p.name == plugin_name), None)
        
    def list_plugins(self) -> List[str]:
        return [p.name for p in self.plugins]

    async def get_memories_by_type(
        self,
        memory_type: MemoryType,
        count: int = 5,
        threshold: float = 0.0,
    ) -> List[Memory]:
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                include=["metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                return []
            
            memories = []
            for idx, memory_id in enumerate(results["ids"]):
                try:
                    memory = deserialize_memory(
                        memory_id=memory_id,
                        metadata=results["metadatas"][idx],
                        embedding=results["embeddings"][idx] if "embeddings" in results else None
                    )
                    
                    if memory.memory_type == memory_type:
                        memories.append(memory)
                        
                except Exception as e:
                    print(f"Failed to deserialize memory {memory_id}: {str(e)}")
                    continue
                
            return memories[:count]
            
        except Exception as e:
            raise MemoryError(f"Failed to get memories by type: {str(e)}")

    async def _get_memories_with_filter(
        self,
        where: Optional[Dict[str, Any]],
        count: int,
        threshold: float = 0.0,
        query_embedding: Optional[List[float]] = None
    ) -> List[Memory]:
        try:
            if query_embedding is not None:
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()
                elif not isinstance(query_embedding, list):
                    query_embedding = list(query_embedding)
            
            query_where = None
            if where is not None and where:
                query_where = where
            if query_embedding is None:
                results = await asyncio.to_thread(
                    self.collection.get,
                    where=query_where,
                    limit=count,
                    include=["metadatas", "embeddings"]
                )
                if results["ids"]:
                    results = {
                        "ids": [results["ids"]],
                        "distances": [[1.0] * len(results["ids"])],
                        "metadatas": [results["metadatas"]],
                        "embeddings": [results["embeddings"]] if "embeddings" in results else None
                    }
                else:
                    return []
            else:
                results = await asyncio.to_thread(
                    self.collection.query,
                    query_embeddings=[query_embedding],
                    n_results=count,
                    where=query_where,
                    include=["metadatas", "embeddings", "distances"]
                )
            
            if not isinstance(results["ids"], list) or not results["ids"]:
                return []
            if not isinstance(results["ids"][0], list) or not results["ids"][0]:
                return []
            
            memories = []
            for idx, memory_id in enumerate(results["ids"][0]):
                similarity = None
                if (isinstance(results.get("distances"), list) and 
                    results["distances"] and 
                    isinstance(results["distances"][0], list) and 
                    len(results["distances"][0]) > idx):
                    try:
                        similarity = float(results["distances"][0][idx])
                    except (TypeError, ValueError):
                        continue
                
                embedding = None
                if (isinstance(results.get("embeddings"), list) and 
                    results["embeddings"] and 
                    isinstance(results["embeddings"][0], list) and 
                    len(results["embeddings"][0]) > idx):
                    embedding_data = results["embeddings"][0][idx]
                    if isinstance(embedding_data, (list, np.ndarray)):
                        embedding = list(embedding_data)
                
                try:
                    
                    memory = deserialize_memory(
                        memory_id=memory_id,
                        metadata=results["metadatas"][0][idx],
                        embedding=embedding,
                        similarity=similarity
                    )
                    if similarity is None or similarity >= threshold:
                        memories.append(memory)
                except Exception as e: 
                    print(f"Failed to deserialize memory {memory_id}: {str(e)}")
                    continue
            
            return memories
            
        except Exception as e:
            raise MemoryError(f"Failed to get memories with filter: {str(e)}")

    async def get_recent_memories(
        self,
        count: int = 5,
    ) -> List[Memory]:
        """Get most recent memories based on created_at timestamp"""
        total_count = await asyncio.to_thread(
            self.collection.count,
        )

        results = await asyncio.to_thread(
            self.collection.get,
            offset=total_count - count,
            limit=count,
            include=["metadatas", "embeddings"]
        )

        if not results["ids"]:
            return []
        
        memories = []
        for idx, memory_id in enumerate(results["ids"]):
            try:
                memory = deserialize_memory(
                    memory_id=memory_id,
                    metadata=results["metadatas"][idx],
                    embedding=results["embeddings"][idx] if "embeddings" in results else None
                )
                memories.append(memory)
            except Exception as e:
                print(f"Failed to deserialize memory {memory_id}: {str(e)}")
                continue
        
        memories.sort(key=lambda x: x.created_at, reverse=True)
        return memories[:count]
       