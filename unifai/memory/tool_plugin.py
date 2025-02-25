from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from .plugin import MemoryRankPlugin, PluginContext, RankingResult
from .base import Memory, ToolInfo

class ToolSimilarityConfig:
    """Configuration for tool similarity plugin"""
    def __init__(
        self, 
        tool_match_bonus: float = 0.5,
        sequence_bonus: float = 0.6,
        recency_weight: float = 0.2
    ):
        self.tool_match_bonus = tool_match_bonus
        self.sequence_bonus = sequence_bonus
        self.recency_weight = recency_weight

class ToolSimilarityPlugin(MemoryRankPlugin[ToolSimilarityConfig]):
    """Plugin that ranks memories based on tool usage patterns and similarity"""
    
    def __init__(self, weight: float = 0.3, config: Optional[ToolSimilarityConfig] = None):
        super().__init__(config or ToolSimilarityConfig())
        self.weight = weight
        self._tool_sequence_cache: Dict[str, Tuple[List[str], datetime]] = {}
    
    def _get_tool_sequence(self, memory: Memory) -> List[str]:
        """Extract ordered sequence of tool names from memory"""
        if not memory.tools:
            return []
        return [tool.name for tool in memory.tools]
    
    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two tool sequences using longest common subsequence"""
        if not seq1 or not seq2:
            return 0.0
            
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        lcs_length = dp[m][n]
        return 2.0 * lcs_length / (m + n)  
    
    async def calculate_scores(
        self,
        memories: List[Memory],
        context: PluginContext
    ) -> Dict[str, float]:
        scores = {}
        
        context_tools: Set[str] = set()
        context_sequence = []
        recent_memories = sorted(
            [m for m in memories if m.tools],
            key=lambda x: x.created_at,
            reverse=True
        )
        if not self.config:
            self.config = ToolSimilarityConfig()
        
        for memory in recent_memories:
            if memory.tools:
                context_tools.update(tool.name for tool in memory.tools)
                context_sequence.extend(self._get_tool_sequence(memory))
        
        if not context_tools:
            return {str(memory.id): 0.0 for memory in memories}
            
        for memory in memories:
            memory_id = str(memory.id)
            if not memory.tools:
                scores[memory_id] = 0.0
                continue
                
            memory_tools = set(tool.name for tool in memory.tools)
            memory_sequence = self._get_tool_sequence(memory)
            
            intersection = len(memory_tools.intersection(context_tools))
            union = len(memory_tools.union(context_tools))
            
            if union == 0:
                base_score = 0.0
            else:
                base_score = intersection / union
                
            if intersection > 0:
                base_score += self.config.tool_match_bonus
                
            if context_sequence and memory_sequence:
                sequence_sim = self._calculate_sequence_similarity(
                    context_sequence,
                    memory_sequence
                )
                base_score += sequence_sim * self.config.sequence_bonus

            if memory.created_at:
                recency_factor = 1.0
                for idx, recent in enumerate(recent_memories):
                    if memory.id == recent.id:
                        recency_factor = 1.0 - (idx * self.config.recency_weight)
                        break
                base_score *= recency_factor
                
            scores[memory_id] = min(base_score, 1.0)
                
        return scores