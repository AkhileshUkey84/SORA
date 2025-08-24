# Session context management
# agents/memory_agent.py
"""
Conversation memory agent maintaining session context across interactions.
Essential for multi-turn queries and pronoun resolution (e.g., "filter it by...").
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from agents.base import BaseAgent, AgentContext, AgentResult
from services.cache_service import CacheService
import json


class ConversationTurn(BaseModel):
    """Single turn in conversation history"""
    query: str
    sql: Optional[str] = None
    results_summary: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ConversationMemory(BaseModel):
    """Full conversation state for a session"""
    conversation_id: str
    turns: List[ConversationTurn] = []
    context: Dict[str, Any] = {}  # Domain-specific context
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryAgent(BaseAgent):
    """
    Manages conversational context for coherent multi-turn interactions.
    Resolves references like "show me more" or "group by category instead".
    """
    
    def __init__(self, cache_service: CacheService):
        super().__init__("MemoryAgent")
        self.cache = cache_service
        self.max_history = 10  # Keep last N turns for context
        self.ttl_hours = 24  # Conversation memory retention
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Retrieves and enriches query with conversation history.
        Performs pronoun resolution and context injection.
        """
        
        if not context.conversation_id:
            # First turn - initialize memory
            memory = ConversationMemory(conversation_id=context.session_id)
        else:
            # Retrieve existing conversation
            memory = await self._get_conversation_memory(context.conversation_id)
            if not memory:
                memory = ConversationMemory(conversation_id=context.conversation_id)
        
        # Analyze query for references to previous context
        enriched_query = await self._resolve_references(
            context.query, 
            memory.turns[-3:] if memory.turns else []  # Last 3 turns
        )
        
        # Extract relevant context for SQL generation
        query_context = self._extract_relevant_context(memory)
        
        # Store original query before enrichment
        context.metadata["original_query"] = context.query
        context.metadata["conversation_context"] = query_context
        
        if enriched_query != context.query:
            self.logger.info("Query enriched with context",
                           original=context.query,
                           enriched=enriched_query)
            context.query = enriched_query
        
        return AgentResult(
            success=True,
            data={
                "enriched_query": enriched_query,
                "context": query_context,
                "turn_number": len(memory.turns) + 1
            },
            metadata={"memory_id": memory.conversation_id}
        )
    
    async def save_turn(self, context: AgentContext, sql: str, 
                       results_summary: str) -> None:
        """
        Saves completed turn to conversation history.
        Critical for maintaining context in follow-up questions.
        """
        memory = await self._get_conversation_memory(context.conversation_id)
        if not memory:
            memory = ConversationMemory(conversation_id=context.conversation_id)
        
        turn = ConversationTurn(
            query=context.metadata.get("original_query", context.query),
            sql=sql,
            results_summary=results_summary,
            timestamp=datetime.utcnow(),
            metadata={
                "dataset_id": context.dataset_id,
                "row_count": context.metadata.get("row_count", 0)
            }
        )
        
        memory.turns.append(turn)
        
        # Trim old turns to prevent unbounded growth
        if len(memory.turns) > self.max_history:
            memory.turns = memory.turns[-self.max_history:]
        
        memory.updated_at = datetime.utcnow()
        
        await self._save_conversation_memory(memory)
    
    async def _resolve_references(self, query: str, 
                                recent_turns: List[ConversationTurn]) -> str:
        """
        Resolves pronouns and references using recent conversation context.
        E.g., "show me sales" -> "filter it by region" becomes 
        "filter sales by region"
        """
        if not recent_turns:
            return query
        
        # Simple reference patterns - in production, use NLP
        reference_patterns = [
            ("it", "them", "that", "those"),  # Pronouns
            ("the same", "similar", "more"),   # Continuity markers
            ("instead", "rather", "but")       # Modification markers
        ]
        
        query_lower = query.lower()
        
        # Check for reference indicators
        has_reference = any(
            marker in query_lower 
            for group in reference_patterns 
            for marker in group
        )
        
        if has_reference and recent_turns:
            # Get the main subject from recent queries
            # In production, use dependency parsing
            last_turn = recent_turns[-1]
            
            # Simple heuristic: extract table/column references
            if last_turn.sql:
                # Extract table name from FROM clause
                from_match = re.search(r'FROM\s+(\w+)', last_turn.sql, re.I)
                if from_match:
                    table_ref = from_match.group(1)
                    
                    # Replace "it" with table reference
                    enriched = re.sub(
                        r'\bit\b', 
                        f"the {table_ref} data", 
                        query, 
                        flags=re.I
                    )
                    
                    return enriched
        
        return query
    
    async def _get_conversation_memory(self, 
                                     conversation_id: str) -> Optional[ConversationMemory]:
        """Retrieves conversation from cache with fallback handling"""
        try:
            cached = await self.cache.get(f"conversation:{conversation_id}")
            if cached:
                return ConversationMemory.parse_raw(cached)
        except Exception as e:
            self.logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _save_conversation_memory(self, memory: ConversationMemory) -> None:
        """Persists conversation to cache with TTL"""
        try:
            await self.cache.set(
                f"conversation:{memory.conversation_id}",
                memory.json(),
                ttl=self.ttl_hours * 3600
            )
        except Exception as e:
            self.logger.warning("Cache save failed", error=str(e))
    
    def _extract_relevant_context(self, memory: ConversationMemory) -> Dict[str, Any]:
        """
        Extracts key context for SQL generation from conversation history.
        Includes previously used filters, groupings, and aggregations.
        """
        context = {
            "turn_count": len(memory.turns),
            "previous_tables": set(),
            "previous_filters": [],
            "previous_aggregations": []
        }
        
        for turn in memory.turns[-3:]:  # Last 3 turns
            if turn.sql:
                # Extract table names
                tables = re.findall(r'FROM\s+(\w+)', turn.sql, re.I)
                context["previous_tables"].update(tables)
                
                # Extract WHERE conditions
                where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', 
                                      turn.sql, re.I | re.S)
                if where_match:
                    context["previous_filters"].append(where_match.group(1).strip())
                
                # Extract aggregations
                agg_functions = re.findall(r'(COUNT|SUM|AVG|MAX|MIN)\s*KATEX_INLINE_OPEN', 
                                         turn.sql, re.I)
                context["previous_aggregations"].extend(agg_functions)
        
        context["previous_tables"] = list(context["previous_tables"])
        
        return context