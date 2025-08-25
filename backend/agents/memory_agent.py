# Session context management
# agents/memory_agent.py
"""
Conversation memory agent maintaining session context across interactions.
Essential for multi-turn queries and pronoun resolution (e.g., "filter it by...").
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from agents.base import BaseAgent, AgentContext, AgentResult
from services.cache_service import CacheService
import json
import re


class ConversationTurn(BaseModel):
    """Single turn in conversation history"""
    query: str
    sql: Optional[str] = None
    results_summary: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class EntityResolution(BaseModel):
    """Tracks resolved entities and their meanings"""
    entity: str  # The ambiguous term (e.g., "it", "recent", "sales")
    resolved_to: str  # What it refers to (e.g., "revenue", "last_30_days")
    confidence: float  # Confidence in the resolution
    turn_number: int  # Which turn established this resolution
    expires_after: Optional[int] = None  # Turns after which this expires


class ContextChain(BaseModel):
    """Tracks context chains across conversation turns"""
    primary_entity: str  # Main subject (e.g., "sales")
    related_entities: List[str] = []  # Related terms mentioned
    time_context: Optional[str] = None  # Time period context
    filter_context: Dict[str, str] = {}  # Applied filters
    metric_context: Optional[str] = None  # Primary metric being analyzed
    confidence: float = 1.0  # Confidence in the chain


class ConversationMemory(BaseModel):
    """Full conversation state for a session with enhanced intelligence"""
    conversation_id: str
    turns: List[ConversationTurn] = []
    context: Dict[str, Any] = {}  # Domain-specific context
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Enhanced intelligence features
    entity_resolutions: List[EntityResolution] = []
    context_chain: Optional[ContextChain] = None
    domain_context: Dict[str, Any] = {}  # Business domain insights
    confidence_history: List[float] = []  # Track confidence over time


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
        
        # Enhanced intelligence parameters
        self.entity_confidence_threshold = 0.7  # Minimum confidence for entity resolution
        self.context_chain_max_age = 5  # Max turns to maintain context chain
        
        # Business-aware entity patterns
        self.business_entity_patterns = self._initialize_business_patterns()
        
        # Compiled regex patterns for entity extraction
        self.entity_patterns = {
            'metric_reference': re.compile(r'\b(sales|revenue|customers?|products?)\b', re.IGNORECASE),
            'time_reference': re.compile(r'\b(recent|last|current|Q[1-4]|this|previous)\s+\w+\b', re.IGNORECASE),
            'comparison': re.compile(r'\b(vs|compared?\s+to|against)\b', re.IGNORECASE),
            'aggregation': re.compile(r'\b(total|sum|average|avg|count|max|min)\b', re.IGNORECASE)
        }
    
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
                agg_functions = re.findall(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', 
                                         turn.sql, re.I)
                context["previous_aggregations"].extend(agg_functions)
        
        context["previous_tables"] = list(context["previous_tables"])
        
        return context
    
    def _initialize_business_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize business-aware entity resolution patterns"""
        return {
            "temporal_entities": {
                "recent": ["last_30_days", "this_month", "past_week"],
                "current": ["this_quarter", "this_year", "current_period"],
                "previous": ["last_quarter", "last_year", "prior_period"]
            },
            "metric_entities": {
                "sales": ["revenue", "total_sales", "sales_amount"],
                "performance": ["efficiency", "conversion_rate", "roi"],
                "volume": ["quantity", "units_sold", "transaction_count"]
            },
            "dimension_entities": {
                "geographic": ["region", "country", "state", "city"],
                "categorical": ["category", "type", "segment", "industry"],
                "temporal": ["month", "quarter", "year", "week"]
            }
        }
    
    async def _enhanced_resolve_references(
        self, 
        query: str, 
        memory: ConversationMemory, 
        context: AgentContext
    ) -> Tuple[str, float]:
        """
        Enhanced reference resolution with entity tracking and confidence scoring.
        Returns enriched query and confidence score (0-1).
        """
        if not memory.turns:
            return query, 1.0
        
        confidence = 1.0
        enriched_query = query
        
        # Extract entities from current query
        current_entities = self._extract_entities(query)
        
        # Look for entity references in recent turns
        entity_mappings = self._build_entity_mappings(memory.turns[-3:])
        
        # Resolve entity references
        for entity_type, current_entities_list in current_entities.items():
            for entity in current_entities_list:
                if entity.lower() in ['it', 'that', 'this', 'them']:
                    # Find best match from context
                    best_match = self._find_best_entity_match(
                        entity_type, entity_mappings
                    )
                    if best_match:
                        enriched_query = enriched_query.replace(
                            entity, best_match['entity']
                        )
                        confidence *= best_match['confidence']
        
        # Handle ambiguous time references
        enriched_query, time_confidence = self._resolve_temporal_references(
            enriched_query, memory.turns[-2:] if len(memory.turns) >= 2 else []
        )
        confidence *= time_confidence
        
        return enriched_query, confidence
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query using business-aware patterns.
        """
        entities = {
            'metrics': [],
            'temporal': [],
            'comparisons': [],
            'operations': []
        }
        
        query_lower = query.lower()
        
        # Extract metric entities
        metric_matches = self.entity_patterns['metric_reference'].findall(query)
        entities['metrics'] = metric_matches
        
        # Extract temporal entities  
        temporal_matches = self.entity_patterns['time_reference'].findall(query)
        entities['temporal'] = temporal_matches
        
        # Extract comparison indicators
        comparison_matches = self.entity_patterns['comparison'].findall(query)
        entities['comparisons'] = comparison_matches
        
        # Extract aggregation operations
        operation_matches = self.entity_patterns['aggregation'].findall(query)
        entities['operations'] = operation_matches
        
        return entities
    
    def _build_entity_mappings(self, recent_turns: List[ConversationTurn]) -> Dict[str, Any]:
        """
        Build entity mappings from recent conversation turns.
        """
        mappings = {
            'metrics': {},
            'temporal': {},
            'tables': {},
            'columns': {}
        }
        
        for turn in recent_turns:
            if not turn.sql:
                continue
            
            # Extract table references
            tables = re.findall(r'FROM\s+(\w+)', turn.sql, re.I)
            for table in tables:
                mappings['tables'][table] = {
                    'entity': table,
                    'confidence': 0.9,
                    'context': turn.query
                }
            
            # Extract column references from SELECT
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', turn.sql, re.I | re.S)
            if select_match:
                columns = [col.strip() for col in select_match.group(1).split(',')]
                for col in columns:
                    # Clean column name
                    clean_col = re.sub(r'\(.*\)|AS\s+\w+', '', col, flags=re.I).strip()
                    if clean_col and clean_col != '*':
                        mappings['columns'][clean_col] = {
                            'entity': clean_col,
                            'confidence': 0.8,
                            'context': turn.query
                        }
        
        return mappings
    
    def _find_best_entity_match(
        self, 
        entity_type: str, 
        mappings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find best entity match with confidence scoring.
        """
        if entity_type not in mappings or not mappings[entity_type]:
            return None
        
        # Return most confident match
        best_match = max(
            mappings[entity_type].values(),
            key=lambda x: x['confidence']
        )
        
        return best_match if best_match['confidence'] > 0.5 else None
    
    def _resolve_temporal_references(
        self, 
        query: str, 
        recent_turns: List[ConversationTurn]
    ) -> Tuple[str, float]:
        """
        Resolve ambiguous temporal references like 'recent', 'last quarter'.
        """
        confidence = 1.0
        enriched_query = query
        
        # Common temporal patterns
        temporal_patterns = {
            r'\brecent\b': 'last 30 days',
            r'\blast quarter\b': 'Q3 2024',  # Would be dynamic in production
            r'\bthis year\b': '2024',
            r'\blast month\b': 'previous month'
        }
        
        for pattern, replacement in temporal_patterns.items():
            if re.search(pattern, query, re.I):
                # In production, would check context for specific dates
                enriched_query = re.sub(pattern, replacement, enriched_query, flags=re.I)
                confidence *= 0.8  # Reduce confidence for assumptions
        
        return enriched_query, confidence
    
    def _build_context_chain(self, memory: ConversationMemory) -> List[Dict[str, Any]]:
        """
        Build context chain showing conversation flow and entity evolution.
        """
        chain = []
        
        for i, turn in enumerate(memory.turns):
            entities = self._extract_entities(turn.query)
            
            chain_entry = {
                'turn_number': i + 1,
                'query': turn.query,
                'entities': entities,
                'timestamp': turn.timestamp,
                'context_score': self._calculate_context_score(turn, memory.turns[:i])
            }
            
            chain.append(chain_entry)
        
        return chain
    
    def _calculate_context_score(self, turn: ConversationTurn, previous_turns: List[ConversationTurn]) -> float:
        """
        Calculate how well this turn builds on previous context.
        """
        if not previous_turns:
            return 1.0
        
        # Score based on entity overlap and reference patterns
        current_entities = self._extract_entities(turn.query)
        
        # Check for explicit references
        reference_indicators = ['it', 'that', 'this', 'them', 'same', 'similar']
        has_references = any(indicator in turn.query.lower() for indicator in reference_indicators)
        
        if has_references:
            return 0.9  # High context dependency
        
        # Check entity continuity
        if previous_turns:
            prev_entities = self._extract_entities(previous_turns[-1].query)
            entity_overlap = self._calculate_entity_overlap(current_entities, prev_entities)
            return max(0.5, entity_overlap)  # Minimum 50% context score
        
        return 0.7  # Default context score
    
    def _calculate_entity_overlap(self, entities1: Dict[str, List[str]], entities2: Dict[str, List[str]]) -> float:
        """
        Calculate overlap between two entity sets.
        """
        total_entities = 0
        overlapping_entities = 0
        
        for entity_type in entities1:
            total_entities += len(entities1[entity_type])
            if entity_type in entities2:
                overlap = len(set(entities1[entity_type]) & set(entities2[entity_type]))
                overlapping_entities += overlap
        
        return overlapping_entities / max(1, total_entities)
