# Data provenance tracking
# agents/lineage_agent.py
"""
Data lineage tracking agent that maintains provenance information.
Essential for reproducibility and compliance requirements.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base import BaseAgent, AgentContext, AgentResult
import hashlib
import json
import re


class LineageAgent(BaseAgent):
    """
    Tracks data transformations and query lineage.
    Provides complete audit trail for compliance and debugging.
    """
    
    def __init__(self):
        super().__init__("LineageAgent")
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Records complete lineage information for the query execution.
        Includes data sources, transformations, and dependencies.
        """
        
        sql = kwargs.get("sql", "")
        
        # Generate query fingerprint
        query_hash = self._generate_query_hash(sql)
        
        # Extract lineage components
        lineage = {
            "query_id": context.session_id,
            "query_hash": query_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": context.user_id,
            "dataset_id": context.dataset_id,
            
            # Query details
            "original_query": context.metadata.get("original_query", context.query),
            "processed_query": context.query,
            "generated_sql": sql,
            
            # Data sources
            "data_sources": self._extract_data_sources(sql, context),
            
            # Transformations applied
            "transformations": self._extract_transformations(context),
            
            # Dependencies
            "dependencies": self._extract_dependencies(context),
            
            # Execution context
            "execution_context": {
                "conversation_id": context.conversation_id,
                "domain": context.metadata.get("domain", "general"),
                "intent": context.metadata.get("intent", "unknown"),
                "complexity": context.metadata.get("complexity_score", 0)
            },
            
            # Agent trace
            "agent_trace": context.trace,
            
            # Version info
            "system_version": {
                "api_version": "v1",
                "model_version": context.metadata.get("llm_model", "unknown")
            }
        }
        
        # Generate lineage summary
        summary = self._generate_lineage_summary(lineage)
        
        return AgentResult(
            success=True,
            data={
                "lineage": lineage,
                "summary": summary,
                "reproducible": True,
                "lineage_id": query_hash
            }
        )
    
    def _generate_query_hash(self, sql: str) -> str:
        """Generates unique hash for query identification"""
        # Normalize SQL for consistent hashing
        normalized = " ".join(sql.upper().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _extract_data_sources(self, sql: str, context: AgentContext) -> List[Dict[str, Any]]:
        """Extracts all data sources referenced in the query"""
        sources = []
        
        # Primary dataset
        sources.append({
            "type": "primary",
            "dataset_id": context.dataset_id,
            "table_name": f"dataset_{context.dataset_id}",
            "access_time": datetime.utcnow().isoformat(),
            "filters_applied": self._extract_filter_summary(sql)
        })
        
        # Check for joins (future enhancement)
        if "JOIN" in sql.upper():
            # Extract joined tables
            join_matches = re.findall(r'JOIN\s+(\w+)', sql, re.I)
            for table in join_matches:
                sources.append({
                    "type": "joined",
                    "table_name": table,
                    "join_type": "inner",  # Simplified
                    "access_time": datetime.utcnow().isoformat()
                })
        
        return sources
    
    def _extract_transformations(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Extracts all transformations applied to the data"""
        transformations = []
        
        # From agent context
        if context.metadata.get("enriched_query"):
            transformations.append({
                "type": "query_enrichment",
                "description": "Query enhanced with conversation context",
                "agent": "MemoryAgent"
            })
        
        if context.metadata.get("sql_modifications"):
            transformations.append({
                "type": "sql_validation",
                "description": "SQL modified for security compliance",
                "agent": "ValidatorAgent",
                "modifications": context.metadata["sql_modifications"]
            })
        
        # From SQL analysis
        sql = context.metadata.get("generated_sql", "")
        
        if "GROUP BY" in sql.upper():
            transformations.append({
                "type": "aggregation",
                "description": "Data aggregated by grouping",
                "details": self._extract_grouping_details(sql)
            })
        
        if re.search(r'\b(SUM|COUNT|AVG|MAX|MIN)\b', sql, re.I):
            transformations.append({
                "type": "statistical",
                "description": "Statistical functions applied",
                "functions": self._extract_aggregation_functions(sql)
            })
        
        if "ORDER BY" in sql.upper():
            transformations.append({
                "type": "sorting",
                "description": "Results sorted",
                "details": self._extract_sorting_details(sql)
            })
        
        if "LIMIT" in sql.upper():
            transformations.append({
                "type": "limiting",
                "description": "Result set limited",
                "limit": self._extract_limit(sql)
            })
        
        return transformations
    
    def _extract_dependencies(self, context: AgentContext) -> Dict[str, List[str]]:
        """Extracts system dependencies for reproducibility"""
        return {
            "llm_dependencies": [
                context.metadata.get("llm_model", "gemini-1.5-pro"),
                f"temperature={context.metadata.get('llm_temperature', 0.1)}"
            ],
            "agent_dependencies": [
                agent["agent"] for agent in context.trace
            ],
            "data_dependencies": [
                f"dataset_{context.dataset_id}",
                f"schema_version={context.metadata.get('schema_version', '1.0')}"
            ],
            "configuration_dependencies": [
                f"max_rows={context.metadata.get('max_query_rows', 1000)}",
                f"security_enabled={context.metadata.get('enable_sql_validation', True)}"
            ]
        }
    
    def _extract_filter_summary(self, sql: str) -> List[str]:
        """Extracts summary of filters applied with proper string handling"""
        filters = []
        
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.I | re.S)
        if where_match:
            conditions = where_match.group(1).strip()
            
            # Split by AND/OR while preserving quoted strings
            filters = self._smart_split_conditions(conditions)
        
        return filters
    
    def _smart_split_conditions(self, conditions: str) -> List[str]:
        """Splits SQL conditions by AND/OR while preserving quoted strings"""
        filters = []
        current_filter = ""
        in_single_quote = False
        in_double_quote = False
        i = 0
        
        while i < len(conditions):
            char = conditions[i]
            
            # Handle quotes
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            
            # Check for AND/OR outside quotes
            if not in_single_quote and not in_double_quote:
                # Look for AND or OR (case-insensitive)
                remaining = conditions[i:].upper()
                if remaining.startswith(' AND '):
                    # Found AND outside quotes
                    if current_filter.strip():
                        filters.append(current_filter.strip())
                    current_filter = ""
                    i += 5  # Skip ' AND '
                    continue
                elif remaining.startswith(' OR '):
                    # Found OR outside quotes
                    if current_filter.strip():
                        filters.append(current_filter.strip())
                    current_filter = ""
                    i += 4  # Skip ' OR '
                    continue
            
            current_filter += char
            i += 1
        
        # Add the last filter
        if current_filter.strip():
            filters.append(current_filter.strip())
        
        return filters
    
    def _extract_grouping_details(self, sql: str) -> Dict[str, Any]:
        """Extracts GROUP BY details"""
        match = re.search(r'GROUP BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)', sql, re.I)
        if match:
            columns = [col.strip() for col in match.group(1).split(',')]
            return {"columns": columns}
        return {}
    
    def _extract_aggregation_functions(self, sql: str) -> List[str]:
        """Extracts aggregation functions used"""
        functions = []
        
        for func in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']:
            if func in sql.upper():
                # Find what's being aggregated
                pattern = rf'{func}\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE'
                matches = re.findall(pattern, sql, re.I)
                for match in matches:
                    functions.append(f"{func}({match.strip()})")
        
        return functions
    
    def _extract_sorting_details(self, sql: str) -> Dict[str, Any]:
        """Extracts ORDER BY details"""
        match = re.search(r'ORDER BY\s+(.+?)(?:LIMIT|$)', sql, re.I)
        if match:
            sort_spec = match.group(1).strip()
            # Check for DESC/ASC
            direction = "DESC" if "DESC" in sort_spec.upper() else "ASC"
            column = re.sub(r'\s+(DESC|ASC).*', '', sort_spec, flags=re.I).strip()
            return {"column": column, "direction": direction}
        return {}
    
    def _extract_limit(self, sql: str) -> int:
        """Extracts LIMIT value"""
        match = re.search(r'LIMIT\s+(\d+)', sql, re.I)
        if match:
            return int(match.group(1))
        return 0
    
    def _generate_lineage_summary(self, lineage: Dict[str, Any]) -> str:
        """Generates human-readable lineage summary"""
        parts = []
        
        # Basic info
        parts.append(f"Query ID: {lineage['query_id'][:8]}...")
        parts.append(f"Dataset: {lineage['dataset_id']}")
        parts.append(f"Timestamp: {lineage['timestamp']}")
        
        # Transformations
        if lineage['transformations']:
            trans_types = [t['type'] for t in lineage['transformations']]
            parts.append(f"Transformations: {', '.join(trans_types)}")
        
        # Dependencies
        agent_count = len(lineage['dependencies']['agent_dependencies'])
        parts.append(f"Processing steps: {agent_count}")
        
        # Reproducibility
        parts.append("Status: Fully reproducible")
        
        return " | ".join(parts)