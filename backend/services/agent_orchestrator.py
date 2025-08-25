# services/agent_orchestrator.py
"""
Master orchestrator coordinating all agents in the pipeline.
This service also handles file uploads and processing.
"""

from typing import Dict, Any, Optional, List
from agents.base import AgentContext, AgentResult
from agents.memory_agent import MemoryAgent
from agents.router_agent import RouterAgent
from agents.planner_agent import PlannerAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.validator_agent import ValidatorAgent
from agents.executor_agent import ExecutorAgent
from agents.explainer_agent import ExplainerAgent
from agents.insight_agent import InsightDiscoveryAgent
from agents.audit_agent import AuditAgent
from agents.lineage_agent import LineageAgent
from agents.report_agent import ReportAgent
from models.responses import QueryResult, ChartConfig
from services.llm_service import LLMService
from services.database_service import DatabaseService
from services.cache_service import CacheService
from utils.security import PIIDetector
from utils.visualization import ChartGenerator
from utils.config import settings
import asyncio
import structlog
import pandas as pd
import numpy as np
import re
from datetime import datetime


logger = structlog.get_logger()


class AgentOrchestrator:
    """
    Coordinates the full agent pipeline with proper error handling.
    Designed for maximum demo impact with fallback strategies.
    """
    
    def __init__(self, db_service=None):
        # Initialize services
        self.llm_service = LLMService()
        self.db_service = db_service or DatabaseService()
        self.cache_service = CacheService()
        self.pii_detector = PIIDetector()
        self.chart_generator = ChartGenerator()
        
        # Initialize agents (but only after db_service is set)
        self._agents_initialized = False
        self.logger = logger.bind(component="AgentOrchestrator")
        
    def _initialize_agents(self):
        """Initialize agents with the current db_service"""
        if not self._agents_initialized:
            self.memory_agent = MemoryAgent(self.cache_service)
            self.router_agent = RouterAgent()
            self.planner_agent = PlannerAgent()
            self.sql_generator = SQLGeneratorAgent(self.llm_service, self.db_service)
            self.validator = ValidatorAgent(self.pii_detector, settings)
            self.executor = ExecutorAgent(self.db_service)
            self.explainer = ExplainerAgent(self.llm_service)
            self.insight_agent = InsightDiscoveryAgent()
            self.audit_agent = AuditAgent(self.db_service)
            self.lineage_agent = LineageAgent()
            self.report_agent = ReportAgent(self.cache_service)
            self._agents_initialized = True
    
    async def initialize(self):
        """Initialize all services"""
        await self.db_service.initialize()
        await self.cache_service.initialize()
        
    async def process_file_upload(self, file, user_id, name=None, description=None):
        """Process file upload and return metadata"""
        try:
            # Initialize storage service
            from services.storage_service import StorageService
            storage_service = StorageService()
            
            # Ensure database service is initialized
            if not hasattr(self.db_service, 'main_conn'):
                await self.db_service.initialize()
            
            # Upload and process file
            upload_result = await storage_service.upload_dataset(
                file=file,
                user_id=user_id,
                dataset_name=name
            )
            
            # Create database view
            schema = await self.db_service.create_dataset_view(
                dataset_id=upload_result["dataset_id"],
                df=upload_result["dataframe"]
            )
            
            return {
                "success": True,
                "dataset_id": upload_result["dataset_id"],
                "metadata": upload_result["metadata"],
                "preview": upload_result["metadata"]["preview"],
                "warnings": []
            }
            
        except Exception as e:
            self.logger.error("File upload processing failed",
                            user_id=user_id,
                            error=str(e),
                            exc_info=True)
            
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}"
            }
    
    async def process_query(
        self,
        context: AgentContext,
        enable_insights: bool = True,
        enable_audit: bool = True,
        enable_multimodal: bool = True
    ) -> QueryResult:
        """
        Executes the full pipeline from natural language to rich results.
        Each stage enhances the response for maximum demo impact.
        """
        
        self.logger.info("Starting query processing",
                        session_id=context.session_id,
                        query=context.query[:100])
        
        # Ensure agents are initialized
        self._initialize_agents()
        
        try:
            # Stage 1: Memory and Context Enhancement
            memory_result = await self.memory_agent.execute(context)
            if not memory_result.success:
                self.logger.warning("Memory agent failed, continuing without context")
            
            # Stage 2: Domain Routing
            routing_result = await self.router_agent.execute(context)
            if routing_result.success:
                context.metadata.update(routing_result.data)
            
            # Stage 3: Query Planning and Clarification
            planning_result = await self.planner_agent.execute(context)
            
            if planning_result.data.get("needs_clarification"):
                return self._create_clarification_response(
                    planning_result.data["clarification_questions"]
                )
            
            # Update context with planning insights
            context.metadata["query_intent"] = planning_result.data.get("intent")
            context.metadata["query_components"] = planning_result.data.get("components", {})
            
            # Stage 4: SQL Generation
            sql_result = await self.sql_generator.execute(context)
            
            if not sql_result.success:
                return self._create_error_response(
                    sql_result.error or "Failed to generate SQL query"
                )
            
            sql_query = sql_result.data["sql"]
            
            # Stage 5: Validation and Security
            validation_result = await self.validator.execute(context, **{"sql": sql_query})
            
            if not validation_result.success:
                return self._create_security_response(validation_result)
            
            validated_sql = validation_result.data["validated_sql"]
            
            # Stage 6: Execution
            exec_result = await self.executor.execute(context, **{"sql": validated_sql})
            
            if not exec_result.success:
                return self._create_error_response(
                    exec_result.error or "Query execution failed"
                )
            
            results = exec_result.data
            
            # Stage 7: Parallel Enhancement
            enhancement_tasks = []
            
            # Always run explanation
            enhancement_tasks.append(
                self._safe_execute(
                    self.explainer.execute(context, **{"sql": validated_sql, "results": results}),
                    "Explainer"
                )
            )
            
            # Conditional enhancements
            if enable_insights and len(results.get("data", [])) > 0:
                enhancement_tasks.append(
                    self._safe_execute(
                        self.insight_agent.execute(context, **{"results": results}),
                        "InsightDiscovery"
                    )
                )
            
            if enable_audit:
                enhancement_tasks.append(
                    self._safe_execute(
                        self.audit_agent.execute(context, **{"sql": validated_sql, "results": results}),
                        "Audit"
                    )
                )
            
            # Always track lineage
            enhancement_tasks.append(
                self._safe_execute(
                    self.lineage_agent.execute(context, **{"sql": validated_sql}),
                    "Lineage"
                )
            )
            
            # Run enhancements in parallel
            enhancement_results = await asyncio.gather(*enhancement_tasks)
            
            # Stage 8: Compile Final Response
            response = await self._compile_response(
                context=context,
                sql_query=validated_sql,
                results=results,
                validation_result=validation_result,
                enhancement_results=enhancement_results,
                enable_multimodal=enable_multimodal
            )
            
            # Stage 9: Save to Memory
            if memory_result.success:
                await self.memory_agent.save_turn(
                    context,
                    validated_sql,
                    response.narrative[:200] if response.narrative else "Query completed"
                )
            
            self.logger.info("Query processing completed successfully",
                           session_id=context.session_id,
                           execution_time_ms=response.execution_time_ms)
            
            return response
            
        except Exception as e:
            self.logger.error("Pipeline failed",
                            session_id=context.session_id,
                            error=str(e),
                            exc_info=True)
            
            return self._create_error_response(
                "An unexpected error occurred. Please try again."
            )
    
    async def _safe_execute(self, coro, agent_name: str) -> AgentResult:
        """Safely execute an agent with error handling"""
        try:
            return await coro
        except Exception as e:
            self.logger.error(f"{agent_name} agent failed",
                            error=str(e),
                            exc_info=True)
            return AgentResult(
                success=False,
                error=f"{agent_name} enhancement failed"
            )
    
    async def _compile_response(
        self,
        context: AgentContext,
        sql_query: str,
        results: Dict[str, Any],
        validation_result: AgentResult,
        enhancement_results: List[AgentResult],
        enable_multimodal: bool
    ) -> QueryResult:
        """
        Compiles all agent outputs into a rich, cohesive response.
        Prioritizes user value and actionability.
        """
        
        # Extract enhancement results safely
        explanation = enhancement_results[0].data if enhancement_results[0].success else {}
        insights = enhancement_results[1].data if len(enhancement_results) > 1 and enhancement_results[1].success else {}
        audit = enhancement_results[2].data if len(enhancement_results) > 2 and enhancement_results[2].success else {}
        lineage = enhancement_results[3].data if len(enhancement_results) > 3 and enhancement_results[3].success else {}
        
        # Generate visualization config if requested
        chart_config = None
        if enable_multimodal and results.get("data"):
            try:
                chart_dict = self.chart_generator.generate_chart_config(
                    results["data"],
                    context.metadata,
                    insights.get("insights", [])
                )
                if chart_dict:
                    chart_config = ChartConfig(**chart_dict)
            except Exception as e:
                self.logger.warning("Chart generation failed", error=str(e))
        
        # Generate follow-up suggestions
        suggested_actions = self._generate_suggestions(insights, audit)
        
        # Compile drill-down queries
        drill_down_queries = []
        if insights.get("insights"):
            for insight in insights["insights"][:3]:
                if insight.get("follow_up_query"):
                    drill_down_queries.append(insight["follow_up_query"])
        
        # Calculate execution time
        exec_time = results.get("execution_time_ms", 0)
        
        # Convert any numpy types in the results before creating the response
        from utils.json_encoder import convert_numpy_types
        
        # Clean up results data
        clean_results = convert_numpy_types({
            "columns": results.get("columns", []),
            "data": results.get("data", []),
            "row_count": len(results.get("data", [])),
            "execution_time_ms": exec_time
        })
        
        # Clean up other potentially problematic data
        clean_insights = convert_numpy_types(insights.get("insights", []))
        clean_audit = convert_numpy_types(audit) if audit else None
        
        return QueryResult(
            success=True,
            sql_query=sql_query,
            results=clean_results,
            narrative=explanation.get("narrative", "Query executed successfully."),
            chart_config=chart_config,
            insights=clean_insights,
            suggested_actions=suggested_actions,
            drill_down_queries=drill_down_queries,
            audit_report=clean_audit,
            security_explanation=validation_result.data.get("explanation"),
            provenance=self._construct_provenance(lineage) if lineage else None,
            session_id=context.session_id,
            conversation_id=context.conversation_id,
            execution_time_ms=exec_time
        )
    
    def _generate_suggestions(self, insights: Dict[str, Any], 
                            audit: Dict[str, Any]) -> List[str]:
        """
        Generates actionable next steps based on results and insights.
        Focuses on business value and exploration paths.
        """
        
        suggestions = []
        
        # Based on insights
        if insights.get("insights"):
            for insight in insights["insights"][:2]:
                if insight["type"] == "outlier":
                    suggestions.append(
                        f"Investigate the {insight['supporting_data'].get('outlier_count', 'identified')} outliers for data quality issues"
                    )
                elif insight["type"] == "trend":
                    suggestions.append(
                        "Create a forecast model to predict future values"
                    )
                elif insight["type"] == "correlation":
                    col1 = insight['supporting_data'].get('column_1', 'first metric')
                    col2 = insight['supporting_data'].get('column_2', 'second metric')
                    suggestions.append(
                        f"Explore causation between {col1} and {col2}"
                    )
        
        # Based on audit confidence
        if audit and audit.get("confidence_score", 1.0) < 0.8:
            suggestions.append(
                "Refine your query for more accurate results"
            )
        
        # Generic valuable suggestions if none generated
        if not suggestions:
            suggestions = [
                "Export results for further analysis",
                "Apply additional filters to focus on specific segments",
                "Compare with historical data for context"
            ]
        
        return suggestions[:3]  # Limit to top 3
    
    def _construct_provenance(self, lineage_data: Dict[str, Any]):
        """Constructs DataProvenance object from LineageAgent output"""
        from models.responses import DataProvenance
        
        if not lineage_data:
            return None
        
        lineage = lineage_data.get("lineage", {})
        
        try:
            return DataProvenance(
                lineage_id=lineage_data.get("lineage_id", lineage.get("query_hash", "unknown")),
                data_sources=lineage.get("data_sources", []),
                transformations=lineage.get("transformations", []),
                dependencies=lineage.get("dependencies", {}),
                timestamp=datetime.fromisoformat(lineage.get("timestamp", datetime.utcnow().isoformat())),
                reproducible=lineage_data.get("reproducible", True)
            )
        except Exception as e:
            self.logger.warning(f"Failed to construct provenance: {e}")
            return None
    
    def _create_clarification_response(self, questions: List[str]) -> QueryResult:
        """Creates response requesting clarification from user"""
        return QueryResult(
            success=True,
            needs_clarification=True,
            clarification_questions=questions,
            narrative="I need a bit more information to answer your question accurately."
        )
    
    def _create_security_response(self, validation_result: AgentResult) -> QueryResult:
        """Creates response explaining security constraints"""
        return QueryResult(
            success=False,
            error="Query blocked for security reasons",
            security_explanation=validation_result.data.get("explanation", "Query validation failed"),
            narrative="Your query was modified or blocked to ensure data security and privacy."
        )
    
    def _create_error_response(self, error: str) -> QueryResult:
        """Creates user-friendly error response"""
        return QueryResult(
            success=False,
            error=error,
            narrative="I encountered an issue processing your request. Please try rephrasing your question."
        )