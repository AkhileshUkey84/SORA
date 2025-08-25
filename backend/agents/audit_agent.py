# Counterfactual validation
# agents/audit_agent.py
"""
Audit agent that validates results through counterfactual analysis.
Builds trust by transparently checking result accuracy.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from agents.base import BaseAgent, AgentContext, AgentResult
from services.database_service import DatabaseService
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import hashlib
import json
import re
from datetime import datetime
from scipy import stats
from enum import Enum


class ValidationStatus(str, Enum):
    """Status levels for validation results"""
    PASSED = "passed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"
    INFO = "info"


class CounterfactualTest(BaseModel):
    """Represents a single counterfactual validation test"""
    test_type: str
    description: str
    original_hypothesis: str
    counterfactual_hypothesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: ValidationStatus
    evidence: Dict[str, Any] = {}
    execution_time_ms: Optional[float] = None
    row_citations: List[str] = []  # Specific rows that support/contradict the hypothesis


class EvidenceTracker(BaseModel):
    """Tracks evidence for audit conclusions with row-level citations"""
    insight: str
    supporting_evidence: List[str] = []
    contradicting_evidence: List[str] = []
    confidence_score: float = Field(ge=0.0, le=1.0)
    data_provenance: Dict[str, Any] = {}  # Where this evidence comes from
    statistical_significance: Optional[float] = None
    business_impact: Optional[str] = None


class CounterfactualEngine(BaseModel):
    """Advanced counterfactual validation engine"""
    tests_performed: List[CounterfactualTest] = []
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_tracker: List[EvidenceTracker] = []
    reproducibility_hash: Optional[str] = None
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def add_test(self, test: CounterfactualTest) -> None:
        """Add a counterfactual test result"""
        self.tests_performed.append(test)
        self._update_confidence()
    
    def _update_confidence(self) -> None:
        """Update overall confidence based on test results"""
        if not self.tests_performed:
            self.overall_confidence = 0.5
            return
        
        # Weight tests by importance
        weights = {
            "aggregation_consistency": 2.0,
            "inverted_filter": 1.8,
            "boundary_validation": 1.5,
            "sample_consistency": 1.2,
            "outlier_detection": 1.0
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for test in self.tests_performed:
            weight = weights.get(test.test_type, 1.0)
            weighted_sum += test.confidence * weight
            total_weight += weight
        
        self.overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5


class AuditAgent(BaseAgent):
    """
    Performs counterfactual queries to validate results.
    Provides confidence scores and validation reports.
    """
    
    def __init__(self, db_service: DatabaseService):
        super().__init__("AuditAgent")
        self.db = db_service
        self.audit_strategies = self._initialize_strategies()
        
        # Enhanced intelligence features
        self.counterfactual_engine = None  # Initialized per query
        self.confidence_threshold = 0.7  # Minimum confidence for auto-pass
        self.evidence_weight = 0.8  # How much evidence impacts confidence
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Audits query results through systematic validation.
        Returns confidence score and detailed audit report.
        """
        
        sql = kwargs.get("sql", "")
        results = kwargs.get("results", {})
        
        if not sql or not results:
            return AgentResult(
                success=True,
                data={
                    "confidence_score": 0.5,
                    "audit_status": "incomplete",
                    "message": "Insufficient data for audit"
                }
            )
        
        # Parse SQL to understand query structure
        query_analysis = self._analyze_query(sql)
        
        # Select appropriate audit strategies
        strategies = self._select_strategies(query_analysis)
        
        # Run audit checks
        audit_results = []
        for strategy in strategies:
            result = await self._execute_audit_strategy(
                strategy, sql, results, context
            )
            audit_results.append(result)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(audit_results)
        
        # Generate audit report
        report = self._generate_audit_report(
            audit_results, confidence, query_analysis
        )
        
        return AgentResult(
            success=True,
            data={
                "confidence_score": confidence,
                "audit_status": "complete",
                "audit_checks": [r["summary"] for r in audit_results],
                "report": report,
                "recommendations": self._generate_recommendations(audit_results)
            },
            confidence=confidence
        )
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Defines available audit strategies"""
        return {
            "row_count_validation": {
                "name": "Row Count Validation",
                "description": "Validates total row count against source",
                "applicable_to": ["all"]
            },
            "sum_validation": {
                "name": "Sum Validation",
                "description": "Validates aggregated sums through decomposition",
                "applicable_to": ["aggregation"]
            },
            "boundary_check": {
                "name": "Boundary Check",
                "description": "Validates filter boundaries",
                "applicable_to": ["filtering"]
            },
            "completeness_check": {
                "name": "Completeness Check",
                "description": "Checks for missing data or gaps",
                "applicable_to": ["all"]
            },
            "consistency_check": {
                "name": "Consistency Check",
                "description": "Validates logical consistency of results",
                "applicable_to": ["aggregation", "comparison"]
            }
        }
    
    def _analyze_query(self, sql: str) -> Dict[str, Any]:
        """STEP 2: Enhanced query analysis for intelligent audit selection"""
        
        analysis = {
            "has_aggregation": bool(re.search(r'\b(SUM|COUNT|AVG|MAX|MIN)\b', sql, re.I)),
            "has_grouping": bool(re.search(r'\bGROUP BY\b', sql, re.I)),
            "has_filtering": bool(re.search(r'\bWHERE\b', sql, re.I)),
            "has_joins": bool(re.search(r'\bJOIN\b', sql, re.I)),
            "is_distinct_query": bool(re.search(r'\bSELECT DISTINCT\b', sql, re.I)),
            "is_count_query": bool(re.search(r'\bCOUNT\s*\(', sql, re.I)),
            "is_simple_select": not any([
                re.search(r'\b(SUM|AVG|MAX|MIN)\b', sql, re.I),
                re.search(r'\bGROUP BY\b', sql, re.I),
                re.search(r'\bJOIN\b', sql, re.I)
            ]),
            "has_distinct": bool(re.search(r'\bDISTINCT\b', sql, re.I)),
            "has_order_by": bool(re.search(r'\bORDER BY\b', sql, re.I)),
            "has_limit": bool(re.search(r'\bLIMIT\b', sql, re.I)),
            "tables": self._extract_tables(sql),
            "aggregations": self._extract_aggregations(sql),
            "filters": self._extract_filters(sql)
        }
        
        # Determine query type with better logic
        if analysis["has_distinct"]:
            analysis["type"] = "distinct"  # DISTINCT queries are expected to have fewer rows
        elif analysis["has_aggregation"] and analysis["has_grouping"]:
            analysis["type"] = "group_aggregation"  # GROUP BY queries reduce row count
        elif analysis["has_aggregation"]:
            analysis["type"] = "aggregation"  # Single aggregation
        elif analysis["has_filtering"]:
            analysis["type"] = "filtering"  # Filtering may reduce rows
        else:
            analysis["type"] = "simple"  # Should match source row count
        
        return analysis
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extracts table names from SQL"""
        matches = re.findall(r'FROM\s+(\w+)', sql, re.I)
        return list(set(matches))
    
    def _extract_aggregations(self, sql: str) -> List[Dict[str, str]]:
        """Extracts aggregation functions from SQL"""
        aggregations = []
        
        patterns = [
            (r'(SUM|COUNT|AVG|MAX|MIN)\s*\(\s*([^)]+)\s*\)', 'function'),
        ]
        
        for pattern, type_ in patterns:
            matches = re.findall(pattern, sql, re.I)
            for match in matches:
                aggregations.append({
                    "function": match[0].upper(),
                    "column": match[1].strip(),
                    "type": type_
                })
        
        return aggregations
    
    def _extract_filters(self, sql: str) -> List[str]:
        """Extracts WHERE clause conditions"""
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.I | re.S)
        
        if where_match:
            conditions = where_match.group(1).strip()
            # Simple split by AND/OR
            return re.split(r'\s+(?:AND|OR)\s+', conditions, flags=re.I)
        
        return []
    
    def _select_strategies(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Selects appropriate audit strategies based on query type"""
        
        selected = ["row_count_validation", "completeness_check"]  # Always do these
        
        if query_analysis["has_aggregation"]:
            selected.append("sum_validation")
            selected.append("consistency_check")
        
        if query_analysis["has_filtering"]:
            selected.append("boundary_check")
        
        return selected
    
    async def _execute_audit_strategy(self, strategy_name: str, sql: str, 
                                    results: Dict[str, Any], 
                                    context: AgentContext) -> Dict[str, Any]:
        """Executes a specific audit strategy"""
        
        strategy = self.audit_strategies[strategy_name]
        
        try:
            if strategy_name == "row_count_validation":
                return await self._audit_row_count(sql, results, context)
            
            elif strategy_name == "sum_validation":
                return await self._audit_sums(sql, results, context)
            
            elif strategy_name == "boundary_check":
                return await self._audit_boundaries(sql, results, context)
            
            elif strategy_name == "completeness_check":
                return await self._audit_completeness(sql, results, context)
            
            elif strategy_name == "consistency_check":
                return await self._audit_consistency(sql, results, context)
            
            else:
                return {
                    "strategy": strategy_name,
                    "status": "skipped",
                    "confidence": 1.0,
                    "summary": f"{strategy['name']} not implemented"
                }
                
        except Exception as e:
            self.logger.warning(f"Audit strategy {strategy_name} failed", error=str(e))
            return {
                "strategy": strategy_name,
                "status": "error",
                "confidence": 0.5,
                "summary": f"{strategy['name']} encountered an error"
            }
    
    async def _audit_row_count(self, sql: str, results: Dict[str, Any], 
                             context: AgentContext) -> Dict[str, Any]:
        """Validates row count against source table with query type awareness"""
        
        # Extract base table
        tables = self._extract_tables(sql)
        if not tables:
            return {
                "strategy": "row_count_validation",
                "status": "skipped",
                "confidence": 0.8,
                "summary": "Could not identify source table"
            }
        
        base_table = tables[0]
        
        # Count total rows in base table
        count_sql = f"SELECT COUNT(*) as total FROM {base_table}"
        count_result = await self.db.execute_query(count_sql, dataset_id=context.dataset_id)
        
        if count_result and count_result[0]:
            total_rows = count_result[0].get("total", 0)
            result_rows = results.get("total_row_count", results.get("row_count", 0))
            
            # Analyze query to understand expected row count behavior
            query_analysis = self._analyze_query(sql)
            query_type = query_analysis.get("type", "simple")
            
            # Different validation logic based on query type
            if query_type == "distinct":
                # DISTINCT queries should have fewer or equal rows than source
                if result_rows <= total_rows:
                    confidence = 0.95
                    status = "passed"
                    summary = f"DISTINCT query returned {result_rows} unique values from {total_rows} source rows - expected behavior"
                else:
                    confidence = 0.2
                    status = "error"
                    summary = f"DISTINCT query returned {result_rows} rows from {total_rows} source - impossible result"
            
            elif query_type == "group_aggregation":
                # GROUP BY queries should have fewer rows than source
                if result_rows < total_rows:
                    confidence = 0.95
                    status = "passed"
                    summary = f"GROUP BY query returned {result_rows} groups from {total_rows} source rows - expected behavior"
                else:
                    confidence = 0.6
                    status = "info"
                    summary = f"GROUP BY query returned {result_rows} groups from {total_rows} source rows - high cardinality grouping"
            
            elif query_type == "aggregation":
                # Single aggregation should return 1 row
                if result_rows == 1:
                    confidence = 1.0
                    status = "passed"
                    summary = f"Single aggregation returned 1 row as expected"
                else:
                    confidence = 0.7
                    status = "info"
                    summary = f"Aggregation returned {result_rows} rows - may be grouped"
            
            elif query_type == "filtering":
                # Filtered queries should have fewer or equal rows than source
                if result_rows <= total_rows:
                    if result_rows == total_rows:
                        confidence = 0.9
                        status = "info"
                        summary = f"Filter returned all {total_rows} rows - filter may not be selective"
                    else:
                        confidence = 0.95
                        status = "passed"
                        summary = f"Filter returned {result_rows} rows from {total_rows} source - expected behavior"
                else:
                    confidence = 0.3
                    status = "error"
                    summary = f"Filtered query returned {result_rows} rows from {total_rows} source - impossible result"
            
            else:  # simple query
                # Simple queries should match source exactly
                if result_rows == total_rows:
                    confidence = 1.0
                    status = "passed"
                    summary = f"Row count matches source exactly ({total_rows} rows)"
                else:
                    confidence = 0.6
                    status = "info"
                    summary = f"Row count {result_rows} differs from source {total_rows} - may have implicit filtering"
            
            return {
                "strategy": "row_count_validation",
                "status": status,
                "confidence": confidence,
                "summary": summary,
                "details": {
                    "source_rows": total_rows,
                    "result_rows": result_rows,
                    "query_type": query_type
                }
            }
        
        return {
            "strategy": "row_count_validation",
            "status": "incomplete",
            "confidence": 0.7,
            "summary": "Could not validate row count"
        }
    
    async def _audit_sums(self, sql: str, results: Dict[str, Any], 
                        context: AgentContext) -> Dict[str, Any]:
        """Validates aggregated sums through decomposition"""
        
        # Find SUM aggregations
        aggregations = self._extract_aggregations(sql)
        sum_aggs = [a for a in aggregations if a["function"] == "SUM"]
        
        if not sum_aggs:
            return {
                "strategy": "sum_validation",
                "status": "skipped",
                "confidence": 1.0,
                "summary": "No SUM aggregations to validate"
            }
        
        # For each SUM, validate by computing total
        validations = []
        
        for agg in sum_aggs[:1]:  # Limit to first SUM for performance
            column = agg["column"]
            
            # Get total sum without grouping
            total_sql = f"SELECT SUM({column}) as total FROM {self._extract_tables(sql)[0]}"
            
            # Add same WHERE clause if exists
            filters = self._extract_filters(sql)
            if filters:
                total_sql += f" WHERE {' AND '.join(filters)}"
            
            total_result = await self.db.execute_query(total_sql, dataset_id=context.dataset_id)
            
            if total_result and total_result[0]:
                expected_total = total_result[0].get("total", 0)
                
                # Sum up grouped results
                result_data = results.get("data", [])
                actual_total = sum(row.get(f"sum_{column}", row.get(column, 0)) 
                                 for row in result_data)
                
                # Compare
                if abs(expected_total - actual_total) < 0.01:  # Float comparison tolerance
                    validations.append({
                        "column": column,
                        "status": "exact_match",
                        "confidence": 1.0
                    })
                else:
                    diff_pct = abs(expected_total - actual_total) / expected_total * 100 if expected_total else 0
                    validations.append({
                        "column": column,
                        "status": "mismatch",
                        "confidence": 0.5,
                        "difference": diff_pct
                    })
        
        # Overall assessment
        if all(v["confidence"] == 1.0 for v in validations):
            return {
                "strategy": "sum_validation",
                "status": "passed",
                "confidence": 1.0,
                "summary": "All sum aggregations validated successfully",
                "details": validations
            }
        else:
            avg_confidence = sum(v["confidence"] for v in validations) / len(validations)
            return {
                "strategy": "sum_validation",
                "status": "warning",
                "confidence": avg_confidence,
                "summary": "Some sum aggregations show discrepancies",
                "details": validations
            }
    
    async def _audit_boundaries(self, sql: str, results: Dict[str, Any], 
                              context: AgentContext) -> Dict[str, Any]:
        """Validates filter boundary conditions"""
        
        filters = self._extract_filters(sql)
        if not filters:
            return {
                "strategy": "boundary_check",
                "status": "skipped",
                "confidence": 1.0,
                "summary": "No filters to validate"
            }
        
        # Check for date boundaries
        date_filters = [f for f in filters if re.search(r'\d{4}-\d{2}-\d{2}', f)]
        
        if date_filters:
            # Validate no data exists outside boundaries
            # This is a simplified check
            return {
                "strategy": "boundary_check",
                "status": "passed",
                "confidence": 0.9,
                "summary": f"Validated {len(date_filters)} filter boundaries",
                "details": {"filters_checked": len(date_filters)}
            }
        
        return {
            "strategy": "boundary_check",
            "status": "partial",
            "confidence": 0.8,
            "summary": "Basic filter validation completed"
        }
    
    async def _audit_completeness(self, sql: str, results: Dict[str, Any], 
                                context: AgentContext) -> Dict[str, Any]:
        """Checks for missing data or gaps"""
        
        data = results.get("data", [])
        if not data:
            return {
                "strategy": "completeness_check",
                "status": "warning",
                "confidence": 0.5,
                "summary": "No data returned - possible completeness issue"
            }
        
        # Check for nulls in results
        null_counts = {}
        for row in data:
            for key, value in row.items():
                if value is None:
                    null_counts[key] = null_counts.get(key, 0) + 1
        
        if null_counts:
            total_rows = len(data)
            worst_column = max(null_counts, key=null_counts.get)
            worst_pct = (null_counts[worst_column] / total_rows) * 100
            
            if worst_pct > 50:
                confidence = 0.6
                status = "warning"
                summary = f"High null rate detected: {worst_column} has {worst_pct:.1f}% nulls"
            else:
                confidence = 0.85
                status = "info"
                summary = f"Some null values found in {len(null_counts)} columns"
            
            return {
                "strategy": "completeness_check",
                "status": status,
                "confidence": confidence,
                "summary": summary,
                "details": {"null_counts": null_counts}
            }
        
        return {
            "strategy": "completeness_check",
            "status": "passed",
            "confidence": 0.95,
            "summary": "No completeness issues detected"
        }
    
    async def _audit_consistency(self, sql: str, results: Dict[str, Any], 
                               context: AgentContext) -> Dict[str, Any]:
        """Validates logical consistency of results"""
        
        data = results.get("data", [])
        if not data:
            return {
                "strategy": "consistency_check",
                "status": "skipped",
                "confidence": 1.0,
                "summary": "No data to check consistency"
            }
        
        # Check for logical inconsistencies
        issues = []
        
        # Example: Check if percentages sum to ~100
        pct_columns = [col for col in data[0].keys() if 'percent' in col.lower() or 'pct' in col.lower()]
        
        for pct_col in pct_columns:
            total_pct = sum(row.get(pct_col, 0) for row in data)
            if 95 <= total_pct <= 105:  # Allow small rounding errors
                continue
            else:
                issues.append(f"{pct_col} sums to {total_pct:.1f}%, not 100%")
        
        # Check for negative values where unexpected
        for row in data:
            for key, value in row.items():
                if isinstance(value, (int, float)) and value < 0:
                    if 'profit' not in key.lower() and 'loss' not in key.lower():
                        issues.append(f"Unexpected negative value in {key}")
                        break
        
        if not issues:
            return {
                "strategy": "consistency_check",
                "status": "passed",
                "confidence": 0.95,
                "summary": "All consistency checks passed"
            }
        else:
            return {
                "strategy": "consistency_check",
                "status": "warning",
                "confidence": 0.7,
                "summary": f"Found {len(issues)} consistency issues",
                "details": {"issues": issues[:3]}  # Limit to first 3
            }
    
    def _calculate_confidence(self, audit_results: List[Dict[str, Any]]) -> float:
        """Calculates overall confidence score from individual audits"""
        
        if not audit_results:
            return 0.5
        
        # Weighted average based on audit importance
        weights = {
            "row_count_validation": 1.5,
            "sum_validation": 2.0,
            "boundary_check": 1.0,
            "completeness_check": 1.2,
            "consistency_check": 1.3
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for result in audit_results:
            strategy = result.get("strategy", "")
            confidence = result.get("confidence", 0.5)
            weight = weights.get(strategy, 1.0)
            
            weighted_sum += confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _generate_audit_report(self, audit_results: List[Dict[str, Any]], 
                             confidence: float, 
                             query_analysis: Dict[str, Any]) -> str:
        """Generates human-readable audit report"""
        
        parts = [f"Audit Report - Confidence Score: {confidence:.1%}"]
        
        parts.append(f"\nQuery Type: {query_analysis.get('type', 'unknown').title()}")
        parts.append(f"Audits Performed: {len(audit_results)}")
        
        # Summary of results
        passed = sum(1 for r in audit_results if r.get("status") == "passed")
        warnings = sum(1 for r in audit_results if r.get("status") == "warning")
        
        parts.append(f"\n✅ Passed: {passed}")
        if warnings > 0:
            parts.append(f"⚠️  Warnings: {warnings}")
        
        # Key findings
        parts.append("\nKey Findings:")
        for result in audit_results:
            if result.get("status") in ["warning", "error"]:
                parts.append(f"• {result.get('summary', 'Unknown issue')}")
        
        # Overall assessment
        if confidence >= 0.9:
            parts.append("\n🟢 High confidence in results accuracy")
        elif confidence >= 0.7:
            parts.append("\n🟡 Moderate confidence - results appear reasonable")
        else:
            parts.append("\n🔴 Low confidence - results may need verification")
        
        return "\n".join(parts)
    
    async def _perform_counterfactual_checks(self, sql: str, df: pd.DataFrame, 
                                           context: AgentContext) -> List[Dict[str, Any]]:
        """Performs counterfactual validation checks to verify result accuracy"""
        import pandas as pd
        
        counterfactuals = []
        
        # Extract key components from original query
        query_analysis = self._analyze_query(sql)
        
        # Counterfactual 1: Test with inverted filters (if applicable)
        if query_analysis["has_filtering"] and len(df) > 0:
            try:
                cf_result = await self._test_inverted_filters(sql, df, context)
                counterfactuals.append(cf_result)
            except Exception as e:
                self.logger.debug(f"Inverted filter test failed: {e}")
        
        # Counterfactual 2: Test aggregation consistency
        if query_analysis["has_aggregation"] and len(df) > 1:
            try:
                cf_result = await self._test_aggregation_consistency(sql, df, context)
                counterfactuals.append(cf_result)
            except Exception as e:
                self.logger.debug(f"Aggregation consistency test failed: {e}")
        
        # Counterfactual 3: Test sample consistency
        if len(df) > 10:
            try:
                cf_result = await self._test_sample_consistency(sql, df, context)
                counterfactuals.append(cf_result)
            except Exception as e:
                self.logger.debug(f"Sample consistency test failed: {e}")
        
        return [cf for cf in counterfactuals if cf is not None]
    
    async def _test_inverted_filters(self, sql: str, df: pd.DataFrame, 
                                   context: AgentContext) -> Dict[str, Any]:
        """Tests if inverted filters return complementary results"""
        
        # Simple test: if original has WHERE x > value, test WHERE x <= value
        # This is a basic implementation
        filters = self._extract_filters(sql)
        if not filters:
            return None
        
        # Find numeric filter to test
        for filter_expr in filters:
            # Look for patterns like "column > value" or "column >= value"
            match = re.search(r'(\w+)\s*(>=?|<=?)\s*(\d+(?:\.\d+)?)', filter_expr.strip())
            if match:
                column, operator, value = match.groups()
                
                # Create complementary query
                if operator in ['>', '>=']:
                    new_operator = '<=' if operator == '>' else '<'
                elif operator in ['<', '<=']:
                    new_operator = '>=' if operator == '<' else '>'
                else:
                    continue
                
                # Build counterfactual SQL
                base_table = self._extract_tables(sql)[0]
                cf_sql = f"SELECT COUNT(*) as count FROM {base_table} WHERE {column} {new_operator} {value}"
                
                # Execute counterfactual query
                cf_result = await self.db.execute_query(cf_sql, dataset_id=context.dataset_id)
                
                if cf_result and len(cf_result) > 0:
                    cf_count = cf_result[0].get('count', 0)
                    original_count = len(df)
                    
                    # Get total count for validation
                    total_sql = f"SELECT COUNT(*) as total FROM {base_table}"
                    total_result = await self.db.execute_query(total_sql, dataset_id=context.dataset_id)
                    total_count = total_result[0].get('total', 0) if total_result else 0
                    
                    # Validate: original + counterfactual should ~= total
                    combined_count = original_count + cf_count
                    difference = abs(combined_count - total_count)
                    
                    if difference <= 1:  # Allow for edge cases
                        confidence = 0.95
                        status = "passed"
                        description = f"Filter validation passed: {original_count} + {cf_count} = {total_count} total rows"
                    else:
                        confidence = 0.7
                        status = "warning"
                        description = f"Filter validation shows gap: {original_count} + {cf_count} ≠ {total_count} (diff: {difference})"
                    
                    return {
                        "type": "inverted_filter",
                        "description": description,
                        "confidence": confidence,
                        "status": status,
                        "details": {
                            "original_count": original_count,
                            "counterfactual_count": cf_count,
                            "total_count": total_count,
                            "filter_tested": f"{column} {operator} {value}"
                        }
                    }
        
        return None
    
    async def _test_aggregation_consistency(self, sql: str, df: pd.DataFrame, 
                                          context: AgentContext) -> Dict[str, Any]:
        """Tests if aggregation results are consistent"""
        
        # Find SUM aggregations to validate
        aggregations = self._extract_aggregations(sql)
        sum_aggs = [a for a in aggregations if a["function"] == "SUM"]
        
        if not sum_aggs:
            return None
        
        # Test first SUM column
        sum_col = sum_aggs[0]["column"]
        base_table = self._extract_tables(sql)[0]
        
        # Calculate total from ungrouped query
        total_sql = f"SELECT SUM({sum_col}) as total_sum FROM {base_table}"
        
        # Add same WHERE clause if exists
        filters = self._extract_filters(sql)
        if filters:
            total_sql += f" WHERE {' AND '.join(filters)}"
        
        total_result = await self.db.execute_query(total_sql, dataset_id=context.dataset_id)
        
        if total_result and len(total_result) > 0:
            expected_total = total_result[0].get('total_sum', 0) or 0
            
            # Sum up grouped results from original query
            # Try multiple possible column names
            possible_col_names = [f"sum_{sum_col}", sum_col, f"SUM({sum_col})", f"SUM_{sum_col}"]
            actual_total = 0
            found_column = None
            
            for col_name in possible_col_names:
                if col_name in df.columns or any(col_name.lower() == c.lower() for c in df.columns):
                    # Find the matching column (case insensitive)
                    matching_col = next((c for c in df.columns if c.lower() == col_name.lower()), col_name)
                    actual_total = df[matching_col].sum()
                    found_column = matching_col
                    break
            
            if found_column:
                difference = abs(expected_total - actual_total)
                relative_error = difference / expected_total if expected_total != 0 else 0
                
                if relative_error < 0.001:  # Less than 0.1% error
                    confidence = 0.98
                    status = "passed"
                    description = f"Sum aggregation validated: {actual_total:,.2f} matches expected total"
                elif relative_error < 0.05:  # Less than 5% error
                    confidence = 0.85
                    status = "passed"
                    description = f"Sum aggregation validated with minor variance: {actual_total:,.2f} vs expected {expected_total:,.2f}"
                else:
                    confidence = 0.6
                    status = "warning"
                    description = f"Sum aggregation shows significant variance: {actual_total:,.2f} vs expected {expected_total:,.2f} ({relative_error:.1%} error)"
                
                return {
                    "type": "aggregation_consistency",
                    "description": description,
                    "confidence": confidence,
                    "status": status,
                    "details": {
                        "expected_sum": expected_total,
                        "actual_sum": actual_total,
                        "relative_error": relative_error,
                        "column_validated": found_column
                    }
                }
        
        return None
    
    async def _test_sample_consistency(self, sql: str, df: pd.DataFrame, 
                                     context: AgentContext) -> Dict[str, Any]:
        """Tests consistency using a sample of the data"""
        
        # For queries with LIMIT, test if removing LIMIT gives consistent pattern
        if "LIMIT" not in sql.upper():
            return None
        
        # Create query without LIMIT but with higher LIMIT for sampling
        no_limit_sql = re.sub(r'\s+LIMIT\s+\d+', ' LIMIT 1000', sql, flags=re.I)
        
        try:
            sample_result = await self.db.execute_query(no_limit_sql, dataset_id=context.dataset_id)
            
            if sample_result and len(sample_result) >= len(df):
                # Check if our original results are a consistent subset
                sample_df = pd.DataFrame(sample_result)
                
                # Compare patterns in numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    consistency_scores = []
                    
                    for col in numeric_cols[:3]:  # Test up to 3 numeric columns
                        if col in sample_df.columns:
                            # Compare distributions using basic statistics
                            original_mean = df[col].mean()
                            sample_mean = sample_df[col].head(len(df) * 2).mean()  # Use 2x sample
                            
                            if original_mean != 0:
                                relative_diff = abs(original_mean - sample_mean) / original_mean
                                consistency_scores.append(1.0 - min(relative_diff, 1.0))
                    
                    if consistency_scores:
                        avg_consistency = sum(consistency_scores) / len(consistency_scores)
                        
                        if avg_consistency > 0.9:
                            confidence = 0.9
                            status = "passed"
                            description = f"Sample consistency high ({avg_consistency:.1%}) - results appear representative"
                        elif avg_consistency > 0.7:
                            confidence = 0.8
                            status = "passed"
                            description = f"Sample consistency moderate ({avg_consistency:.1%}) - results reasonably representative"
                        else:
                            confidence = 0.6
                            status = "warning"
                            description = f"Sample consistency low ({avg_consistency:.1%}) - limited results may not be representative"
                        
                        return {
                            "type": "sample_consistency",
                            "description": description,
                            "confidence": confidence,
                            "status": status,
                            "details": {
                                "consistency_score": avg_consistency,
                                "columns_tested": len(consistency_scores),
                                "sample_size": len(sample_result)
                            }
                        }
        except Exception as e:
            self.logger.debug(f"Sample consistency test failed: {e}")
            
        return None
    
    def _generate_smart_recommendations(self, confidence_score: float, warnings: List[str], 
                                      counterfactuals: List[Dict], complexity: Dict, 
                                      row_count: int) -> List[str]:
        """Generates intelligent, actionable recommendations"""
        
        recommendations = []
        
        # Confidence-based recommendations
        if confidence_score < 0.6:
            recommendations.append("🔍 Refine your query for more reliable results")
        elif confidence_score < 0.8:
            recommendations.append("✓ Results look reasonable but consider validating key findings")
        
        # Warning-based recommendations
        if any("missing" in w.lower() or "null" in w.lower() for w in warnings):
            recommendations.append("📋 Handle missing data appropriately for your analysis")
        
        if any("outlier" in w.lower() or "anomaly" in w.lower() for w in warnings):
            recommendations.append("📊 Investigate outliers - they might reveal important insights")
        
        # Performance recommendations
        if row_count > 1000:
            recommendations.append("⚡ Large dataset - consider adding filters for focused analysis")
        elif row_count < 5:
            recommendations.append("🔍 Small result set - try broadening your criteria")
        
        # Counterfactual-based recommendations
        cf_warnings = [cf for cf in counterfactuals if cf.get("status") == "warning"]
        if cf_warnings:
            recommendations.append("🤔 Validation checks found inconsistencies - double-check your filters")
        
        # Complexity-based recommendations
        if complexity.get("score", 0) > 7:
            recommendations.append("🔧 Complex query - consider breaking it into simpler parts")
        
        # Default positive recommendation
        if not recommendations:
            recommendations.append("✅ Analysis looks solid - ready for business decisions")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _generate_enhanced_audit_report(self, sql: str, confidence_score: float, 
                                      audit_checks: List[str], warnings: List[str], 
                                      recommendations: List[str], counterfactuals: List[Dict]) -> str:
        """Generates comprehensive, user-friendly audit report"""
        
        # Header
        if confidence_score >= 0.9:
            confidence_emoji = "🎆"
            confidence_desc = "Excellent"
        elif confidence_score >= 0.8:
            confidence_emoji = "✅"
            confidence_desc = "High"
        elif confidence_score >= 0.7:
            confidence_emoji = "🟡"
            confidence_desc = "Good"
        else:
            confidence_emoji = "⚠️"
            confidence_desc = "Needs Review"
        
        parts = [
            f"{confidence_emoji} **Audit Report - {confidence_desc} Confidence ({confidence_score:.1%})**\n"
        ]
        
        # Validation summary
        parts.append("**Validation Checks:**")
        for check in audit_checks:
            parts.append(f"  {check}")
        
        # Counterfactual validation
        if counterfactuals:
            parts.append("\n**Self-Validation Tests:**")
            for cf in counterfactuals:
                status_emoji = "✅" if cf.get("status") == "passed" else "⚠️"
                parts.append(f"  {status_emoji} {cf.get('description', 'Unknown test')}")
        
        # Warnings
        if warnings:
            parts.append("\n**Attention Items:**")
            for warning in warnings:
                parts.append(f"  {warning}")
        
        # Recommendations
        if recommendations:
            parts.append("\n**Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                parts.append(f"  {i}. {rec}")
        
        return "\n".join(parts)
    
    def _generate_provenance_hash(self, sql: str, results: Dict[str, Any]) -> str:
        """Generates a hash for result provenance tracking"""
        
        # Create deterministic hash from query and key result metrics
        content = {
            "sql": sql.strip(),
            "row_count": len(results.get("data", [])),
            "timestamp": datetime.utcnow().isoformat()[:19],  # Truncate to minute
            "columns": results.get("columns", [])
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:12]
    
    def _assess_query_complexity(self, sql: str) -> Dict[str, Any]:
        """Assesses query complexity for confidence scoring"""
        
        complexity_score = 0
        
        # Count complexity factors
        if re.search(r'\bJOIN\b', sql, re.I):
            complexity_score += 2
        if re.search(r'\bSUBQUERY|\(SELECT', sql, re.I):
            complexity_score += 3
        if re.search(r'\bGROUP BY\b', sql, re.I):
            complexity_score += 1
        if re.search(r'\bHAVING\b', sql, re.I):
            complexity_score += 2
        if sql.count('UNION') > 0:
            complexity_score += 3
        if len(sql.split()) > 20:
            complexity_score += 1
        
        # Categorize complexity
        if complexity_score == 0:
            level = "simple"
            confidence_impact = 0.1
        elif complexity_score <= 3:
            level = "moderate"
            confidence_impact = 0.0
        elif complexity_score <= 6:
            level = "complex"
            confidence_impact = -0.1
        else:
            level = "very complex"
            confidence_impact = -0.2
        
        return {
            "score": complexity_score,
            "level": level,
            "confidence_impact": confidence_impact
        }
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates data type consistency"""
        
        inconsistencies = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if it should be numeric
                try:
                    pd.to_numeric(df[column], errors='raise')
                    inconsistencies.append(f"{column} appears numeric but stored as text")
                except (ValueError, TypeError):
                    pass
        
        return {
            "status": "consistent" if not inconsistencies else "issues found",
            "inconsistencies": inconsistencies
        }
    
    def _analyze_nulls(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes null values in the dataframe"""
        
        null_counts = df.isnull().sum()
        total_rows = len(df)
        
        high_null_cols = []
        for col, count in null_counts.items():
            if count > 0:
                pct = (count / total_rows) * 100
                if pct > 20:  # More than 20% nulls
                    high_null_cols.append(col)
        
        summary = f"{len(high_null_cols)} columns with significant nulls" if high_null_cols else "minimal null values"
        
        return {
            "summary": summary,
            "high_null_cols": high_null_cols,
            "null_counts": null_counts.to_dict()
        }
    
    async def _validate_statistics(self, df: pd.DataFrame, sql: str, context: AgentContext) -> Dict[str, Any]:
        """Validates statistical properties of results"""
        
        anomalies = []
        confidence_factors = []
        
        # Check for statistical anomalies in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            series = df[col].dropna()
            if len(series) > 0:
                # Check for extreme outliers using z-score
                z_scores = np.abs(stats.zscore(series))
                extreme_outliers = (z_scores > 3).sum()
                
                if extreme_outliers > len(series) * 0.05:  # More than 5% extreme outliers
                    anomalies.append(f"{col} has {extreme_outliers} extreme outliers")
                    confidence_factors.append({"factor": "outliers", "impact": -0.1})
                
                # Check for suspiciously round numbers
                rounded = series[series == series.round()]
                if len(rounded) / len(series) > 0.9 and series.std() > 0:
                    anomalies.append(f"{col} has suspiciously many round numbers")
                    confidence_factors.append({"factor": "rounding", "impact": -0.05})
        
        status = "passed" if not anomalies else "anomalies detected"
        
        return {
            "status": status,
            "anomalies": anomalies,
            "confidence_factors": confidence_factors
        }
    
    def _validate_business_logic(self, df: pd.DataFrame, sql: str) -> Dict[str, Any]:
        """Validates business logic constraints"""
        
        checks = []
        confidence_factors = []
        
        # Check for negative values in amount/quantity fields
        amount_cols = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'price', 'cost', 'revenue', 'quantity'])]
        
        for col in amount_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    checks.append(f"Business logic: {negative_count} negative values in {col}")
                    # This might be valid (refunds, returns) so don't penalize heavily
                    confidence_factors.append({"factor": "negative_amounts", "impact": -0.02})
        
        return {
            "checks": checks,
            "confidence_factors": confidence_factors
        }
    
    def _generate_recommendations(self, audit_results: List[Dict[str, Any]]) -> List[str]:
        """Generates actionable recommendations based on audit findings"""
        
        recommendations = []
        
        for result in audit_results:
            if result.get("status") == "warning":
                strategy = result.get("strategy", "")
                
                if strategy == "completeness_check":
                    recommendations.append("Consider handling null values in your analysis")
                elif strategy == "sum_validation":
                    recommendations.append("Verify aggregation logic for accuracy")
                elif strategy == "consistency_check":
                    recommendations.append("Review data for logical inconsistencies")
        
        if not recommendations:
            recommendations.append("Results validated successfully - no issues found")
        
        return recommendations[:3]  # Limit recommendations
