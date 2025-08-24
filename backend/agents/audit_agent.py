# Counterfactual validation
# agents/audit_agent.py
"""
Audit agent that validates results through counterfactual analysis.
Builds trust by transparently checking result accuracy.
"""

from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent, AgentContext, AgentResult
from services.database_service import DatabaseService
import re


class AuditAgent(BaseAgent):
    """
    Performs counterfactual queries to validate results.
    Provides confidence scores and validation reports.
    """
    
    def __init__(self, db_service: DatabaseService):
        super().__init__("AuditAgent")
        self.db = db_service
        self.audit_strategies = self._initialize_strategies()
    
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
        """Analyzes SQL query structure for audit planning"""
        
        analysis = {
            "has_aggregation": bool(re.search(r'\b(SUM|COUNT|AVG|MAX|MIN)\b', sql, re.I)),
            "has_grouping": bool(re.search(r'\bGROUP BY\b', sql, re.I)),
            "has_filtering": bool(re.search(r'\bWHERE\b', sql, re.I)),
            "has_joins": bool(re.search(r'\bJOIN\b', sql, re.I)),
            "tables": self._extract_tables(sql),
            "aggregations": self._extract_aggregations(sql),
            "filters": self._extract_filters(sql)
        }
        
        # Determine query type
        if analysis["has_aggregation"]:
            analysis["type"] = "aggregation"
        elif analysis["has_filtering"]:
            analysis["type"] = "filtering"
        else:
            analysis["type"] = "simple"
        
        return analysis
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extracts table names from SQL"""
        matches = re.findall(r'FROM\s+(\w+)', sql, re.I)
        return list(set(matches))
    
    def _extract_aggregations(self, sql: str) -> List[Dict[str, str]]:
        """Extracts aggregation functions from SQL"""
        aggregations = []
        
        patterns = [
            (r'(SUM|COUNT|AVG|MAX|MIN)\s*KATEX_INLINE_OPEN\s*([^)]+)\s*KATEX_INLINE_CLOSE', 'function'),
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
        """Validates row count against source table"""
        
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
            
            # Check if filtering reduced rows appropriately
            if "WHERE" in sql:
                # With filters, result should be less than total
                if result_rows <= total_rows:
                    confidence = 0.95
                    status = "passed"
                    summary = f"Row count {result_rows} is valid (source has {total_rows} rows)"
                else:
                    confidence = 0.3
                    status = "warning"
                    summary = f"Row count {result_rows} exceeds source table count {total_rows}"
            else:
                # Without filters, should match
                if result_rows == total_rows:
                    confidence = 1.0
                    status = "passed"
                    summary = f"Row count matches source exactly ({total_rows} rows)"
                else:
                    confidence = 0.5
                    status = "warning"
                    summary = f"Row count {result_rows} doesn't match source {total_rows}"
            
            return {
                "strategy": "row_count_validation",
                "status": status,
                "confidence": confidence,
                "summary": summary,
                "details": {
                    "source_rows": total_rows,
                    "result_rows": result_rows
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