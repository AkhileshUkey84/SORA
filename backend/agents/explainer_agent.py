# Result narration
# agents/explainer_agent.py
"""
Natural language explanation agent that translates results into insights.
Makes data accessible to non-technical users through clear narratives.
"""

from typing import Dict, Any, List, Optional
from agents.base import BaseAgent, AgentContext, AgentResult
from services.llm_service import LLMService
import json
import re


class ExplainerAgent(BaseAgent):
    """
    Generates human-friendly explanations of query results.
    Focuses on business insights rather than technical details.
    """
    
    def __init__(self, llm_service: LLMService):
        super().__init__("ExplainerAgent")
        self.llm = llm_service
        self.explanation_templates = self._initialize_templates()
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Generates natural language explanation of results.
        Adapts tone and detail level based on context.
        """
        
        sql = kwargs.get("sql", "")
        results = kwargs.get("results", {})
        
        if not results or not results.get("data"):
            return AgentResult(
                success=True,
                data={
                    "narrative": "No data was found matching your query criteria.",
                    "key_findings": [],
                    "interpretation": "Consider broadening your search criteria."
                }
            )
        
        # Analyze results structure
        analysis = self._analyze_results(results)
        analysis["data"] = results.get("data", [])  # Add data to analysis for methods
        
        # Generate explanation based on query type
        if analysis["is_aggregation"]:
            narrative = await self._explain_aggregation(analysis, context)
        elif analysis["is_time_series"]:
            narrative = await self._explain_time_series(analysis, context)
        elif analysis["is_comparison"]:
            narrative = await self._explain_comparison(analysis, context)
        else:
            narrative = await self._explain_detail(analysis, context)
        
        # Extract key findings
        key_findings = self._extract_key_findings(analysis)
        
        # Generate interpretation with fallback
        interpretation = await self._generate_interpretation_with_fallback(analysis, context)
        
        return AgentResult(
            success=True,
            data={
                "narrative": narrative,
                "key_findings": key_findings,
                "interpretation": interpretation,
                "analysis_type": analysis["type"],
                "confidence": 0.9
            }
        )
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Templates for different explanation types"""
        return {
            "aggregation_intro": "Your analysis of {metric} by {dimension} reveals:",
            "time_series_intro": "Looking at {metric} over time shows:",
            "comparison_intro": "Comparing {entities} reveals:",
            "detail_intro": "Here are the {row_count} records matching your criteria:",
            
            "top_value": "The highest {metric} is {value} for {entity}",
            "trend_up": "{metric} shows an increasing trend of {percent}%",
            "trend_down": "{metric} shows a decreasing trend of {percent}%",
            "outlier": "{entity} stands out with {metric} of {value}",
            
            "summary": "In total, there are {count} {entity} with {aggregation}",
            "distribution": "The values range from {min} to {max} with an average of {avg}"
        }
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes result structure to determine explanation approach"""
        
        data = results.get("data", [])
        columns = results.get("columns", [])
        
        analysis = {
            "row_count": len(data),
            "column_count": len(columns),
            "columns": columns,
            "is_aggregation": False,
            "is_time_series": False,
            "is_comparison": False,
            "has_groups": False,
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "type": "detail"
        }
        
        if not data:
            return analysis
        
        # Analyze column types
        for col in columns:
            sample_values = [row.get(col) for row in data[:5] if row.get(col) is not None]
            
            if not sample_values:
                continue
            
            if all(isinstance(v, (int, float)) for v in sample_values):
                analysis["numeric_columns"].append(col)
            elif any(self._is_date_value(v) for v in sample_values):
                analysis["date_columns"].append(col)
            else:
                analysis["categorical_columns"].append(col)
        
        # Detect analysis type
        if any(col.lower() in ['count', 'sum', 'avg', 'average', 'total'] for col in columns):
            analysis["is_aggregation"] = True
            analysis["type"] = "aggregation"
        
        if analysis["date_columns"] and analysis["numeric_columns"]:
            analysis["is_time_series"] = True
            analysis["type"] = "time_series"
        
        if len(analysis["categorical_columns"]) > 1 and analysis["numeric_columns"]:
            analysis["is_comparison"] = True
            analysis["type"] = "comparison"
        
        return analysis
    
    async def _explain_aggregation(self, analysis: Dict[str, Any], 
                                 context: AgentContext) -> str:
        """Explains aggregated results with focus on key insights"""
        
        data = analysis.get("data", [])
        if not data:
            return "No aggregated results to explain."
        
        # Find the metric and dimension columns
        metric_col = analysis["numeric_columns"][0] if analysis["numeric_columns"] else None
        dimension_col = analysis["categorical_columns"][0] if analysis["categorical_columns"] else None
        
        if not metric_col:
            return "The query returned results, but no numeric values were found to analyze."
        
        # Build explanation parts
        parts = []
        
        # Introduction
        if dimension_col:
            parts.append(f"Your analysis of {metric_col} by {dimension_col} shows:")
        else:
            parts.append(f"Your analysis shows a {metric_col} of {data[0].get(metric_col, 'N/A')}.")
        
        if dimension_col and len(data) > 1:
            # Top values
            sorted_data = sorted(data, key=lambda x: x.get(metric_col, 0), reverse=True)
            
            # Highlight top 3
            parts.append(f"\n📊 Top performers:")
            for i, row in enumerate(sorted_data[:3], 1):
                parts.append(f"   {i}. {row.get(dimension_col)}: {self._format_number(row.get(metric_col))}")
            
            # Summary statistics
            values = [row.get(metric_col, 0) for row in data if row.get(metric_col) is not None]
            if values:
                total = sum(values)
                avg = total / len(values)
                parts.append(f"\n📈 Summary:")
                parts.append(f"   • Total: {self._format_number(total)}")
                parts.append(f"   • Average: {self._format_number(avg)}")
                parts.append(f"   • Range: {self._format_number(min(values))} to {self._format_number(max(values))}")
        
        return "\n".join(parts)
    
    async def _explain_time_series(self, analysis: Dict[str, Any], 
                                 context: AgentContext) -> str:
        """Explains time-based patterns and trends"""
        
        data = analysis.get("data", [])
        date_col = analysis["date_columns"][0] if analysis["date_columns"] else None
        metric_col = analysis["numeric_columns"][0] if analysis["numeric_columns"] else None
        
        if not date_col or not metric_col:
            return "Unable to identify time series pattern in the results."
        
        parts = [f"Analyzing {metric_col} over time:"]
        
        # Sort by date
        sorted_data = sorted(data, key=lambda x: x.get(date_col, ''))
        
        # Calculate trend
        if len(sorted_data) >= 2:
            first_value = sorted_data[0].get(metric_col, 0)
            last_value = sorted_data[-1].get(metric_col, 0)
            
            if first_value != 0:
                change_pct = ((last_value - first_value) / first_value) * 100
                trend = "increased" if change_pct > 0 else "decreased"
                
                parts.append(f"\n📈 Overall trend: {metric_col} {trend} by {abs(change_pct):.1f}%")
                parts.append(f"   • Starting value: {self._format_number(first_value)}")
                parts.append(f"   • Ending value: {self._format_number(last_value)}")
        
        # Identify peaks and valleys
        values = [row.get(metric_col, 0) for row in sorted_data]
        if values:
            max_idx = values.index(max(values))
            min_idx = values.index(min(values))
            
            parts.append(f"\n🔝 Peak: {self._format_number(values[max_idx])} on {sorted_data[max_idx].get(date_col)}")
            parts.append(f"🔻 Valley: {self._format_number(values[min_idx])} on {sorted_data[min_idx].get(date_col)}")
        
        return "\n".join(parts)
    
    async def _explain_comparison(self, analysis: Dict[str, Any], 
                                context: AgentContext) -> str:
        """Explains comparative analysis between groups"""
        
        data = analysis.get("data", [])
        
        # Find the main comparison dimension and metric
        categorical_cols = analysis["categorical_columns"]
        numeric_cols = analysis["numeric_columns"]
        
        if not categorical_cols or not numeric_cols:
            return self._explain_detail(analysis, context)
        
        comparison_col = categorical_cols[0]
        metric_col = numeric_cols[0]
        
        parts = [f"Comparing {metric_col} across different {comparison_col}:"]
        
        # Group by comparison dimension
        groups = {}
        for row in data:
            key = row.get(comparison_col)
            if key:
                groups.setdefault(key, []).append(row.get(metric_col, 0))
        
        # Calculate statistics per group
        group_stats = []
        for group, values in groups.items():
            if values:
                group_stats.append({
                    "name": group,
                    "avg": sum(values) / len(values),
                    "total": sum(values),
                    "count": len(values)
                })
        
        # Sort by average
        group_stats.sort(key=lambda x: x["avg"], reverse=True)
        
        # Explain differences
        if len(group_stats) >= 2:
            best = group_stats[0]
            worst = group_stats[-1]
            
            diff_pct = ((best["avg"] - worst["avg"]) / worst["avg"] * 100) if worst["avg"] != 0 else 0
            
            parts.append(f"\n🏆 Best performer: {best['name']}")
            parts.append(f"   Average {metric_col}: {self._format_number(best['avg'])}")
            
            parts.append(f"\n📊 Comparison insights:")
            parts.append(f"   • {best['name']} outperforms {worst['name']} by {diff_pct:.1f}%")
            parts.append(f"   • Spread between highest and lowest: {self._format_number(best['avg'] - worst['avg'])}")
        
        return "\n".join(parts)
    
    async def _explain_detail(self, analysis: Dict[str, Any], 
                            context: AgentContext) -> str:
        """Explains detailed record-level results"""
        
        row_count = analysis["row_count"]
        
        parts = [f"Found {row_count} record{'s' if row_count != 1 else ''} matching your criteria."]
        
        if row_count > 10:
            parts.append(f"\n📋 Showing first {min(row_count, 100)} results.")
            parts.append("Consider adding filters or aggregations for better insights.")
        elif row_count == 0:
            parts.append("\nNo matching records found. Try:")
            parts.append("• Broadening your search criteria")
            parts.append("• Checking for typos in filter values")
            parts.append("• Verifying the date range")
        else:
            # Describe the data briefly
            columns = analysis["columns"]
            parts.append(f"\nThe results include {len(columns)} columns: {', '.join(columns[:5])}")
            if len(columns) > 5:
                parts.append(f"...and {len(columns) - 5} more columns")
        
        return "\n".join(parts)
    
    def _extract_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Extracts bullet-point key findings from results"""
        
        findings = []
        data = analysis.get("data", [])
        
        if not data:
            return ["No data found matching the query criteria"]
        
        # Based on analysis type
        if analysis["is_aggregation"] and analysis["numeric_columns"]:
            metric_col = analysis["numeric_columns"][0]
            values = [row.get(metric_col, 0) for row in data if row.get(metric_col) is not None]
            
            if values:
                findings.append(f"Total {metric_col}: {self._format_number(sum(values))}")
                findings.append(f"Average {metric_col}: {self._format_number(sum(values)/len(values))}")
                
                if len(data) > 1 and analysis["categorical_columns"]:
                    top_row = max(data, key=lambda x: x.get(metric_col, 0))
                    findings.append(f"Highest: {top_row.get(analysis['categorical_columns'][0])}")
        
        elif analysis["is_time_series"]:
            findings.append(f"Time period analyzed: {len(data)} data points")
            # Add trend finding if calculated
        
        else:
            findings.append(f"Total records: {len(data)}")
            if analysis["numeric_columns"]:
                # Add range for first numeric column
                col = analysis["numeric_columns"][0]
                values = [row.get(col) for row in data if row.get(col) is not None]
                if values:
                    findings.append(f"{col} range: {min(values)} to {max(values)}")
        
        return findings[:5]  # Limit to 5 findings
    
    async def _generate_interpretation_with_fallback(self, analysis: Dict[str, Any], 
                                                   context: AgentContext) -> str:
        """Generates business interpretation with fallback"""
        try:
            return await self._generate_interpretation(analysis, context)
        except Exception as e:
            self.logger.warning(f"Failed to generate LLM interpretation, using fallback: {e}")
            return self._generate_interpretation_fallback(analysis, context)
    
    def _generate_interpretation_fallback(self, analysis: Dict[str, Any], 
                                        context: AgentContext) -> str:
        """Fallback interpretation without LLM"""
        interpretations = {
            "aggregation": "This aggregated view helps identify top performers and patterns in your data.",
            "time_series": "The time series analysis reveals trends and patterns over the selected period.",
            "comparison": "This comparison highlights differences between groups and can guide strategic decisions.",
            "detail": "These detailed records provide granular insights into individual data points."
        }
        
        base_interpretation = interpretations.get(analysis["type"], 
                                                "The query results provide insights into your data.")
        
        # Add context-specific interpretation
        if context.metadata.get("domain") == "sales":
            base_interpretation += " Focus on high-value segments for revenue optimization."
        elif context.metadata.get("domain") == "operations":
            base_interpretation += " Look for efficiency improvements and bottlenecks."
        else:
            base_interpretation += " Consider what patterns or trends might be relevant to your business goals."
        
        return base_interpretation
    
    async def _generate_interpretation(self, analysis: Dict[str, Any], 
                                     context: AgentContext) -> str:
        """Generates business interpretation of the results"""
        
        interpretations = {
            "aggregation": "This aggregated view helps identify top performers and patterns in your data.",
            "time_series": "The time series analysis reveals trends and patterns over the selected period.",
            "comparison": "This comparison highlights differences between groups and can guide strategic decisions.",
            "detail": "These detailed records provide granular insights into individual data points."
        }
        
        base_interpretation = interpretations.get(analysis["type"], "")
        
        # Add context-specific interpretation
        if context.metadata.get("domain") == "sales":
            base_interpretation += " Focus on high-value segments for revenue optimization."
        elif context.metadata.get("domain") == "operations":
            base_interpretation += " Look for efficiency improvements and bottlenecks."
        
        return base_interpretation
    
    def _format_number(self, value: Any) -> str:
        """Formats numbers for readability"""
        if value is None:
            return "N/A"
        
        try:
            num = float(value)
            
            # Large numbers
            if abs(num) >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:.1f}K"
            
            # Decimals
            elif abs(num) < 10 and num != int(num):
                return f"{num:.2f}"
            else:
                return f"{int(num):,}"
                
        except (ValueError, TypeError):
            return str(value)
    
    def _is_date_value(self, value: Any) -> bool:
        """Checks if a value represents a date"""
        if not isinstance(value, str):
            return False
        
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',
            r'^\d{1,2}/\d{1,2}/\d{2,4}',
            r'^\d{1,2}-\w{3}-\d{2,4}'
        ]
        
        return any(re.match(pattern, value) for pattern in date_patterns)