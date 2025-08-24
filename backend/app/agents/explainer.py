import logging
from typing import Dict, Any, List, Optional
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class Explainer:
    """Generates natural language explanations for query results"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
    async def explain(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        counterfactual: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation of results"""
        
        # Generate base explanation
        base_explanation = await self._generate_base_explanation(
            question, sql, results
        )
        
        # Detect insights
        insights = await self._detect_insights(results)
        
        # Add counterfactual analysis if available
        counterfactual_analysis = None
        if counterfactual:
            counterfactual_analysis = await self._analyze_counterfactual(
                results, counterfactual
            )
        
        # Generate narrative
        narrative = await self._generate_narrative(
            base_explanation,
            insights,
            counterfactual_analysis
        )
        
        return {
            "summary": base_explanation["summary"],
            "narrative": narrative,
            "insights": insights,
            "counterfactual_analysis": counterfactual_analysis,
            "evidence_citations": self._generate_citations(results),
            "visualization_suggestions": self._suggest_visualizations(results, insights)
        }
    
    async def _generate_base_explanation(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate basic explanation of what the query found"""
        
        if not results:
            return {
                "summary": "No results found for your query.",
                "details": "The query executed successfully but returned no matching records."
            }
        
        prompt = f"""Explain the following query results in simple business terms.

Question: {question}
Number of results: {len(results)}
Sample results: {results[:3] if len(results) > 3 else results}

Provide:
1. A one-sentence summary
2. Key findings
3. Any notable patterns

Return as JSON:
{{
    "summary": "one sentence summary",
    "key_findings": ["finding 1", "finding 2"],
    "patterns": ["pattern 1", "pattern 2"]
}}
"""
        
        explanation = await self.llm_service.generate_explanation(prompt)
        return explanation
    
    async def _detect_insights(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect statistical insights from results"""
        insights = []
        
        if not results:
            return insights
        
        # Detect numeric columns
        numeric_columns = []
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                numeric_columns.append(key)
        
        # Calculate basic statistics for numeric columns
        for col in numeric_columns:
            values = [row[col] for row in results if row[col] is not None]
            if values:
                avg_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                
                # Detect outliers
                std_dev = (sum((x - avg_val) ** 2 for x in values) / len(values)) ** 0.5
                outliers = [v for v in values if abs(v - avg_val) > 2 * std_dev]
                
                if outliers:
                    insights.append({
                        "type": "outlier",
                        "description": f"Found {len(outliers)} outliers in {col}",
                        "evidence": {"column": col, "outlier_count": len(outliers)},
                        "confidence": 0.8
                    })
                
                # Detect trends (simplified)
                if len(values) > 5:
                    first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
                    second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                    
                    if second_half_avg > first_half_avg * 1.1:
                        insights.append({
                            "type": "trend",
                            "description": f"Increasing trend detected in {col}",
                            "evidence": {
                                "column": col,
                                "change": f"+{((second_half_avg/first_half_avg - 1) * 100):.1f}%"
                            },
                            "confidence": 0.7
                        })
        
        return insights
    
    async def _analyze_counterfactual(
        self,
        original_results: List[Dict[str, Any]],
        counterfactual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the difference between original and counterfactual results"""
        
        cf_results = counterfactual.get("results", [])
        
        analysis = {
            "condition_tested": counterfactual.get("condition_removed"),
            "original_count": len(original_results),
            "counterfactual_count": len(cf_results),
            "difference": len(original_results) - len(cf_results),
            "impact_percentage": 0
        }
        
        if len(original_results) > 0:
            analysis["impact_percentage"] = (
                (len(original_results) - len(cf_results)) / len(original_results) * 100
            )
        
        # Generate interpretation
        if analysis["impact_percentage"] > 20:
            analysis["interpretation"] = (
                f"The condition '{analysis['condition_tested']}' has a significant impact, "
                f"affecting {analysis['impact_percentage']:.1f}% of the results."
            )
        else:
            analysis["interpretation"] = (
                f"The condition '{analysis['condition_tested']}' has minimal impact, "
                f"affecting only {analysis['impact_percentage']:.1f}% of the results."
            )
        
        return analysis
    
    async def _generate_narrative(
        self,
        base_explanation: Dict[str, Any],
        insights: List[Dict[str, Any]],
        counterfactual_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a coherent narrative combining all analyses"""
        
        narrative_parts = []
        
        # Add summary
        narrative_parts.append(base_explanation.get("summary", ""))
        
        # Add key findings
        if base_explanation.get("key_findings"):
            findings = base_explanation["key_findings"][:3]
            narrative_parts.append(f"Key findings include: {', '.join(findings)}.")
        
        # Add insights
        if insights:
            insight_descriptions = [i["description"] for i in insights[:2]]
            narrative_parts.append(f"Notable patterns: {' Additionally, '.join(insight_descriptions)}.")
        
        # Add counterfactual
        if counterfactual_analysis:
            narrative_parts.append(counterfactual_analysis.get("interpretation", ""))
        
        return " ".join(narrative_parts)
    
    def _generate_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate evidence citations for key claims"""
        citations = []
        
        if results:
            # Cite top result
            citations.append({
                "claim": "Top result",
                "evidence": results[0],
                "row_index": 0
            })
            
            # Cite median if available
            if len(results) > 2:
                citations.append({
                    "claim": "Median result",
                    "evidence": results[len(results)//2],
                    "row_index": len(results)//2
                })
        
        return citations
    
    def _suggest_visualizations(
        self,
        results: List[Dict[str, Any]],
        insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest appropriate visualizations based on data characteristics"""
        suggestions = []
        
        if not results:
            return suggestions
        
        # Check data characteristics
        has_time = any("date" in k.lower() or "time" in k.lower() for k in results[0].keys())
        has_categories = any(isinstance(v, str) for v in results[0].values())
        has_numbers = any(isinstance(v, (int, float)) for v in results[0].values())
        
        if has_time and has_numbers:
            suggestions.append("line_chart")
        if has_categories and has_numbers:
            suggestions.append("bar_chart")
        if len(results) > 1 and has_numbers:
            suggestions.append("histogram")
        
        # Add suggestions based on insights
        for insight in insights:
            if insight["type"] == "trend":
                suggestions.append("trend_line")
            elif insight["type"] == "outlier":
                suggestions.append("box_plot")
        
        return list(set(suggestions))  # Remove duplicates