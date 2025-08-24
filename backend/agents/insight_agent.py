# Auto-discovery of patterns
# agents/insight_agent.py
"""
Proactive insight discovery agent that automatically identifies patterns,
trends, and anomalies in query results. Key differentiator for demos.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from enum import Enum
from pydantic import BaseModel
from agents.base import BaseAgent, AgentContext, AgentResult


class InsightType(str, Enum):
    TREND = "trend"
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"


class Insight(BaseModel):
    """Structured insight with actionable information"""
    type: InsightType
    title: str
    description: str
    importance: float  # 0-1 score
    visual_hint: Optional[str] = None  # Suggested visualization
    follow_up_query: Optional[str] = None  # Suggested next question
    supporting_data: Dict[str, Any] = {}


class InsightDiscoveryAgent(BaseAgent):
    """
    Automatically discovers interesting patterns in data.
    Goes beyond answering questions to proactively surface insights.
    """
    
    def __init__(self):
        super().__init__("InsightDiscoveryAgent")
        self.min_rows_for_insights = 10
        self.max_insights = 5
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Analyzes query results to discover meaningful insights.
        Prioritizes actionable, business-relevant findings.
        """
        
        results = kwargs.get("results", {})
        if not results or not results.get("data"):
            return AgentResult(
                success=True,
                data={"insights": [], "summary": "No data available for analysis"}
            )
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results["data"])
        
        if len(df) < self.min_rows_for_insights:
            return AgentResult(
                success=True,
                data={
                    "insights": [],
                    "summary": f"Dataset too small ({len(df)} rows) for meaningful insights"
                }
            )
        
        insights = []
        
        # Detect column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Run different insight detectors based on data types
        if numeric_cols:
            insights.extend(await self._find_numeric_insights(df, numeric_cols))
            
            if len(numeric_cols) > 1:
                insights.extend(await self._find_correlations(df, numeric_cols))
        
        if categorical_cols:
            insights.extend(await self._find_categorical_insights(df, categorical_cols))
        
        if datetime_cols and numeric_cols:
            insights.extend(await self._find_temporal_insights(df, datetime_cols[0], numeric_cols))
        
        if numeric_cols and categorical_cols:
            insights.extend(await self._find_segmentation_insights(df, categorical_cols[0], numeric_cols[0]))
        
        # Detect anomalies across all suitable columns
        insights.extend(await self._find_anomalies(df))
        
        # Sort by importance and limit
        insights.sort(key=lambda x: x.importance, reverse=True)
        top_insights = insights[:self.max_insights]
        
        # Generate summary
        summary = self._generate_insight_summary(top_insights, len(df))
        
        return AgentResult(
            success=True,
            data={
                "insights": [i.dict() for i in top_insights],
                "summary": summary,
                "total_insights_found": len(insights)
            },
            confidence=0.85
        )
    
    async def _find_numeric_insights(self, df: pd.DataFrame, 
                                   columns: List[str]) -> List[Insight]:
        """Discovers insights in numeric columns"""
        insights = []
        
        for col in columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) < 5:
                continue
            
            # Statistical summary
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()
            
            # Skewness detection
            if abs(mean_val - median_val) > 0.2 * std_val:
                skew_direction = "right" if mean_val > median_val else "left"
                insights.append(Insight(
                    type=InsightType.DISTRIBUTION,
                    title=f"Skewed Distribution in {col}",
                    description=f"The {col} distribution is {skew_direction}-skewed. "
                               f"Mean ({mean_val:.2f}) differs significantly from median ({median_val:.2f}). "
                               f"This suggests presence of {skew_direction}-side outliers.",
                    importance=0.7,
                    visual_hint="histogram",
                    follow_up_query=f"Show me the top 10 highest {col} values",
                    supporting_data={
                        "mean": mean_val,
                        "median": median_val,
                        "skewness": float(stats.skew(series))
                    }
                ))
            
            # Outlier detection using IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(series)) * 100
                insights.append(Insight(
                    type=InsightType.OUTLIER,
                    title=f"Outliers Detected in {col}",
                    description=f"Found {len(outliers)} outliers ({outlier_pct:.1f}% of data) in {col}. "
                               f"These values fall outside the range [{q1 - 1.5 * iqr:.2f}, {q3 + 1.5 * iqr:.2f}].",
                    importance=0.8 if outlier_pct > 5 else 0.6,
                    visual_hint="box_plot",
                    follow_up_query=f"Show me rows where {col} < {q1 - 1.5 * iqr:.2f} OR {col} > {q3 + 1.5 * iqr:.2f}",
                    supporting_data={
                        "outlier_count": len(outliers),
                        "outlier_percentage": outlier_pct,
                        "outlier_values": outliers.tolist()[:5]  # Sample
                    }
                ))
        
        return insights
    
    async def _find_correlations(self, df: pd.DataFrame, 
                               columns: List[str]) -> List[Insight]:
        """Finds significant correlations between numeric columns"""
        insights = []
        
        if len(columns) < 2:
            return insights
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Find strong correlations (excluding diagonal)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    relationship = "positive" if corr_value > 0 else "negative"
                    insights.append(Insight(
                        type=InsightType.CORRELATION,
                        title=f"Strong {relationship.title()} Correlation",
                        description=f"{columns[i]} and {columns[j]} show a strong {relationship} "
                                   f"correlation (r={corr_value:.3f}). "
                                   f"{'Higher' if corr_value > 0 else 'Lower'} {columns[i]} values "
                                   f"tend to coincide with {'higher' if corr_value > 0 else 'lower'} {columns[j]} values.",
                        importance=abs(corr_value),
                        visual_hint="scatter_plot",
                        follow_up_query=f"Show me {columns[i]} vs {columns[j]} grouped by top categories",
                        supporting_data={
                            "correlation": corr_value,
                            "column_1": columns[i],
                            "column_2": columns[j]
                        }
                    ))
        
        return insights
    
    async def _find_categorical_insights(self, df: pd.DataFrame, 
                                       columns: List[str]) -> List[Insight]:
        """Analyzes categorical columns for interesting patterns"""
        insights = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            value_counts = df[col].value_counts()
            
            # Concentration insight
            top_n = 3
            if len(value_counts) > top_n:
                top_n_pct = (value_counts.head(top_n).sum() / len(df)) * 100
                
                if top_n_pct > 60:  # High concentration
                    insights.append(Insight(
                        type=InsightType.DISTRIBUTION,
                        title=f"High Concentration in {col}",
                        description=f"The top {top_n} {col} values account for {top_n_pct:.1f}% of all data. "
                                   f"The most common value '{value_counts.index[0]}' appears {value_counts.iloc[0]} times ({value_counts.iloc[0]/len(df)*100:.1f}%).",
                        importance=0.7,
                        visual_hint="bar_chart",
                        follow_up_query=f"Show me breakdown by {col} for top {top_n} categories",
                        supporting_data={
                            "top_values": value_counts.head(top_n).to_dict(),
                            "concentration_pct": top_n_pct
                        }
                    ))
            
            # Long tail detection
            if len(value_counts) > 20:
                tail_values = value_counts[value_counts == 1].count()
                if tail_values > len(value_counts) * 0.5:
                    insights.append(Insight(
                        type=InsightType.DISTRIBUTION,
                        title=f"Long Tail Distribution in {col}",
                        description=f"{col} has {len(value_counts)} unique values with {tail_values} appearing only once. "
                                   f"This suggests a long-tail distribution with many rare categories.",
                        importance=0.6,
                        visual_hint="pareto_chart",
                        follow_up_query=f"Show me {col} values that appear more than 5 times",
                        supporting_data={
                            "unique_values": len(value_counts),
                            "singleton_values": tail_values
                        }
                    ))
        
        return insights
    
    async def _find_temporal_insights(self, df: pd.DataFrame, date_col: str, 
                                    value_cols: List[str]) -> List[Insight]:
        """Discovers time-based patterns and trends"""
        insights = []
        
        if date_col not in df.columns or not value_cols:
            return insights
        
        # Ensure datetime type
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df_clean = df.dropna(subset=[date_col])
        
        for val_col in value_cols[:2]:  # Limit to avoid too many insights
            if val_col not in df_clean.columns:
                continue
            
            # Sort by date
            df_sorted = df_clean.sort_values(date_col)
            
            # Simple trend detection using linear regression
            if len(df_sorted) > 10:
                x = np.arange(len(df_sorted))
                y = df_sorted[val_col].fillna(method='ffill').values
                
                # Calculate trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                if p_value < 0.05:  # Statistically significant
                    trend_direction = "increasing" if slope > 0 else "decreasing"
                    pct_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
                    
                    insights.append(Insight(
                        type=InsightType.TREND,
                        title=f"{trend_direction.title()} Trend in {val_col}",
                        description=f"{val_col} shows a statistically significant {trend_direction} trend over time. "
                                   f"Values changed by {abs(pct_change):.1f}% from start to end of the period.",
                        importance=abs(r_value),
                        visual_hint="line_chart",
                        follow_up_query=f"Show me monthly average {val_col} over time",
                        supporting_data={
                            "slope": slope,
                            "r_squared": r_value ** 2,
                            "p_value": p_value,
                            "percent_change": pct_change
                        }
                    ))
        
        return insights
    
    async def _find_segmentation_insights(self, df: pd.DataFrame, 
                                        cat_col: str, val_col: str) -> List[Insight]:
        """Finds insights by segmenting data by categories"""
        insights = []
        
        if cat_col not in df.columns or val_col not in df.columns:
            return insights
        
        # Group statistics
        grouped = df.groupby(cat_col)[val_col].agg(['mean', 'count', 'std'])
        
        if len(grouped) < 2:
            return insights
        
        # Find significant differences
        overall_mean = df[val_col].mean()
        
        for category, stats in grouped.iterrows():
            if stats['count'] < 5:  # Skip small groups
                continue
            
            # Calculate z-score for difference from overall mean
            if stats['std'] > 0:
                z_score = (stats['mean'] - overall_mean) / (stats['std'] / np.sqrt(stats['count']))
                
                if abs(z_score) > 2:  # Significant difference
                    diff_pct = ((stats['mean'] - overall_mean) / overall_mean) * 100
                    higher_lower = "higher" if stats['mean'] > overall_mean else "lower"
                    
                    insights.append(Insight(
                        type=InsightType.COMPARISON,
                        title=f"{category} Shows Significant Difference",
                        description=f"{cat_col}='{category}' has {abs(diff_pct):.1f}% {higher_lower} "
                                   f"{val_col} ({stats['mean']:.2f}) compared to overall average ({overall_mean:.2f}).",
                        importance=min(abs(z_score) / 5, 0.9),
                        visual_hint="grouped_bar_chart",
                        follow_up_query=f"Show me detailed breakdown for {cat_col} = '{category}'",
                        supporting_data={
                            "category": category,
                            "category_mean": stats['mean'],
                            "overall_mean": overall_mean,
                            "z_score": z_score,
                            "sample_size": stats['count']
                        }
                    ))
        
        return insights
    
    async def _find_anomalies(self, df: pd.DataFrame) -> List[Insight]:
        """Detects anomalous patterns across the dataset"""
        insights = []
        
        # Check for duplicate rows
        duplicates = df.duplicated()
        if duplicates.sum() > 0:
            dup_pct = (duplicates.sum() / len(df)) * 100
            insights.append(Insight(
                type=InsightType.ANOMALY,
                title="Duplicate Rows Detected",
                description=f"Found {duplicates.sum()} duplicate rows ({dup_pct:.1f}% of data). "
                           "This might indicate data quality issues or legitimate repeated entries.",
                importance=0.6 if dup_pct > 10 else 0.4,
                visual_hint=None,
                follow_up_query="Show me duplicate rows grouped by count",
                supporting_data={
                    "duplicate_count": int(duplicates.sum()),
                    "duplicate_percentage": dup_pct
                }
            ))
        
        # Check for missing values
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        
        if len(cols_with_missing) > 0:
            worst_col = cols_with_missing.idxmax()
            worst_pct = (cols_with_missing[worst_col] / len(df)) * 100
            
            if worst_pct > 20:
                insights.append(Insight(
                    type=InsightType.ANOMALY,
                    title=f"High Missing Values in {worst_col}",
                    description=f"{worst_col} has {worst_pct:.1f}% missing values. "
                               "Consider the impact on analysis accuracy.",
                    importance=0.7 if worst_pct > 50 else 0.5,
                    visual_hint="missing_data_matrix",
                    follow_up_query=f"Show me rows where {worst_col} is not null",
                    supporting_data={
                        "missing_columns": cols_with_missing.to_dict(),
                        "worst_column": worst_col,
                        "worst_percentage": worst_pct
                    }
                ))
        
        return insights
    
    def _generate_insight_summary(self, insights: List[Insight], row_count: int) -> str:
        """Creates a natural language summary of discovered insights"""
        
        if not insights:
            return f"Analyzed {row_count:,} rows. No significant patterns detected."
        
        summary_parts = [f"Analyzed {row_count:,} rows and discovered {len(insights)} key insights:"]
        
        # Group by type
        by_type = {}
        for insight in insights:
            by_type.setdefault(insight.type, []).append(insight)
        
        type_descriptions = {
            InsightType.TREND: "trends",
            InsightType.OUTLIER: "outliers",
            InsightType.CORRELATION: "correlations",
            InsightType.DISTRIBUTION: "distribution patterns",
            InsightType.COMPARISON: "significant differences",
            InsightType.ANOMALY: "anomalies"
        }
        
        for itype, items in by_type.items():
            desc = type_descriptions.get(itype, "patterns")
            summary_parts.append(f"• {len(items)} {desc}")
        
        # Add most important insight
        if insights:
            top_insight = insights[0]
            summary_parts.append(f"\nMost notable: {top_insight.title}")
        
        return " ".join(summary_parts)