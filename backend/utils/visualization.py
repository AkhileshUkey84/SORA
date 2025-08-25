# Chart config generation
# utils/visualization.py
"""Chart configuration generation utilities"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import structlog


logger = structlog.get_logger()


class ChartGenerator:
    """
    Intelligent chart configuration generator.
    Automatically selects appropriate visualizations based on data.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="ChartGenerator")
    
    def generate_chart_config(
        self,
        data: List[Dict[str, Any]],
        query_context: Dict[str, Any],
        insights: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate smart chart configuration based on data shape and context.
        Returns None if data isn't suitable for visualization.
        """
        
        if not data or len(data) < 2:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        # Analyze data types
        analysis = self._analyze_dataframe(df)
        
        # Determine best chart type
        chart_type = self._select_chart_type(analysis, query_context)
        
        if not chart_type:
            return None
        
        # Generate configuration
        config = self._generate_config(chart_type, analysis, df)
        
        # Add insights if available
        if insights:
            config["insights"] = self._format_insights_for_chart(insights, chart_type)
        
        return config
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame structure and content"""
        
        analysis = {
            "row_count": len(df),
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "high_cardinality_columns": []
        }
        
        for column in df.columns:
            dtype = df[column].dtype
            unique_count = df[column].nunique()
            
            # Classify columns
            if pd.api.types.is_numeric_dtype(dtype):
                analysis["numeric_columns"].append({
                    "name": column,
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "has_negative": (df[column] < 0).any()
                })
            
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                analysis["datetime_columns"].append({
                    "name": column,
                    "min": df[column].min(),
                    "max": df[column].max()
                })
            
            else:
                # Categorical
                if unique_count > 20:
                    analysis["high_cardinality_columns"].append(column)
                else:
                    analysis["categorical_columns"].append({
                        "name": column,
                        "unique_count": unique_count,
                        "top_values": df[column].value_counts().head(5).index.tolist()
                    })
        
        return analysis
    
    def _select_chart_type(
        self,
        analysis: Dict[str, Any],
        query_context: Dict[str, Any]
    ) -> Optional[str]:
        """Select appropriate chart type based on data and context"""
        
        numeric_count = len(analysis["numeric_columns"])
        categorical_count = len(analysis["categorical_columns"])
        datetime_count = len(analysis["datetime_columns"])
        row_count = analysis["row_count"]
        
        # Query intent hints
        query_lower = query_context.get("query", "").lower()
        
        # Decision tree for chart selection
        
        # Time series
        if datetime_count > 0 and numeric_count > 0:
            if "trend" in query_lower or "over time" in query_lower:
                return "line"
            elif row_count > 50:
                return "line"
        
        # Correlations
        if numeric_count >= 2:
            if "correlation" in query_lower or "vs" in query_lower:
                return "scatter"
            elif row_count < 100:
                return "scatter"
        
        # Distributions
        if numeric_count == 1 and categorical_count == 0:
            return "histogram"
        
        # Comparisons
        if categorical_count == 1 and numeric_count >= 1:
            cat_info = analysis["categorical_columns"][0]
            if cat_info["unique_count"] <= 10:
                return "bar"
            elif cat_info["unique_count"] <= 5 and "percentage" in query_lower:
                return "pie"
        
        # Multi-category comparison
        if categorical_count >= 2 and numeric_count >= 1:
            if row_count <= 50:
                return "heatmap"
        
        # Default to bar chart for aggregations
        if numeric_count >= 1 and row_count <= 20:
            return "bar"
        
        return None
    
    def _generate_config(
        self,
        chart_type: str,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate specific configuration for chart type"""
        
        config = {
            "type": chart_type,
            "title": self._generate_title(chart_type, analysis)
        }
        
        if chart_type == "bar":
            config.update(self._config_bar_chart(analysis, df))
        
        elif chart_type == "line":
            config.update(self._config_line_chart(analysis, df))
        
        elif chart_type == "scatter":
            config.update(self._config_scatter_chart(analysis, df))
        
        elif chart_type == "pie":
            config.update(self._config_pie_chart(analysis, df))
        
        elif chart_type == "histogram":
            config.update(self._config_histogram(analysis, df))
        
        elif chart_type == "heatmap":
            config.update(self._config_heatmap(analysis, df))
        
        # Add common enhancements
        config["color_scheme"] = "viridis"
        config["responsive"] = True
        
        return config
    
    def _config_bar_chart(
        self,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Configure bar chart"""
        
        # Select x and y axes
        if analysis["categorical_columns"]:
            x_col = analysis["categorical_columns"][0]["name"]
        else:
            # Use first column
            x_col = df.columns[0]
        
        if analysis["numeric_columns"]:
            y_col = analysis["numeric_columns"][0]["name"]
        else:
            # Count occurrences
            y_col = "count"
            df = df.groupby(x_col).size().reset_index(name='count')
        
        # Sort by value
        df_sorted = df.sort_values(y_col, ascending=False)
        
        return {
            "x": x_col,
            "y": y_col,
            "x_label": self._format_label(x_col),
            "y_label": self._format_label(y_col),
            "sort": "descending",
            "limit": min(10, len(df_sorted))  # Top 10
        }
    
    def _config_line_chart(
        self,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Configure line chart"""
        
        # X axis: datetime or sequential
        if analysis["datetime_columns"]:
            x_col = analysis["datetime_columns"][0]["name"]
        else:
            # Use index
            x_col = "index"
            df[x_col] = df.index
        
        # Y axis: first numeric column
        y_col = analysis["numeric_columns"][0]["name"]
        
        return {
            "x": x_col,
            "y": y_col,
            "x_label": "Time" if analysis["datetime_columns"] else "Sequence",
            "y_label": self._format_label(y_col),
            "show_trend": True,
            "show_points": len(df) < 50
        }
    
    def _config_scatter_chart(
        self,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Configure scatter plot"""
        
        # Use first two numeric columns
        x_col = analysis["numeric_columns"][0]["name"]
        y_col = analysis["numeric_columns"][1]["name"]
        
        config = {
            "x": x_col,
            "y": y_col,
            "x_label": self._format_label(x_col),
            "y_label": self._format_label(y_col),
            "show_correlation": True
        }
        
        # Add color dimension if available
        if analysis["categorical_columns"]:
            config["color"] = analysis["categorical_columns"][0]["name"]
        
        # Add size dimension if more numeric columns
        if len(analysis["numeric_columns"]) > 2:
            config["size"] = analysis["numeric_columns"][2]["name"]
        
        return config
    
    def _config_pie_chart(
        self,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Configure pie chart"""
        
        # Category and value columns
        cat_col = analysis["categorical_columns"][0]["name"]
        val_col = analysis["numeric_columns"][0]["name"]
        
        return {
            "labels": cat_col,
            "values": val_col,
            "show_percentages": True,
            "limit": 5  # Top 5 + "Others"
        }
    
    def _config_histogram(
        self,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Configure histogram"""
        
        col = analysis["numeric_columns"][0]["name"]
        
        return {
            "x": col,
            "x_label": self._format_label(col),
            "y_label": "Frequency",
            "bins": min(20, int(np.sqrt(len(df)))),
            "show_distribution": True
        }
    
    def _config_heatmap(
        self,
        analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Configure heatmap"""
        
        # Use first two categorical columns and a numeric column
        x_col = analysis["categorical_columns"][0]["name"]
        y_col = analysis["categorical_columns"][1]["name"]
        z_col = analysis["numeric_columns"][0]["name"]
        
        return {
            "x": x_col,
            "y": y_col,
            "z": z_col,
            "color_scale": "RdYlBu",
            "show_values": len(df) < 100
        }
    
    def _generate_title(self, chart_type: str, analysis: Dict[str, Any]) -> str:
        """Generate descriptive chart title"""
        
        if chart_type == "bar" and analysis["categorical_columns"]:
            return f"{analysis['numeric_columns'][0]['name']} by {analysis['categorical_columns'][0]['name']}"
        
        elif chart_type == "line" and analysis["datetime_columns"]:
            return f"{analysis['numeric_columns'][0]['name']} Over Time"
        
        elif chart_type == "scatter":
            return f"{analysis['numeric_columns'][1]['name']} vs {analysis['numeric_columns'][0]['name']}"
        
        elif chart_type == "pie":
            return f"Distribution of {analysis['numeric_columns'][0]['name']}"
        
        elif chart_type == "histogram":
            return f"Distribution of {analysis['numeric_columns'][0]['name']}"
        
        else:
            return "Data Visualization"
    
    def _format_label(self, column_name: str) -> str:
        """Format column name as human-readable label"""
        
        # Replace underscores with spaces
        label = column_name.replace('_', ' ')
        
        # Title case
        label = label.title()
        
        # Common replacements
        replacements = {
            'Id': 'ID',
            'Url': 'URL',
            'Api': 'API',
            'Csv': 'CSV'
        }
        
        for old, new in replacements.items():
            label = label.replace(old, new)
        
        return label
    
    def _format_insights_for_chart(
        self,
        insights: List[Dict[str, Any]],
        chart_type: str
    ) -> List[str]:
        """Format insights as annotations for the chart"""
        
        chart_insights = []
        
        for insight in insights[:3]:  # Limit to top 3
            if insight.get("visual_hint") == chart_type:
                # Direct match
                chart_insights.append(insight["title"])
            elif chart_type == "line" and insight.get("type") == "trend":
                # Trend insights for line charts
                chart_insights.append(insight["description"][:100])
            elif chart_type == "bar" and insight.get("type") == "comparison":
                # Comparison insights for bar charts
                chart_insights.append(insight["title"])
        
        return chart_insights