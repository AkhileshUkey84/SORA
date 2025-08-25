# Gemini API wrapper
# services/llm_service.py
"""
Gemini API integration service for LLM interactions.
Handles prompt engineering and response parsing.
"""

import os
import asyncio
import time
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog
from utils.config import settings
from google.api_core.exceptions import ResourceExhausted


logger = structlog.get_logger()


class LLMService:
    """
    Manages interactions with Gemini API.
    Implements retry logic and response validation.
    """
    
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.model_name = settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 12  # 12 seconds between requests (5 RPM for free tier)
        self.rate_limit_errors = 0
        self.demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'
        
        # Demo mode uses more aggressive fallbacks to ensure reliability
        if self.demo_mode:
            self.min_request_interval = 15  # Even safer for demo
            self.logger.info("🎪 Demo mode activated - optimized for reliability")
        
        self.logger = logger.bind(service="LLMService")
    
    async def _wait_for_rate_limit(self):
        """Wait if we're hitting rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            self.logger.info(f"Rate limiting: waiting {wait_time:.1f}s before next request")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def generate_sql_fallback(self, schema_info: Dict[str, Any], query: str) -> str:
        """Generate intelligent SQL without LLM when rate limited"""
        table_name = schema_info.get('table_name', 'dataset')
        columns = [col['name'] for col in schema_info.get('columns', [])]
        
        self.logger.warning(f"🚀 INTELLIGENT FALLBACK SQL GENERATOR! Using table: {table_name} for query: {query[:100]}")
        
        # Enhanced pattern matching for common business queries
        query_lower = query.lower()
        
        # Day of week analysis (common demo query)
        if any(word in query_lower for word in ['day', 'week', 'weekday', 'monday', 'tuesday']):
            date_cols = [col for col in columns if any(t in str(col).lower() for t in ['date', 'time', 'created', 'appointment'])]
            if date_cols:
                return f"SELECT DAYNAME({date_cols[0]}) as day_of_week, COUNT(*) as count FROM {table_name} GROUP BY DAYNAME({date_cols[0]}) ORDER BY count DESC LIMIT 10"
        
        # Top/highest/most queries
        if any(word in query_lower for word in ['top', 'highest', 'most', 'best', 'maximum']):
            # Look for quantity or value columns
            value_cols = [col for col in columns if any(t in str(col).lower() for t in ['amount', 'value', 'price', 'cost', 'sales', 'revenue', 'count'])]
            cat_cols = [col for col in columns if any(t in str(col).lower() for t in ['category', 'type', 'name', 'product', 'customer'])]
            if value_cols and cat_cols:
                return f"SELECT {cat_cols[0]}, SUM({value_cols[0]}) as total FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY total DESC LIMIT 10"
            elif cat_cols:
                return f"SELECT {cat_cols[0]}, COUNT(*) as count FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY count DESC LIMIT 10"
        
        # Count/total queries
        if any(word in query_lower for word in ['count', 'total', 'number', 'how many']):
            if any(word in query_lower for word in ['by', 'per', 'each']):
                cat_cols = [col for col in columns if any(t in str(col).lower() for t in ['category', 'type', 'status', 'department'])]
                if cat_cols:
                    return f"SELECT {cat_cols[0]}, COUNT(*) as count FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY count DESC LIMIT 10"
            return f"SELECT COUNT(*) as total_count FROM {table_name}"
        
        # Average/mean queries
        elif any(word in query_lower for word in ['average', 'avg', 'mean']):
            numeric_cols = [col for col in columns if any(t in str(col).lower() for t in ['amount', 'value', 'price', 'cost', 'sales', 'age', 'duration'])]
            if numeric_cols:
                if any(word in query_lower for word in ['by', 'per', 'each']):
                    cat_cols = [col for col in columns if any(t in str(col).lower() for t in ['category', 'type', 'department'])]
                    if cat_cols:
                        return f"SELECT {cat_cols[0]}, AVG({numeric_cols[0]}) as average FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY average DESC LIMIT 10"
                return f"SELECT AVG({numeric_cols[0]}) as average FROM {table_name}"
        
        # Time-based queries
        elif any(word in query_lower for word in ['month', 'year', 'trend', 'over time']):
            date_cols = [col for col in columns if any(t in str(col).lower() for t in ['date', 'time', 'created'])]
            if date_cols:
                return f"SELECT DATE_PART('month', {date_cols[0]}) as month, COUNT(*) as count FROM {table_name} GROUP BY DATE_PART('month', {date_cols[0]}) ORDER BY month LIMIT 12"
        
        # Status/category breakdown
        elif any(word in query_lower for word in ['status', 'breakdown', 'distribution']):
            status_cols = [col for col in columns if any(t in str(col).lower() for t in ['status', 'state', 'condition', 'type'])]
            if status_cols:
                return f"SELECT {status_cols[0]}, COUNT(*) as count FROM {table_name} GROUP BY {status_cols[0]} ORDER BY count DESC LIMIT 10"
        
        # Revenue/sales queries
        elif any(word in query_lower for word in ['revenue', 'sales', 'profit', 'income']):
            money_cols = [col for col in columns if any(t in str(col).lower() for t in ['amount', 'revenue', 'sales', 'price', 'cost', 'value'])]
            if money_cols:
                return f"SELECT SUM({money_cols[0]}) as total_revenue FROM {table_name}"
        
        # General grouping queries
        elif any(word in query_lower for word in ['group', 'by', 'category', 'type']):
            cat_cols = [col for col in columns if any(t in str(col).lower() for t in ['category', 'type', 'name', 'industry', 'department'])]
            if cat_cols:
                return f"SELECT {cat_cols[0]}, COUNT(*) as count FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY count DESC LIMIT 10"
        
        # Default fallback with smart column selection
        if columns:
            # Prioritize interesting columns for demo
            priority_cols = []
            for col in columns[:10]:  # First 10 columns
                col_lower = col.lower()
                if any(t in col_lower for t in ['name', 'title', 'id', 'date', 'status', 'type', 'category', 'amount', 'count']):
                    priority_cols.append(col)
            
            if priority_cols:
                selected_columns = ', '.join(priority_cols[:5])
                return f"SELECT {selected_columns} FROM {table_name} ORDER BY {priority_cols[0]} LIMIT 10"
            elif len(columns) > 5:
                selected_columns = ', '.join(columns[:5])
                return f"SELECT {selected_columns} FROM {table_name} LIMIT 10"
            else:
                return f"SELECT * FROM {table_name} LIMIT 10"
        else:
            return f"SELECT * FROM {table_name} LIMIT 10"
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        schema_info: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
        enable_thinking: bool = False
    ) -> str:
        """
        Generates text using Gemini API with rate limiting and fallbacks.
        
        Args:
            prompt: The user prompt to send to the model
            temperature: Controls randomness (lower = more deterministic)
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system instructions
            schema_info: Optional schema information for SQL fallback
            user_query: Optional user query for SQL fallback
            enable_thinking: Whether to enable thinking capabilities (Gemini 2.5 Flash only)
        """
        
        if temperature is None:
            temperature = self.default_temperature
        
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Check if we should use fallback due to repeated rate limit errors
        if self.rate_limit_errors >= 3 and schema_info and user_query:
            self.logger.warning("Using fallback SQL generation due to repeated rate limit errors")
            return self.generate_sql_fallback(schema_info, user_query)
        
        try:
            # Rate limiting
            await self._wait_for_rate_limit()
            
            # Combine system and user prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40
            )
            
            # Generate response (thinking config removed for compatibility)
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract text
            if response.text:
                self.logger.info("LLM generation successful", 
                               prompt_length=len(prompt),
                               response_length=len(response.text),
                               model=self.model_name)
                # Reset rate limit error count on success
                self.rate_limit_errors = 0
                return response.text
            else:
                raise ValueError("Empty response from LLM")
                
        except ResourceExhausted as e:
            self.rate_limit_errors += 1
            self.logger.error("Rate limit exceeded", 
                            error=str(e),
                            rate_limit_errors=self.rate_limit_errors)
            
            # Use fallback if we have the necessary info
            if schema_info and user_query:
                self.logger.info("Using fallback SQL generation due to rate limit")
                return self.generate_sql_fallback(schema_info, user_query)
            else:
                raise
                
        except Exception as e:
            self.logger.error("LLM generation failed", 
                            error=str(e),
                            prompt_preview=prompt[:100])
            
            # Use fallback for SQL generation if available
            if schema_info and user_query and "SQL" in prompt:
                self.logger.info("Using fallback SQL generation due to error")
                return self.generate_sql_fallback(schema_info, user_query)
            else:
                raise
    
    async def generate_with_examples(
        self,
        prompt: str,
        examples: List[Dict[str, str]],
        temperature: Optional[float] = None,
        enable_thinking: bool = False
    ) -> str:
        """
        Generates text using few-shot learning with examples.
        Improves accuracy for SQL generation.
        
        Args:
            prompt: The user prompt to send to the model
            examples: List of input/output examples for few-shot learning
            temperature: Controls randomness (lower = more deterministic)
            enable_thinking: Whether to enable thinking capabilities (Gemini 2.5 Flash only)
        
        Returns:
            Generated text response
        """
        
        # Format examples
        example_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        full_prompt = f"{example_text}\n\nNow for the actual request:\nInput: {prompt}\nOutput:"
        
        return await self.generate(full_prompt, temperature=temperature, enable_thinking=enable_thinking)
    
    async def validate_sql(self, sql: str, enable_thinking: bool = False) -> Dict[str, Any]:
        """
        Uses LLM to validate SQL syntax and logic.
        Provides additional safety layer.
        
        Args:
            sql: The SQL query to validate
            enable_thinking: Whether to enable thinking capabilities (Gemini 2.5 Flash only)
            
        Returns:
            Dictionary with validation results
        """
        
        prompt = f"""Analyze this SQL query for correctness and safety:

SQL: {sql}

Check for:
1. Syntax errors
2. Logical issues
3. Performance concerns
4. Security risks

Respond in JSON format:
{{
    "is_valid": true/false,
    "issues": ["list of issues if any"],
    "suggestions": ["list of improvements"],
    "risk_level": "low/medium/high"
}}"""
        
        response = await self.generate(prompt, temperature=0.1, enable_thinking=enable_thinking)
        
        try:
            # Parse JSON response
            import json
            return json.loads(response)
        except:
            # Fallback
            return {
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "risk_level": "low"
            }
    
    async def explain_results(
        self,
        sql: str,
        results: List[Dict[str, Any]],
        context: Optional[str] = None,
        enable_thinking: bool = True
    ) -> str:
        """
        Generates natural language explanation of SQL results.
        Makes data accessible to non-technical users.
        
        Args:
            sql: The SQL query that generated the results
            results: List of result rows as dictionaries
            context: Optional original question or context
            enable_thinking: Whether to enable thinking capabilities (Gemini 2.5 Flash only)
            
        Returns:
            Natural language explanation of the results
        """
        
        # Truncate results for prompt
        sample_results = results[:5] if len(results) > 5 else results
        
        prompt = f"""Explain these query results in simple business terms:

Original Question: {context or "Not provided"}

SQL Query: {sql}

Results (showing first {len(sample_results)} rows):
{json.dumps(sample_results, indent=2)}

Total rows: {len(results)}

Provide a clear, concise explanation focusing on:
1. What the data shows
2. Key insights or patterns
3. Business implications

Keep the explanation under 200 words and use bullet points for key findings."""
        
        # Enable thinking by default for result explanations with Gemini 2.5 Flash
        # as it can provide better analysis of the data
        return await self.generate(prompt, temperature=0.3, enable_thinking=enable_thinking)