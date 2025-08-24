# Environment configuration
# utils/config.py
"""
Configuration management with environment variables and fallbacks.
Designed for rapid deployment with minimal setup friction during demos.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized configuration with intelligent defaults for demo resilience.
    Falls back gracefully when external services unavailable.
    """
    
    # API Configuration
    app_name: str = "LLM Data Analyst"
    api_version: str = "v1"
    debug: bool = Field(False, env="DEBUG")
    
    # Security
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiry_minutes: int = Field(60, env="JWT_EXPIRY_MINUTES")
    
    # LLM Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    llm_model: str = Field("gemini-1.5-pro", env="LLM_MODEL")
    llm_temperature: float = Field(0.1, env="LLM_TEMPERATURE")  # Low for accuracy
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")
    
    # Database
    duckdb_path: str = Field(":memory:", env="DUCKDB_PATH")  # In-memory for speed
    max_query_rows: int = Field(1000, env="MAX_QUERY_ROWS")
    query_timeout_seconds: int = Field(30, env="QUERY_TIMEOUT_SECONDS")
    
    # Caching
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    enable_cache_fallback: bool = Field(True, env="ENABLE_CACHE_FALLBACK")
    
    # Rate Limiting
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(60, env="RATE_LIMIT_WINDOW_SECONDS")
    
    # Supabase
    supabase_url: Optional[str] = Field(None, env="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(None, env="SUPABASE_ANON_KEY")
    
    # Security Policies
    enable_pii_detection: bool = Field(True, env="ENABLE_PII_DETECTION")
    enable_sql_validation: bool = Field(True, env="ENABLE_SQL_VALIDATION")
    enable_row_limits: bool = Field(True, env="ENABLE_ROW_LIMITS")
    
    # Feature Flags
    enable_insights: bool = Field(True, env="ENABLE_INSIGHTS")
    enable_audit: bool = Field(True, env="ENABLE_AUDIT")
    enable_multimodal: bool = Field(True, env="ENABLE_MULTIMODAL")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "allow"  # Allow extra fields
    }


settings = Settings()