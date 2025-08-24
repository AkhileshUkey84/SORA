# Security utilities
import re
from typing import List, Dict, Any
from app.config import settings

def sanitize_sql(sql: str) -> str:
    """Sanitize SQL query to prevent injection"""
    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Remove multiple statements
    sql = sql.split(';')[0] + ';'
    
    return sql.strip()

def check_dangerous_operations(sql: str) -> List[str]:
    """Check for dangerous SQL operations"""
    violations = []
    sql_upper = sql.upper()
    
    for keyword in settings.DANGEROUS_SQL_KEYWORDS:
        if keyword in sql_upper:
            violations.append(f"Dangerous operation: {keyword}")
    
    return violations

def detect_pii_in_text(text: str) -> Dict[str, List[str]]:
    """Detect PII patterns in text"""
    detected_pii = {}
    
    for pii_type, pattern in settings.PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            detected_pii[pii_type] = matches
    
    return detected_pii

def mask_pii(text: str) -> str:
    """Mask PII in text"""
    masked_text = text
    
    for pii_type, pattern in settings.PII_PATTERNS.items():
        masked_text = re.sub(
            pattern,
            f"[REDACTED-{pii_type.upper()}]",
            masked_text
        )
    
    return masked_text