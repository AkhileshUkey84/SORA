# SQL sanitization, PII detection
# utils/security.py
"""Security utilities for SQL validation and PII detection"""

import re
from typing import List, Dict, Any, Set, Optional
import sqlparse
from sqlparse.sql import Token, TokenList
import structlog


logger = structlog.get_logger()


class PIIDetector:
    """
    Detects personally identifiable information in queries and data.
    Critical for GDPR/privacy compliance.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="PIIDetector")
        
        # PII patterns
        self.patterns = {
            "email": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "ssn": re.compile(
                r'\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b'
            ),
            "phone": re.compile(
                r'\b(?:\+?1[-.\s]?)?KATEX_INLINE_OPEN?\d{3}KATEX_INLINE_CLOSE?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ),
            "credit_card": re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            "ip_address": re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            "date_of_birth": re.compile(
                r'\b(?:dob|birth.*date|date.*birth)\b',
                re.IGNORECASE
            )
        }
        
        # Sensitive column names
        self.sensitive_columns = {
            'ssn', 'social_security', 'social_security_number',
            'email', 'email_address', 'e_mail',
            'phone', 'phone_number', 'telephone', 'mobile',
            'address', 'street', 'city', 'state', 'zip', 'postal_code',
            'credit_card', 'card_number', 'cvv', 'expiry',
            'password', 'pwd', 'pass',
            'salary', 'wage', 'income',
            'dob', 'date_of_birth', 'birth_date',
            'medical', 'diagnosis', 'prescription'
        }
    
    def detect_pii_in_text(self, text: str) -> Dict[str, List[str]]:
        """Detect PII patterns in text"""
        
        found_pii = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_pii[pii_type] = matches
        
        return found_pii
    
    def detect_pii_in_sql(self, sql: str) -> Dict[str, Any]:
        """Detect PII in SQL query"""
        
        result = {
            "has_pii": False,
            "pii_in_values": {},
            "sensitive_columns": [],
            "risk_level": "low"
        }
        
        # Check for PII in literal values
        result["pii_in_values"] = self.detect_pii_in_text(sql)
        
        # Parse SQL to check column names
        try:
            parsed = sqlparse.parse(sql)[0]
            columns = self._extract_columns(parsed)
            
            # Check for sensitive column names
            for col in columns:
                col_lower = col.lower()
                if any(sensitive in col_lower for sensitive in self.sensitive_columns):
                    result["sensitive_columns"].append(col)
            
        except Exception as e:
            self.logger.warning("Failed to parse SQL for PII detection", error=str(e))
        
        # Determine if PII present
        if result["pii_in_values"] or result["sensitive_columns"]:
            result["has_pii"] = True
            
            # Assess risk level
            if result["pii_in_values"]:
                result["risk_level"] = "high"
            elif len(result["sensitive_columns"]) > 2:
                result["risk_level"] = "medium"
        
        return result
    
    def detect_pii_in_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect PII in query results"""
        
        result = {
            "has_pii": False,
            "pii_columns": {},
            "pii_count": 0
        }
        
        if not data:
            return result
        
        # Check each column
        for column in data[0].keys():
            column_values = [str(row.get(column, "")) for row in data[:100]]  # Sample
            
            # Check column name
            if any(sensitive in column.lower() for sensitive in self.sensitive_columns):
                result["pii_columns"][column] = "sensitive_name"
                result["has_pii"] = True
                continue
            
            # Check values
            for value in column_values:
                pii_found = self.detect_pii_in_text(value)
                if pii_found:
                    result["pii_columns"][column] = list(pii_found.keys())[0]
                    result["pii_count"] += 1
                    result["has_pii"] = True
                    break
        
        return result
    
    def mask_pii(self, text: str) -> str:
        """Mask PII in text"""
        
        masked = text
        
        # Mask each PII type
        for pii_type, pattern in self.patterns.items():
            if pii_type == "email":
                masked = pattern.sub(lambda m: f"{m.group()[:3]}***@***.***", masked)
            elif pii_type == "ssn":
                masked = pattern.sub("***-**-****", masked)
            elif pii_type == "phone":
                masked = pattern.sub("***-***-****", masked)
            elif pii_type == "credit_card":
                masked = pattern.sub("****-****-****-****", masked)
            elif pii_type == "ip_address":
                masked = pattern.sub("***.***.***.***", masked)
        
        return masked
    
    def _extract_columns(self, parsed: TokenList) -> List[str]:
        """Extract column names from parsed SQL"""
        
        columns = []
        
        def extract_identifiers(token):
            if hasattr(token, 'tokens'):
                for t in token.tokens:
                    extract_identifiers(t)
            elif token.ttype in (Token.Name, Token.Name.Builtin):
                columns.append(str(token.value))
        
        extract_identifiers(parsed)
        
        return columns


class SQLSanitizer:
    """
    SQL sanitization and validation utilities.
    Prevents injection attacks and ensures query safety.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="SQLSanitizer")
        
        # Forbidden keywords
        self.forbidden_keywords = {
            'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT',
            'CREATE', 'ALTER', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'
        }
        
        # Suspicious patterns
        self.suspicious_patterns = [
            re.compile(r';\s*$'),  # Multiple statements
            re.compile(r'--\s*$'),  # SQL comments
            re.compile(r'/\*.*\*/'),  # Block comments
            re.compile(r'UNION\s+ALL\s+SELECT', re.I),
            re.compile(r'OR\s+1\s*=\s*1', re.I),
            re.compile(r'OR\s+\'1\'\s*=\s*\'1\'', re.I),
            re.compile(r'SLEEP\s*KATEX_INLINE_OPEN', re.I),
            re.compile(r'WAITFOR\s+DELAY', re.I)
        ]
    
    def sanitize_sql(self, sql: str) -> Dict[str, Any]:
        """
        Sanitize SQL query and check for security issues.
        Returns sanitized SQL and security assessment.
        """
        
        result = {
            "original_sql": sql,
            "sanitized_sql": sql,
            "is_safe": True,
            "issues": [],
            "modifications": []
        }
        
        # Check for forbidden keywords
        sql_upper = sql.upper()
        for keyword in self.forbidden_keywords:
            if keyword in sql_upper:
                result["is_safe"] = False
                result["issues"].append(f"Forbidden keyword: {keyword}")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(sql):
                result["is_safe"] = False
                result["issues"].append(f"Suspicious pattern detected")
        
        # Sanitize if needed
        if not result["is_safe"]:
            # Remove comments
            sanitized = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
            sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
            
            # Remove trailing semicolons
            sanitized = sanitized.rstrip(';')
            
            # Ensure single statement
            if ';' in sanitized:
                sanitized = sanitized.split(';')[0]
                result["modifications"].append("Removed multiple statements")
            
            result["sanitized_sql"] = sanitized.strip()
            
            # Re-check sanitized version
            if self._is_read_only(result["sanitized_sql"]):
                result["is_safe"] = True
                result["modifications"].append("Query sanitized successfully")
        
        return result
    
    def validate_identifiers(self, sql: str, allowed_tables: Set[str]) -> bool:
        """Validate that SQL only references allowed tables"""
        
        try:
            parsed = sqlparse.parse(sql)[0]
            
            # Extract table references
            tables = self._extract_tables(parsed)
            
            # Check all tables are allowed
            for table in tables:
                if table.lower() not in {t.lower() for t in allowed_tables}:
                    self.logger.warning("Unauthorized table access attempted",
                                      table=table)
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to validate identifiers", error=str(e))
            return False
    
    def parameterize_values(self, sql: str) -> Dict[str, Any]:
        """
        Convert literal values to parameters.
        Prevents SQL injection through values.
        """
        
        params = {}
        param_counter = 0
        
        def replace_literal(match):
            nonlocal param_counter
            param_name = f"param_{param_counter}"
            param_counter += 1
            
            value = match.group(1) if match.group(1) else match.group(2)
            params[param_name] = value
            
            return f":{param_name}"
        
        # Replace string literals
        parameterized = re.sub(
            r"'([^']*)'|\"([^\"]*)\"",
            replace_literal,
            sql
        )
        
        return {
            "sql": parameterized,
            "params": params
        }
    
    def _is_read_only(self, sql: str) -> bool:
        """Check if SQL is read-only"""
        
        sql_upper = sql.upper().strip()
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Check for write operations
        write_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
            'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE'
        ]
        
        for keyword in write_keywords:
            if keyword in sql_upper:
                return False
        
        return True
    
    def _extract_tables(self, parsed: TokenList) -> Set[str]:
        """Extract table names from parsed SQL"""
        
        tables = set()
        
        # Simple extraction - would be more complex in production
        from_clause = False
        
        for token in parsed.tokens:
            if token.ttype is Token.Keyword and token.value.upper() == 'FROM':
                from_clause = True
            elif from_clause and token.ttype in (Token.Name, None):
                if hasattr(token, 'value'):
                    table_name = str(token.value).strip()
                    if table_name and not table_name.upper() in ['WHERE', 'GROUP', 'ORDER']:
                        tables.add(table_name)
            elif token.ttype is Token.Keyword:
                from_clause = False
        
        return tables


class SecurityValidator:
    """
    Comprehensive security validation combining multiple checks.
    Last line of defense against malicious queries.
    """
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.sql_sanitizer = SQLSanitizer()
        self.logger = logger.bind(component="SecurityValidator")
    
    def validate_query(
        self,
        sql: str,
        user_context: Dict[str, Any],
        dataset_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive security validation.
        Returns detailed security assessment.
        """
        
        validation_result = {
            "is_valid": True,
            "risk_level": "low",
            "checks_performed": [],
            "issues_found": [],
            "recommendations": []
        }
        
        # Check 1: SQL Sanitization
        sanitization = self.sql_sanitizer.sanitize_sql(sql)
        validation_result["checks_performed"].append("sql_sanitization")
        
        if not sanitization["is_safe"]:
            validation_result["is_valid"] = False
            validation_result["risk_level"] = "high"
            validation_result["issues_found"].extend(sanitization["issues"])
        
        # Check 2: PII Detection
        pii_check = self.pii_detector.detect_pii_in_sql(sql)
        validation_result["checks_performed"].append("pii_detection")
        
        if pii_check["has_pii"]:
            validation_result["risk_level"] = max(
                validation_result["risk_level"],
                pii_check["risk_level"]
            )
            validation_result["issues_found"].append("PII detected in query")
            validation_result["recommendations"].append(
                "Consider masking or excluding PII from results"
            )
        
        # Check 3: Resource Limits
        if "LIMIT" not in sql.upper():
            validation_result["issues_found"].append("No row limit specified")
            validation_result["recommendations"].append(
                "Add LIMIT clause to prevent resource exhaustion"
            )
        
        # Check 4: User Permissions
        if not self._check_user_permissions(user_context, dataset_context):
            validation_result["is_valid"] = False
            validation_result["risk_level"] = "critical"
            validation_result["issues_found"].append("Insufficient permissions")
        
        return validation_result
    
    def _check_user_permissions(
        self,
        user_context: Dict[str, Any],
        dataset_context: Dict[str, Any]
    ) -> bool:
        """Check if user has required permissions"""
        
        # Simplified check - in production would be more complex
        user_role = user_context.get("role", "user")
        dataset_access = dataset_context.get("access_level", "private")
        
        if dataset_access == "public":
            return True
        
        if dataset_access == "private" and user_role in ["admin", "analyst"]:
            return True
        
        return False