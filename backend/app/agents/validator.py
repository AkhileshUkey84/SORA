import logging
import re
from typing import Dict, Any, List
from app.models.query import GuardrailResult
from app.config import settings

logger = logging.getLogger(__name__)

class Validator:
    """Validates SQL queries for safety and correctness"""
    
    def __init__(self):
        self.dangerous_keywords = settings.DANGEROUS_SQL_KEYWORDS
        self.pii_patterns = settings.PII_PATTERNS
        
    async def validate(
        self,
        sql: str,
        schema: Dict[str, Any],
        question: str
    ) -> GuardrailResult:
        """Validate SQL query for safety and correctness"""
        
        violations = []
        modified_sql = sql
        passed = True
        
        # Check if this is a metadata/schema query
        if self._is_metadata_query(question, sql):
            return GuardrailResult(
                passed=True,
                violations=[],
                modified_query=None,
                explanation="Metadata query allowed"
            )
        
        # Check for dangerous operations
        danger_check = self._check_dangerous_operations(sql)
        if danger_check["found"]:
            violations.extend(danger_check["violations"])
            passed = False
        
        # Check for PII exposure
        pii_check = self._check_pii_exposure(sql, schema)
        if pii_check["found"]:
            violations.extend(pii_check["violations"])
            modified_sql = pii_check["modified_sql"]
        
        # Check for performance issues
        perf_check = self._check_performance_issues(sql)
        if perf_check["found"]:
            violations.extend(perf_check["violations"])
            modified_sql = perf_check["modified_sql"]
        
        # Validate column references
        column_check = self._validate_columns(sql, schema)
        if column_check["invalid"]:
            violations.extend(column_check["violations"])
            passed = False
        
        # Generate explanation
        if not passed:
            explanation = f"Query blocked due to: {', '.join(violations[:3])}"
        elif violations:
            explanation = f"Query modified for safety: {', '.join(violations[:3])}"
        else:
            explanation = "Query passed all safety checks"
        
        result = GuardrailResult(
            passed=passed,
            violations=violations,
            modified_query=modified_sql if violations else None,
            explanation=explanation
        )
        
        logger.info(f"Validation result: {result}")
        return result
    
    def _check_dangerous_operations(self, sql: str) -> Dict[str, Any]:
        """Check for dangerous SQL operations"""
        found_operations = []
        sql_upper = sql.upper()
        
        for keyword in self.dangerous_keywords:
            if keyword in sql_upper:
                found_operations.append(f"{keyword} operation detected")
        
        return {
            "found": len(found_operations) > 0,
            "violations": found_operations
        }
    
    def _check_pii_exposure(self, sql: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential PII exposure"""
        violations = []
        modified_sql = sql
        
        # Check if selecting sensitive columns
        sensitive_columns = []
        for col_name, col_info in schema.items():
            if col_info.get("has_pii", False):
                sensitive_columns.append(col_name)
                if col_name.lower() in sql.lower():
                    violations.append(f"PII column '{col_name}' accessed")
                    # Mask the column in SELECT
                    pattern = rf"\b{col_name}\b"
                    modified_sql = re.sub(
                        pattern,
                        f"'[REDACTED-{col_info.get('pii_type', 'PII')}]' as {col_name}",
                        modified_sql,
                        flags=re.IGNORECASE
                    )
        
        # Check for SELECT *
        if "SELECT *" in sql.upper() and sensitive_columns:
            violations.append("SELECT * used with PII columns present")
            # Replace SELECT * with specific columns excluding PII
            safe_columns = [
                col for col, info in schema.items() 
                if not info.get("has_pii", False)
            ]
            if safe_columns:
                modified_sql = modified_sql.replace(
                    "SELECT *",
                    f"SELECT {', '.join(safe_columns[:10])}"
                )
        
        return {
            "found": len(violations) > 0,
            "violations": violations,
            "modified_sql": modified_sql
        }
    
    def _check_performance_issues(self, sql: str) -> Dict[str, Any]:
        """Check for performance issues"""
        violations = []
        modified_sql = sql
        
        # Check for missing LIMIT
        if "LIMIT" not in sql.upper():
            violations.append("Missing LIMIT clause")
            modified_sql = modified_sql.rstrip(";") + f" LIMIT {settings.MAX_QUERY_ROWS};"
        
        # Check for excessive LIMIT
        limit_match = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value > settings.MAX_QUERY_ROWS:
                violations.append(f"LIMIT {limit_value} exceeds maximum {settings.MAX_QUERY_ROWS}")
                modified_sql = re.sub(
                    r"LIMIT\s+\d+",
                    f"LIMIT {settings.MAX_QUERY_ROWS}",
                    modified_sql,
                    flags=re.IGNORECASE
                )
        
        return {
            "found": len(violations) > 0,
            "violations": violations,
            "modified_sql": modified_sql
        }
    
    def _validate_columns(self, sql: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all referenced columns exist"""
        violations = []
        
        # Create case-insensitive schema column set
        schema_columns = set(col.lower() for col in schema.keys())
        
        # Common SQL keywords and functions to ignore
        sql_keywords = {
            'select', 'from', 'where', 'group', 'by', 'order', 'limit', 'as', 'and', 'or',
            'sum', 'count', 'avg', 'max', 'min', 'distinct', 'asc', 'desc', 'having',
            'inner', 'left', 'right', 'join', 'on', 'union', 'case', 'when', 'then', 'else', 'end'
        }
        
        # Extract all potential column names from the query
        # This is a simplified approach - in production, use a proper SQL parser
        potential_columns = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql, re.IGNORECASE)
        
        for col in potential_columns:
            col_lower = col.lower()
            
            # Skip SQL keywords, functions, and table names
            if (col_lower not in sql_keywords and 
                not col_lower.startswith('dataset_') and  # Skip table names
                not col_lower.startswith('total_') and    # Skip common aliases
                col_lower not in ['sales', 'revenue', 'value']):
                
                # Check if it's a valid schema column
                if col_lower not in schema_columns:
                    # Only flag if it looks like a column reference (not an alias or function result)
                    if not self._is_likely_alias_or_function_result(col, sql):
                        violations.append(f"Column '{col}' not found in schema")
        
        return {
            "invalid": len(violations) > 0,
            "violations": violations
        }
    
    def _is_likely_alias_or_function_result(self, col: str, sql: str) -> bool:
        """Check if a column reference is likely an alias or function result"""
        col_lower = col.lower()
        sql_lower = sql.lower()
        
        # Check if it appears after AS keyword (alias)
        if f' as {col_lower}' in sql_lower:
            return True
        
        # Check if it appears in ORDER BY but not in SELECT columns (likely an alias)
        if f'order by {col_lower}' in sql_lower and col_lower not in ['region', 'price', 'quantity']:
            return True
            
        return False
    
    def _is_metadata_query(self, question: str, sql: str) -> bool:
        """Check if this is a metadata/schema information query"""
        question_lower = question.lower()
        sql_lower = sql.lower()
        
        # Keywords that indicate metadata queries
        metadata_keywords = [
            'column', 'schema', 'structure', 'describe', 'show', 'information',
            'how many column', 'what column', 'list column', 'column name',
            'table structure', 'field', 'attribute'
        ]
        
        # Check if question contains metadata keywords
        for keyword in metadata_keywords:
            if keyword in question_lower:
                return True
        
        # Check if SQL contains information schema or metadata patterns
        metadata_sql_patterns = [
            'information_schema', 'show columns', 'describe', 'pragma',
            'union all', 'group_concat', 'column_name'
        ]
        
        for pattern in metadata_sql_patterns:
            if pattern in sql_lower:
                return True
                
        return False
