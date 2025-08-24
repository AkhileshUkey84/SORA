# Security validation
# agents/validator_agent.py
"""
Security-focused validation agent ensuring SQL safety and data privacy.
Last line of defense against malicious queries and data exposure.
"""

from typing import Dict, Any, List, Optional, Set
import re
import sqlparse
from pydantic import BaseModel
from agents.base import BaseAgent, AgentContext, AgentResult
from utils.security import PIIDetector, SQLSanitizer


class ValidationRule:
    """Base class for validation rules"""
    def validate(self, sql: str, context: AgentContext) -> tuple[bool, Optional[str]]:
        raise NotImplementedError


class SecurityViolation(BaseModel):
    """Details about a security violation"""
    rule: str
    severity: str  # "high", "medium", "low"
    message: str
    suggestion: Optional[str] = None


class ValidatorAgent(BaseAgent):
    """
    Comprehensive SQL validation with focus on security and privacy.
    Provides clear explanations when queries are modified or blocked.
    """
    
    def __init__(self, pii_detector: PIIDetector, settings):
        super().__init__("ValidatorAgent")
        self.pii_detector = pii_detector
        self.settings = settings
        self.sanitizer = SQLSanitizer()
        self.rules = self._initialize_rules()
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Validates and potentially modifies SQL for security compliance.
        Returns modified SQL with explanations of any changes.
        """
        
        sql = kwargs.get("sql", "")
        if not sql:
            return AgentResult(
                success=False,
                error="No SQL query provided for validation"
            )
        
        violations = []
        modified_sql = sql
        
        # Run all validation rules
        for rule in self.rules:
            passed, message = rule.validate(modified_sql, context)
            if not passed:
                violations.append(SecurityViolation(
                    rule=rule.__class__.__name__,
                    severity=rule.severity,
                    message=message,
                    suggestion=rule.suggestion
                ))
                
                # Attempt to fix if possible
                if hasattr(rule, 'fix'):
                    fixed_sql = rule.fix(modified_sql, context)
                    if fixed_sql:
                        modified_sql = fixed_sql
                        violations[-1].suggestion = f"Query modified: {rule.fix_description}"
        
        # Check for PII in the query itself
        pii_found = await self._check_query_pii(modified_sql)
        if pii_found:
            violations.append(SecurityViolation(
                rule="PIIDetection",
                severity="high",
                message="Query contains potential PII",
                suggestion="Remove personal identifiers from query"
            ))
        
        # Determine if query should be blocked
        high_severity_violations = [v for v in violations if v.severity == "high"]
        
        if high_severity_violations and modified_sql == sql:
            # Couldn't fix high severity issues
            return AgentResult(
                success=False,
                error="Query blocked due to security violations",
                data={
                    "violations": [v.dict() for v in violations],
                    "original_sql": sql
                }
            )
        
        # Add row limit if missing
        if self.settings.enable_row_limits:
            modified_sql = self._enforce_row_limit(modified_sql)
        
        # Generate human-friendly explanation
        explanation = self._generate_explanation(violations, sql != modified_sql)
        
        return AgentResult(
            success=True,
            data={
                "validated_sql": modified_sql,
                "original_sql": sql,
                "modifications": sql != modified_sql,
                "violations": [v.dict() for v in violations],
                "explanation": explanation
            },
            confidence=0.9 if not violations else 0.7
        )
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize all validation rules"""
        
        class SelectOnlyRule(ValidationRule):
            severity = "high"
            suggestion = "Only SELECT queries are allowed"
            
            def validate(self, sql: str, context: AgentContext) -> tuple[bool, Optional[str]]:
                parsed = sqlparse.parse(sql)
                if not parsed:
                    return False, "Invalid SQL syntax"
                
                first_token = parsed[0].token_first(skip_ws=True, skip_cm=True)
                if not first_token or first_token.ttype != sqlparse.tokens.DML or \
                   first_token.value.upper() != 'SELECT':
                    return False, "Query must be a SELECT statement"
                
                return True, None
        
        class NoSubqueryRule(ValidationRule):
            severity = "medium"
            suggestion = "Subqueries can impact performance"
            fix_description = "Simplified to single query"
            
            def validate(self, sql: str, context: AgentContext) -> tuple[bool, Optional[str]]:
                # Check for common subquery patterns
                if '(SELECT' in sql.upper():
                    return False, "Subqueries are not allowed for performance reasons"
                return True, None
        
        class ColumnWhitelistRule(ValidationRule):
            severity = "medium"
            suggestion = "Use only existing columns"
            
            def validate(self, sql: str, context: AgentContext) -> tuple[bool, Optional[str]]:
                # This would check against actual schema
                # For now, just check for obvious issues
                if re.search(r'\b(password|ssn|credit_card)\b', sql, re.I):
                    return False, "Query attempts to access sensitive columns"
                return True, None
        
        class InjectionPreventionRule(ValidationRule):
            severity = "high"
            suggestion = "Remove special characters"
            
            def validate(self, sql: str, context: AgentContext) -> tuple[bool, Optional[str]]:
                # Check for SQL injection patterns
                injection_patterns = [
                    r';\s*DROP',
                    r';\s*DELETE',
                    r'--\s*$',
                    r'\/\*.*\*\/',
                    r'UNION\s+SELECT',
                    r'OR\s+1\s*=\s*1',
                    r'AND\s+1\s*=\s*0'
                ]
                
                for pattern in injection_patterns:
                    if re.search(pattern, sql, re.I):
                        return False, "Query contains potential SQL injection"
                
                return True, None
        
        return [
            SelectOnlyRule(),
            NoSubqueryRule(),
            ColumnWhitelistRule(),
            InjectionPreventionRule()
        ]
    
    def _enforce_row_limit(self, sql: str) -> str:
        """Ensures query has appropriate row limit"""
        
        # Parse to check for existing LIMIT
        parsed = sqlparse.parse(sql)[0]
        has_limit = False
        
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'LIMIT':
                has_limit = True
                break
        
        if not has_limit:
            # Add default limit
            sql = f"{sql} LIMIT {self.settings.max_query_rows}"
        else:
            # Ensure limit isn't too high
            limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.I)
            if limit_match:
                current_limit = int(limit_match.group(1))
                if current_limit > self.settings.max_query_rows:
                    sql = re.sub(
                        r'LIMIT\s+\d+',
                        f'LIMIT {self.settings.max_query_rows}',
                        sql,
                        flags=re.I
                    )
        
        return sql
    
    async def _check_query_pii(self, sql: str) -> bool:
        """Check if query contains PII patterns"""
        
        # Check for common PII patterns in WHERE clauses
        pii_patterns = [
            r"'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'",  # Email
            r"'\d{3}-\d{2}-\d{4}'",  # SSN
            r"'\d{4}\s?\d{4}\s?\d{4}\s?\d{4}'",  # Credit card
            r"'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'",  # Phone
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, sql):
                return True
        
        return False
    
    def _generate_explanation(self, violations: List[SecurityViolation], 
                            modified: bool) -> str:
        """
        Generates human-friendly explanation of validation results.
        Critical for user trust and understanding.
        """
        
        if not violations and not modified:
            return "Query validated successfully with no modifications needed."
        
        parts = []
        
        if modified:
            parts.append("Your query was modified for security and performance:")
        
        # Group by severity
        high = [v for v in violations if v.severity == "high"]
        medium = [v for v in violations if v.severity == "medium"]
        
        if high:
            parts.append(f"\n⚠️  Critical issues fixed ({len(high)}):")
            for v in high:
                parts.append(f"  • {v.message}")
                if v.suggestion:
                    parts.append(f"    → {v.suggestion}")
        
        if medium:
            parts.append(f"\n⚡ Performance optimizations ({len(medium)}):")
            for v in medium:
                parts.append(f"  • {v.message}")
        
        if modified and not violations:
            parts.append("\n✓ Added row limit for performance")
        
        return "\n".join(parts)