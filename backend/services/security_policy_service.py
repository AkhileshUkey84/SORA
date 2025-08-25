# services/security_policy_service.py
"""
Security Policy Service with educational transparency.
Key hackathon differentiator - makes security educational rather than frustrating.
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger()


class PolicyType(str, Enum):
    """Types of security policies"""
    DATA_ACCESS = "data_access"
    QUERY_SAFETY = "query_safety"
    RESOURCE_LIMITS = "resource_limits"
    PRIVACY_PROTECTION = "privacy_protection"
    COMPLIANCE = "compliance"


class PolicySeverity(str, Enum):
    """Severity levels for policy violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityPolicy(BaseModel):
    """Individual security policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    rule_logic: str
    educational_message: str
    violation_message: str
    remediation_steps: List[str] = Field(default_factory=list)
    learning_resources: List[str] = Field(default_factory=list)
    severity: PolicySeverity = PolicySeverity.WARNING
    auto_fixable: bool = False
    business_rationale: str = ""


class PolicyViolation(BaseModel):
    """Details of a policy violation"""
    policy_id: str
    policy_name: str
    violation_type: str
    severity: PolicySeverity
    message: str
    educational_explanation: str
    remediation_steps: List[str] = Field(default_factory=list)
    auto_fix_available: bool = False
    auto_fix_description: Optional[str] = None
    learning_opportunity: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SecurityPolicyService:
    """
    Educational security policy service that makes security violations
    into learning opportunities rather than frustrating roadblocks.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="SecurityPolicyService")
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_constitution = self._initialize_constitution()
        self._initialize_policies()
    
    def _initialize_constitution(self) -> Dict[str, str]:
        """Initialize the security policy constitution - the 'why' behind rules"""
        return {
            "data_protection": """
                🛡️ **Data Protection Principles**
                
                We protect your data because:
                • Your privacy is fundamental - you own your data
                • Regulatory compliance (GDPR, HIPAA, etc.) isn't optional
                • Data breaches can destroy businesses and lives
                • Trust is earned through consistent security practices
            """,
            
            "query_safety": """
                🔒 **Query Safety Guidelines**
                
                We validate queries because:
                • SQL injection attacks are still a top threat vector
                • Runaway queries can impact system performance for everyone
                • Accidental data exposure often happens through oversight
                • Defense in depth protects against human error
            """,
            
            "resource_management": """
                ⚡ **Resource Management Philosophy**
                
                We limit resource usage because:
                • Shared systems require fair resource allocation
                • Infinite queries can crash databases and cost money
                • Performance impacts user experience for everyone
                • Sustainability means being responsible with compute resources
            """,
            
            "privacy_by_design": """
                🔐 **Privacy by Design**
                
                We build privacy in because:
                • Privacy violations can't be undone after they happen
                • Different data requires different protection levels
                • Aggregation often provides better insights than raw data
                • User consent and data minimization are core principles
            """
        }
    
    def _initialize_policies(self):
        """Initialize comprehensive security policies"""
        
        # SQL Injection Prevention
        self.add_policy(SecurityPolicy(
            policy_id="sql_injection_prevention",
            name="SQL Injection Prevention",
            description="Prevents potentially malicious SQL patterns",
            policy_type=PolicyType.QUERY_SAFETY,
            rule_logic="Check for injection patterns like UNION SELECT, OR 1=1, etc.",
            educational_message="SQL injection is one of the oldest and most dangerous web vulnerabilities. Even innocent queries can accidentally contain patterns that look like injection attempts.",
            violation_message="Query contains patterns that could indicate SQL injection",
            remediation_steps=[
                "Remove suspicious patterns like 'OR 1=1' or 'UNION SELECT'",
                "Use parameterized queries instead of string concatenation",
                "Validate all user inputs before query construction"
            ],
            learning_resources=[
                "OWASP SQL Injection Prevention Cheat Sheet",
                "Why SQL Injection Still Matters in 2024"
            ],
            severity=PolicySeverity.CRITICAL,
            auto_fixable=False,
            business_rationale="SQL injection can lead to complete database compromise, data theft, and regulatory violations"
        ))
        
        # Row Limit Enforcement
        self.add_policy(SecurityPolicy(
            policy_id="row_limit_enforcement",
            name="Query Row Limit",
            description="Ensures queries don't return excessive amounts of data",
            policy_type=PolicyType.RESOURCE_LIMITS,
            rule_logic="Enforce maximum row limits to prevent resource exhaustion",
            educational_message="Large result sets can overwhelm both the system and your analysis. Most insights come from well-filtered, focused data rather than massive dumps.",
            violation_message="Query may return too many rows, impacting performance",
            remediation_steps=[
                "Add a LIMIT clause to your query",
                "Use filters to focus on relevant data",
                "Consider aggregation instead of raw row retrieval"
            ],
            learning_resources=[
                "Query Optimization Best Practices",
                "When to Use Aggregation vs Detail Queries"
            ],
            severity=PolicySeverity.WARNING,
            auto_fixable=True,
            business_rationale="Resource limits ensure fair system access and prevent accidental system overload"
        ))
        
        # PII Protection
        self.add_policy(SecurityPolicy(
            policy_id="pii_protection",
            name="Personal Information Protection",
            description="Prevents exposure of personally identifiable information",
            policy_type=PolicyType.PRIVACY_PROTECTION,
            rule_logic="Detect and prevent exposure of PII like emails, SSNs, phone numbers",
            educational_message="Personal information requires special handling. Even accidental exposure can violate privacy laws and damage trust.",
            violation_message="Query may expose personally identifiable information",
            remediation_steps=[
                "Use aggregated or anonymized data instead",
                "Apply differential privacy techniques",
                "Ensure proper consent for any PII usage"
            ],
            learning_resources=[
                "GDPR Compliance Guide",
                "Differential Privacy Techniques"
            ],
            severity=PolicySeverity.ERROR,
            auto_fixable=False,
            business_rationale="Privacy protection prevents legal violations and maintains customer trust"
        ))
        
        # Sensitive Column Access
        self.add_policy(SecurityPolicy(
            policy_id="sensitive_column_access",
            name="Sensitive Data Access Control",
            description="Controls access to columns containing sensitive information",
            policy_type=PolicyType.DATA_ACCESS,
            rule_logic="Restrict access to predefined sensitive columns",
            educational_message="Some data columns contain sensitive information that requires special permissions or handling procedures.",
            violation_message="Query attempts to access restricted data columns",
            remediation_steps=[
                "Request appropriate permissions if you have a legitimate need",
                "Use anonymized or aggregated versions of the data",
                "Consider whether your analysis truly needs this sensitive data"
            ],
            learning_resources=[
                "Data Classification Standards",
                "Access Control Best Practices"
            ],
            severity=PolicySeverity.ERROR,
            auto_fixable=False,
            business_rationale="Access controls protect sensitive business and personal information"
        ))
        
        # Query Complexity Limits
        self.add_policy(SecurityPolicy(
            policy_id="query_complexity_limit",
            name="Query Complexity Control",
            description="Prevents overly complex queries that may impact performance",
            policy_type=PolicyType.RESOURCE_LIMITS,
            rule_logic="Limit nested subqueries, joins, and complex operations",
            educational_message="Very complex queries can be hard to understand, debug, and may have unexpected performance impacts.",
            violation_message="Query is too complex and may impact system performance",
            remediation_steps=[
                "Break complex queries into simpler steps",
                "Consider using views or materialized tables for complex joins",
                "Review if all complexity is truly necessary"
            ],
            learning_resources=[
                "Query Simplification Techniques",
                "Database Performance Optimization"
            ],
            severity=PolicySeverity.WARNING,
            auto_fixable=False,
            business_rationale="Complexity limits ensure system stability and query maintainability"
        ))
        
        # Data Export Restrictions
        self.add_policy(SecurityPolicy(
            policy_id="data_export_restrictions",
            name="Data Export Controls",
            description="Manages how and what data can be exported from the system",
            policy_type=PolicyType.DATA_ACCESS,
            rule_logic="Control data export based on volume, content, and user permissions",
            educational_message="Data exports create copies outside our security controls. Understanding what you're exporting and why helps maintain security.",
            violation_message="Data export request requires additional review",
            remediation_steps=[
                "Verify you have permission to export this data",
                "Consider if analysis can be done within the system instead",
                "Ensure exported data will be handled securely"
            ],
            learning_resources=[
                "Data Governance Policies",
                "Secure Data Handling Practices"
            ],
            severity=PolicySeverity.INFO,
            auto_fixable=False,
            business_rationale="Export controls help prevent data leakage and maintain compliance"
        ))
    
    def add_policy(self, policy: SecurityPolicy):
        """Add a new security policy"""
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Added security policy: {policy.name}")
    
    def validate_query(self, sql: str, context: Dict[str, Any] = None) -> Tuple[bool, List[PolicyViolation]]:
        """
        Validate a query against all security policies.
        Returns (is_valid, violations)
        """
        violations = []
        context = context or {}
        
        for policy_id, policy in self.policies.items():
            violation = self._check_policy(sql, policy, context)
            if violation:
                violations.append(violation)
        
        # Determine if query should be blocked
        critical_violations = [v for v in violations if v.severity == PolicySeverity.CRITICAL]
        error_violations = [v for v in violations if v.severity == PolicySeverity.ERROR]
        
        is_valid = len(critical_violations) == 0 and len(error_violations) == 0
        
        return is_valid, violations
    
    def _check_policy(self, sql: str, policy: SecurityPolicy, context: Dict[str, Any]) -> Optional[PolicyViolation]:
        """Check a single policy against the query"""
        
        # SQL Injection patterns
        if policy.policy_id == "sql_injection_prevention":
            injection_patterns = [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bOR\b\s+\d+\s*=\s*\d+)",
                r"(\bAND\b\s+\d+\s*=\s*\d+)",
                r"(;\s*(DROP|DELETE|UPDATE|INSERT)\b)",
                r"(--\s*$)",
                r"(/\*.*?\*/)"
            ]
            
            import re
            for pattern in injection_patterns:
                if re.search(pattern, sql, re.IGNORECASE):
                    return PolicyViolation(
                        policy_id=policy.policy_id,
                        policy_name=policy.name,
                        violation_type="injection_pattern",
                        severity=policy.severity,
                        message=policy.violation_message,
                        educational_explanation=self._get_educational_explanation(policy, pattern),
                        remediation_steps=policy.remediation_steps,
                        learning_opportunity=policy.educational_message
                    )
        
        # Row limit check
        elif policy.policy_id == "row_limit_enforcement":
            if not re.search(r'\bLIMIT\s+\d+\b', sql, re.IGNORECASE):
                return PolicyViolation(
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    violation_type="missing_limit",
                    severity=policy.severity,
                    message="Query lacks row limit - may return excessive data",
                    educational_explanation="Adding a LIMIT clause helps control resource usage and improves query performance. Most business questions can be answered with a reasonable sample size.",
                    remediation_steps=policy.remediation_steps,
                    auto_fix_available=True,
                    auto_fix_description="Add LIMIT 1000 to query",
                    learning_opportunity=policy.educational_message
                )
        
        # PII detection
        elif policy.policy_id == "pii_protection":
            pii_patterns = [
                r"'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'",  # Email
                r"'\d{3}-\d{2}-\d{4}'",  # SSN
                r"'\d{4}\s?\d{4}\s?\d{4}\s?\d{4}'",  # Credit card
            ]
            
            for pattern in pii_patterns:
                if re.search(pattern, sql):
                    return PolicyViolation(
                        policy_id=policy.policy_id,
                        policy_name=policy.name,
                        violation_type="pii_detected",
                        severity=policy.severity,
                        message=policy.violation_message,
                        educational_explanation="Personal information detected in query. Consider if this data is truly necessary for your analysis.",
                        remediation_steps=policy.remediation_steps,
                        learning_opportunity=policy.educational_message
                    )
        
        # Sensitive columns
        elif policy.policy_id == "sensitive_column_access":
            sensitive_columns = ["password", "ssn", "credit_card", "api_key", "token"]
            sql_lower = sql.lower()
            
            for column in sensitive_columns:
                if column in sql_lower:
                    return PolicyViolation(
                        policy_id=policy.policy_id,
                        policy_name=policy.name,
                        violation_type="sensitive_column",
                        severity=policy.severity,
                        message=f"Query accesses sensitive column: {column}",
                        educational_explanation=f"The '{column}' column contains sensitive information that requires special handling and permissions.",
                        remediation_steps=policy.remediation_steps,
                        learning_opportunity=policy.educational_message
                    )
        
        # Query complexity
        elif policy.policy_id == "query_complexity_limit":
            complexity_score = self._calculate_query_complexity(sql)
            if complexity_score > 10:  # Arbitrary threshold
                return PolicyViolation(
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    violation_type="high_complexity",
                    severity=policy.severity,
                    message=f"Query complexity score ({complexity_score}) exceeds limit",
                    educational_explanation="Complex queries are harder to understand, maintain, and can have unpredictable performance impacts.",
                    remediation_steps=policy.remediation_steps,
                    learning_opportunity=policy.educational_message
                )
        
        return None
    
    def _calculate_query_complexity(self, sql: str) -> int:
        """Calculate a simple complexity score for the query"""
        complexity = 0
        sql_upper = sql.upper()
        
        # Count various complexity factors
        complexity += sql_upper.count("SELECT") * 2  # Multiple selects
        complexity += sql_upper.count("JOIN") * 3     # Joins add complexity
        complexity += sql_upper.count("UNION") * 4    # Unions are complex
        complexity += sql_upper.count("CASE") * 2     # Case statements
        complexity += sql_upper.count("EXISTS") * 3   # Subquery existence checks
        complexity += len(re.findall(r'\(SELECT', sql_upper)) * 3  # Subqueries
        
        return complexity
    
    def _get_educational_explanation(self, policy: SecurityPolicy, pattern: str) -> str:
        """Generate educational explanation for specific violations"""
        explanations = {
            "injection_pattern": "This pattern resembles SQL injection techniques used by attackers to manipulate queries and gain unauthorized access.",
            "missing_limit": "Without row limits, queries can return millions of rows, overwhelming both system resources and your ability to analyze the data effectively.",
            "pii_detected": "Personal information requires special privacy protections and may be subject to regulations like GDPR.",
            "sensitive_column": "This column contains sensitive data that requires appropriate security clearance and handling procedures."
        }
        
        return explanations.get(pattern, policy.educational_message)
    
    def generate_policy_explanation(self, policy_id: str) -> str:
        """Generate detailed explanation of a specific policy"""
        if policy_id not in self.policies:
            return "Policy not found"
        
        policy = self.policies[policy_id]
        
        explanation = [f"# 🛡️ {policy.name}\n"]
        explanation.append(f"**Purpose:** {policy.description}\n")
        explanation.append(f"**Why this matters:** {policy.educational_message}\n")
        
        if policy.business_rationale:
            explanation.append(f"**Business impact:** {policy.business_rationale}\n")
        
        explanation.append("**What to do if you encounter this:**")
        for i, step in enumerate(policy.remediation_steps, 1):
            explanation.append(f"{i}. {step}")
        
        if policy.learning_resources:
            explanation.append("\n**Learn more:**")
            for resource in policy.learning_resources:
                explanation.append(f"• {resource}")
        
        # Add constitutional context
        constitution_key = self._get_constitutional_context(policy.policy_type)
        if constitution_key and constitution_key in self.policy_constitution:
            explanation.append(f"\n{self.policy_constitution[constitution_key]}")
        
        return "\n".join(explanation)
    
    def _get_constitutional_context(self, policy_type: PolicyType) -> Optional[str]:
        """Map policy type to constitutional context"""
        mapping = {
            PolicyType.DATA_ACCESS: "data_protection",
            PolicyType.QUERY_SAFETY: "query_safety",
            PolicyType.RESOURCE_LIMITS: "resource_management",
            PolicyType.PRIVACY_PROTECTION: "privacy_by_design",
            PolicyType.COMPLIANCE: "data_protection"
        }
        return mapping.get(policy_type)
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of all policies for documentation"""
        policy_summary = {
            "total_policies": len(self.policies),
            "by_type": {},
            "by_severity": {},
            "auto_fixable": 0,
            "constitution": self.policy_constitution
        }
        
        for policy in self.policies.values():
            # Count by type
            policy_type = policy.policy_type.value
            policy_summary["by_type"][policy_type] = policy_summary["by_type"].get(policy_type, 0) + 1
            
            # Count by severity
            severity = policy.severity.value
            policy_summary["by_severity"][severity] = policy_summary["by_severity"].get(severity, 0) + 1
            
            # Count auto-fixable
            if policy.auto_fixable:
                policy_summary["auto_fixable"] += 1
        
        return policy_summary
    
    def suggest_query_improvements(self, sql: str, violations: List[PolicyViolation]) -> List[str]:
        """Suggest specific improvements based on violations"""
        suggestions = []
        
        for violation in violations:
            if violation.auto_fix_available:
                suggestions.append(f"✅ **Auto-fix available:** {violation.auto_fix_description}")
            else:
                suggestions.append(f"💡 **{violation.policy_name}:** {violation.remediation_steps[0] if violation.remediation_steps else 'Review policy requirements'}")
        
        # General suggestions
        if not re.search(r'\bLIMIT\s+\d+\b', sql, re.IGNORECASE):
            suggestions.append("⚡ **Performance tip:** Add LIMIT clause to control result size")
        
        if sql.upper().count("SELECT") > 1:
            suggestions.append("🔍 **Complexity tip:** Consider breaking complex queries into steps")
        
        return suggestions[:5]  # Top 5 suggestions
    
    def create_learning_path(self, violations: List[PolicyViolation]) -> Dict[str, Any]:
        """Create personalized learning path based on violations"""
        learning_path = {
            "title": "Your Security Learning Journey",
            "total_topics": len(set(v.policy_id for v in violations)),
            "estimated_time": f"{len(violations) * 5} minutes",
            "topics": []
        }
        
        seen_policies = set()
        for violation in violations:
            if violation.policy_id in seen_policies:
                continue
            
            seen_policies.add(violation.policy_id)
            policy = self.policies[violation.policy_id]
            
            topic = {
                "title": policy.name,
                "description": policy.educational_message,
                "difficulty": "beginner" if policy.severity in [PolicySeverity.INFO, PolicySeverity.WARNING] else "intermediate",
                "resources": policy.learning_resources,
                "hands_on_tips": policy.remediation_steps[:2]  # Top 2 tips
            }
            
            learning_path["topics"].append(topic)
        
        return learning_path
    
    def get_security_insights(self, query_history: List[str] = None) -> Dict[str, Any]:
        """Generate security insights and recommendations"""
        insights = {
            "security_score": 85,  # Would be calculated from actual usage
            "common_issues": [
                {"issue": "Missing LIMIT clauses", "frequency": "40%", "impact": "Performance"},
                {"issue": "Complex joins", "frequency": "25%", "impact": "Maintainability"},
                {"issue": "Unnecessary wildcards", "frequency": "15%", "impact": "Resource usage"}
            ],
            "improvements": [
                "🎯 Add LIMIT clauses to reduce resource usage",
                "🔍 Break complex queries into simpler components", 
                "⚡ Use specific column names instead of SELECT *"
            ],
            "security_trends": {
                "improving_areas": ["Query complexity management", "Resource efficiency"],
                "attention_needed": ["PII handling awareness", "Access pattern optimization"]
            },
            "next_steps": [
                "Review the Query Optimization guide",
                "Complete the Privacy Protection training",
                "Set up query performance monitoring"
            ]
        }
        
        return insights
