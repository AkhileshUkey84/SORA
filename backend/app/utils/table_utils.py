"""Utilities for handling table names and SQL identifiers"""

def sanitize_table_name(table_name: str) -> str:
    """
    Sanitize table name to be SQL-compatible.
    
    Args:
        table_name: Original table name that may contain special characters
        
    Returns:
        SQL-compatible table name
    """
    # Replace hyphens and other special characters with underscores
    sanitized = table_name.replace("-", "_")
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.replace(".", "_")
    sanitized = sanitized.replace("@", "_")
    sanitized = sanitized.replace("#", "_")
    sanitized = sanitized.replace("%", "_")
    sanitized = sanitized.replace("&", "_")
    
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "table_unnamed"
    
    return sanitized
