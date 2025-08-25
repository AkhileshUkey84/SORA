#!/usr/bin/env python3
"""
Token Generator for LLM Data Analyst API
Generates JWT bearer tokens for testing and development.
"""

import os
import sys
from datetime import datetime, timedelta
import jwt
import argparse
import json


def generate_token(
    user_id: str = "demo-user",
    email: str = "demo@example.com",
    role: str = "user",
    permissions: list = None,
    secret: str = None,
    algorithm: str = "HS256",
    expiry_minutes: int = 60
) -> str:
    """Generate a JWT token with specified parameters."""
    
    if permissions is None:
        permissions = ["query:execute", "dataset:read", "session:manage"]
    
    if secret is None:
        # Use the same secret as defined in .env
        secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "email": email,
        "role": role,
        "permissions": permissions,
        "exp": datetime.utcnow() + timedelta(minutes=expiry_minutes),
        "iat": datetime.utcnow(),
        "iss": "llm-data-analyst",
        "aud": "api-client"
    }
    
    token = jwt.encode(payload, secret, algorithm=algorithm)
    return token


def main():
    parser = argparse.ArgumentParser(description="Generate JWT Bearer Token")
    parser.add_argument("--user-id", default="demo-user", help="User ID")
    parser.add_argument("--email", default="demo@example.com", help="User email")
    parser.add_argument("--role", default="user", choices=["user", "admin"], help="User role")
    parser.add_argument("--permissions", nargs="*", help="User permissions")
    parser.add_argument("--expiry", type=int, default=60, help="Token expiry in minutes")
    parser.add_argument("--secret", help="JWT secret (defaults to env JWT_SECRET)")
    parser.add_argument("--format", choices=["token", "header", "curl"], default="token", 
                       help="Output format")
    parser.add_argument("--decode", help="Decode an existing token")
    
    args = parser.parse_args()
    
    if args.decode:
        # Decode existing token
        try:
            secret = args.secret or os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
            payload = jwt.decode(args.decode, secret, algorithms=["HS256"], options={"verify_exp": False, "verify_aud": False})
            print("Decoded token payload:")
            print(json.dumps(payload, indent=2, default=str))
        except Exception as e:
            print(f"Error decoding token: {e}")
            sys.exit(1)
        return
    
    # Generate token
    permissions = args.permissions or ["query:execute", "dataset:read", "session:manage"]
    if args.role == "admin":
        permissions = ["*"]  # Admin gets all permissions
    
    try:
        token = generate_token(
            user_id=args.user_id,
            email=args.email,
            role=args.role,
            permissions=permissions,
            secret=args.secret,
            expiry_minutes=args.expiry
        )
        
        if args.format == "token":
            print(token)
        elif args.format == "header":
            print(f"Authorization: Bearer {token}")
        elif args.format == "curl":
            print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/')
            
    except Exception as e:
        print(f"Error generating token: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
