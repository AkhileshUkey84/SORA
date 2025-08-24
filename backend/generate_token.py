#!/usr/bin/env python3
"""
Generate JWT Bearer Token for testing
"""
import sys
import os
sys.path.append('.')

from app.middleware.auth import create_access_token
from datetime import timedelta

def generate_token(user_id: str = "test_user", minutes: int = 60):
    """Generate a JWT token for testing"""
    token_data = {"sub": user_id}
    expires = timedelta(minutes=minutes)
    token = create_access_token(token_data, expires)
    return token

if __name__ == "__main__":
    user_id = input("Enter user ID (default: test_user): ").strip() or "test_user"
    minutes = input("Token expiry in minutes (default: 60): ").strip()
    minutes = int(minutes) if minutes.isdigit() else 60
    
    print(f"\nGenerating token for user: {user_id}")
    print(f"Token expires in: {minutes} minutes")
    print("-" * 50)
    
    token = generate_token(user_id, minutes)
    
    print(f"Bearer Token:")
    print(f"{token}")
    print("\nUsage in HTTP requests:")
    print(f"Authorization: Bearer {token}")
    print("\nCurl example:")
    print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/api/datasets')
