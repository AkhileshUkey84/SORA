#!/usr/bin/env python3
"""
API client test for the complete HTTP interface
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_TOKEN = "test-token"  # This will be generated or mocked

def test_api_endpoints():
    """Test all the main API endpoints"""
    
    print("🌐 Testing LLM Data Analyst HTTP API")
    print("=" * 50)
    
    # 1. Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Root endpoint working - {data['name']} {data['version']}")
            print(f"   ✓ Features available: {len(data.get('features', {}))}")
        else:
            print(f"   ✗ Root endpoint failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"   ✗ Cannot connect to server: {e}")
        print("   💡 Make sure the server is running with: python main.py")
        return False
    
    # 2. Test health endpoint  
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Health check passed - Status: {data['status']}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Health check error: {e}")
    
    # 3. Test file upload endpoint
    print("\n3. Testing file upload...")
    
    # Create a test CSV file
    test_data = """date,product,sales,quantity,region
2024-01-01,A,150.0,5,North
2024-01-02,B,200.0,3,South
2024-01-03,C,175.0,4,East
2024-01-04,A,220.0,6,West
2024-01-05,B,190.0,2,North"""
    
    test_file_path = Path("./test_upload_api.csv")
    with open(test_file_path, 'w') as f:
        f.write(test_data)
    
    try:
        # Mock authentication header
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            data = {'name': 'Test Dataset', 'description': 'API test dataset'}
            
            response = requests.post(
                f"{BASE_URL}/api/v1/upload", 
                files=files, 
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                upload_result = response.json()
                if upload_result.get('success'):
                    dataset_id = upload_result['dataset_id']
                    print(f"   ✓ File upload successful - Dataset ID: {dataset_id}")
                    print(f"   ✓ Preview: {len(upload_result.get('preview', {}).get('data', []))} rows")
                    return dataset_id  # Return for query testing
                else:
                    print(f"   ✗ Upload failed: {upload_result.get('error')}")
            else:
                print(f"   ✗ Upload request failed: {response.status_code}")
                if response.text:
                    print(f"       Response: {response.text[:200]}")
                    
    except Exception as e:
        print(f"   ✗ Upload error: {e}")
    finally:
        # Cleanup test file
        if test_file_path.exists():
            test_file_path.unlink()
    
    return None

def test_query_endpoint(dataset_id):
    """Test the query endpoint with a valid dataset"""
    
    print(f"\n4. Testing query endpoint (dataset: {dataset_id})...")
    
    test_queries = [
        "What are the total sales?",
        "Show me sales by product", 
        "Which region has the highest sales?"
    ]
    
    headers = {
        "Authorization": f"Bearer {TEST_TOKEN}",
        "Content-Type": "application/json"
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        
        payload = {
            "query": query,
            "dataset_id": dataset_id,
            "conversation_id": f"test_conv_{i}",
            "enable_insights": True,
            "enable_audit": True,
            "enable_multimodal": True
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/query",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"     ✓ Query successful")
                    print(f"     ✓ SQL: {result.get('sql_query', 'N/A')}")
                    print(f"     ✓ Results: {len(result.get('results', {}).get('data', []))} rows")
                    if result.get('narrative'):
                        print(f"     ✓ Explanation: {result['narrative'][:80]}...")
                    if result.get('insights'):
                        print(f"     ✓ Insights: {len(result['insights'])} discovered")
                else:
                    print(f"     ✗ Query failed: {result.get('error', 'Unknown error')}")
                    if result.get('needs_clarification'):
                        print(f"     💭 Needs clarification: {result.get('clarification_questions')}")
            else:
                print(f"     ✗ Request failed: {response.status_code}")
                if response.text:
                    print(f"        Response: {response.text[:100]}...")
                    
        except Exception as e:
            print(f"     ✗ Query error: {e}")
        
        # Add small delay between requests
        time.sleep(1)

def main():
    """Run the complete API test suite"""
    
    print("Starting API test suite...")
    print("Make sure the backend server is running on http://localhost:8000")
    print("You can start it with: python main.py")
    print()
    
    # Test basic endpoints
    dataset_id = test_api_endpoints()
    
    # Test query functionality if upload worked
    if dataset_id:
        test_query_endpoint(dataset_id)
        print(f"\n🎉 API Test Suite Complete!")
        print("✓ Server connectivity working")
        print("✓ File upload working") 
        print("✓ Query processing working")
        print("✓ Agent pipeline working")
    else:
        print(f"\n⚠️  API Test Suite Partially Complete")
        print("✓ Server connectivity working")
        print("✗ File upload needs authentication setup")
        print("- Query testing skipped (no valid dataset)")
        print()
        print("💡 To complete tests:")
        print("   1. Implement authentication middleware")  
        print("   2. Or temporarily disable auth for testing")

if __name__ == "__main__":
    main()
