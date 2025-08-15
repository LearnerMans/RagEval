#!/usr/bin/env python3
"""
Simple test script to verify the server is working
"""
import requests
import json

BASE_URL = "http://localhost:3000"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_create_project():
    """Test creating a project"""
    try:
        project_data = {
            "name": "Test Project",
            "description": "This is a test project"
        }
        
        response = requests.post(
            f"{BASE_URL}/projects/",
            json=project_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Create project: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Created project: {result}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Create project failed: {e}")
        return None

def test_get_projects():
    """Test getting all projects"""
    try:
        response = requests.get(f"{BASE_URL}/projects/")
        print(f"Get projects: {response.status_code}")
        if response.status_code == 200:
            projects = response.json()
            print(f"Found {len(projects)} projects:")
            for project in projects:
                print(f"  - {project['name']} (ID: {project['id']})")
            return projects
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Get projects failed: {e}")
        return None

if __name__ == "__main__":
    print("Testing RAG Eval Server...")
    print("=" * 40)
    
    # Test health endpoint
    if not test_health():
        print("Server is not running or health check failed!")
        exit(1)
    
    print("\n" + "=" * 40)
    
    # Test creating a project
    created_project = test_create_project()
    
    print("\n" + "=" * 40)
    
    # Test getting all projects
    projects = test_get_projects()
    
    print("\n" + "=" * 40)
    print("Test completed!")
