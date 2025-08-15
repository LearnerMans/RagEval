#!/usr/bin/env python3
"""
Simple test script for project handler functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_create_project():
    """Test creating a new project"""
    print("Testing project creation...")
    
    project_data = {
        "name": "Test Project",
        "description": "A test project for RAG evaluation"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/projects/", json=project_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            project = response.json()
            print(f"Created Project: {json.dumps(project, indent=2)}")
            return project['id']
        else:
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the server is running.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def test_get_project(project_id):
    """Test getting a specific project"""
    if not project_id:
        print("No project ID provided")
        return
        
    print(f"\nTesting get project with ID: {project_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/projects/{project_id}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            project = response.json()
            print(f"Retrieved Project: {json.dumps(project, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_get_all_projects():
    """Test getting all projects"""
    print("\nTesting get all projects...")
    
    try:
        response = requests.get(f"{BASE_URL}/projects/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            projects = response.json()
            print(f"All Projects: {json.dumps(projects, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_update_project(project_id):
    """Test updating a project"""
    if not project_id:
        print("No project ID provided")
        return
        
    print(f"\nTesting update project with ID: {project_id}")
    
    update_data = {
        "name": "Updated Test Project",
        "description": "An updated test project description"
    }
    
    try:
        response = requests.put(f"{BASE_URL}/projects/{project_id}", json=update_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            project = response.json()
            print(f"Updated Project: {json.dumps(project, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_delete_project(project_id):
    """Test deleting a project"""
    if not project_id:
        print("No project ID provided")
        return
        
    print(f"\nTesting delete project with ID: {project_id}")
    
    try:
        response = requests.delete(f"{BASE_URL}/projects/{project_id}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Delete Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("=== Project Handler Test Suite ===\n")
    
    # Test project creation
    project_id = test_create_project()
    
    # Test getting all projects
    test_get_all_projects()
    
    if project_id:
        # Test getting specific project
        test_get_project(project_id)
        
        # Test updating project
        test_update_project(project_id)
        
        # Test deleting project
        test_delete_project(project_id)
    
    print("\n=== Test Suite Complete ===")
