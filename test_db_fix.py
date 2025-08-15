#!/usr/bin/env python3
"""
Simple test script to verify the database fix without uvicorn
"""

import sqlite3
from pathlib import Path
from db.db import open, get_conn
from store.project_repo import ProjectRepo
import uuid

def test_database_connection():
    """Test database connection and row factory"""
    print("Testing database connection...")
    
    try:
        # Initialize database
        db_path = "server/data/db.db"
        conn = open(db_path)
        print("✓ Database connection established")
        
        # Test row factory
        if conn.row_factory is not None:
            print("✓ Row factory is set")
        else:
            print("✗ Row factory is not set")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_project_creation():
    """Test project creation and retrieval"""
    print("\nTesting project creation...")
    
    try:
        conn = get_conn()
        repo = ProjectRepo(conn)
        
        # Create a test project
        project_id = str(uuid.uuid4())
        name = "Test Project"
        description = "A test project for database fix"
        
        repo.create_project(project_id, name, description)
        print("✓ Project created successfully")
        
        # Retrieve the project
        project = repo.get_project(project_id)
        
        if project is None:
            print("✗ Failed to retrieve project")
            return False
            
        # Test dictionary-style access
        try:
            project_id_retrieved = project['id']
            project_name = project['name']
            project_description = project['description']
            project_created_at = project['created_at']
            
            print(f"✓ Project retrieved successfully:")
            print(f"  ID: {project_id_retrieved}")
            print(f"  Name: {project_name}")
            print(f"  Description: {project_description}")
            print(f"  Created at: {project_created_at}")
            
            return True
            
        except (KeyError, TypeError) as e:
            print(f"✗ Failed to access project data: {e}")
            print(f"Project object type: {type(project)}")
            print(f"Project object: {project}")
            return False
            
    except Exception as e:
        print(f"✗ Project creation failed: {e}")
        return False

def test_get_all_projects():
    """Test getting all projects"""
    print("\nTesting get all projects...")
    
    try:
        conn = get_conn()
        repo = ProjectRepo(conn)
        
        projects = repo.get_all_projects()
        
        if projects is None:
            print("✗ Failed to get projects")
            return False
            
        print(f"✓ Retrieved {len(projects)} projects")
        
        # Test accessing first project if any exist
        if len(projects) > 0:
            first_project = projects[0]
            try:
                project_name = first_project['name']
                print(f"✓ First project name: {project_name}")
                return True
            except (KeyError, TypeError) as e:
                print(f"✗ Failed to access first project data: {e}")
                return False
        else:
            print("✓ No projects found (this is expected for a new database)")
            return True
            
    except Exception as e:
        print(f"✗ Get all projects failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Database Fix Test Suite ===\n")
    
    # Test database connection
    if not test_database_connection():
        print("\n✗ Database connection test failed")
        exit(1)
    
    # Test project creation
    if not test_project_creation():
        print("\n✗ Project creation test failed")
        exit(1)
    
    # Test get all projects
    if not test_get_all_projects():
        print("\n✗ Get all projects test failed")
        exit(1)
    
    print("\n=== All Tests Passed! ===")
    print("The database fix is working correctly.")
    print("The 'tuple indices must be integers or slices, not str' error should be resolved.")
