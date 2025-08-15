import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from store.project_repo import ProjectRepo
from db.db import get_conn

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str

# Router
router = APIRouter(prefix="/projects", tags=["projects"])

def get_project_repo():
    """Dependency to get project repository with database connection"""
    conn = get_conn()
    return ProjectRepo(conn)

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate,
    repo: ProjectRepo = Depends(get_project_repo)
):
    """
    Create a new project
    
    Args:
        project_data: Project creation data
        repo: Project repository dependency
        
    Returns:
        ProjectResponse: Created project details
        
    Raises:
        HTTPException: If project creation fails
    """
    try:
        logger.info(f"Creating project: {project_data.name}")
        
        # Generate unique ID for the project
        project_id = str(uuid.uuid4())
        
        # Create project in database
        repo.create_project(project_id, project_data.name, project_data.description)
        
        # Fetch the created project to return
        project = repo.get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=500, detail="Failed to create project")
        
        logger.info(f"Project created successfully with ID: {project_id}")
        
        return ProjectResponse(
            id=project['id'],
            name=project['name'],
            description=project['description'],
            created_at=project['created_at']
        )
        
    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    repo: ProjectRepo = Depends(get_project_repo)
):
    """
    Get a specific project by ID
    
    Args:
        project_id: Project ID
        repo: Project repository dependency
        
    Returns:
        ProjectResponse: Project details
        
    Raises:
        HTTPException: If project not found
    """
    try:
        logger.info(f"Fetching project: {project_id}")
        
        project = repo.get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectResponse(
            id=project['id'],
            name=project['name'],
            description=project['description'],
            created_at=project['created_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch project: {str(e)}")

@router.get("/", response_model=List[ProjectResponse])
async def get_all_projects(
    repo: ProjectRepo = Depends(get_project_repo)
):
    """
    Get all projects
    
    Args:
        repo: Project repository dependency
        
    Returns:
        List[ProjectResponse]: List of all projects
        
    Raises:
        HTTPException: If fetching projects fails
    """
    try:
        logger.info("Fetching all projects")
        
        projects = repo.get_all_projects()
        
        return [
            ProjectResponse(
                id=project['id'],
                name=project['name'],
                description=project['description'],
                created_at=project['created_at']
            )
            for project in projects
        ]
        
    except Exception as e:
        logger.error(f"Error fetching all projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch projects: {str(e)}")

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    repo: ProjectRepo = Depends(get_project_repo)
):
    """
    Update a project
    
    Args:
        project_id: Project ID
        project_data: Project update data
        repo: Project repository dependency
        
    Returns:
        ProjectResponse: Updated project details
        
    Raises:
        HTTPException: If project not found or update fails
    """
    try:
        logger.info(f"Updating project: {project_id}")
        
        # Check if project exists
        existing_project = repo.get_project(project_id)
        if not existing_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Prepare update data
        name = project_data.name if project_data.name is not None else existing_project['name']
        description = project_data.description if project_data.description is not None else existing_project['description']
        
        # Update project
        repo.update_project(project_id, name, description)
        
        # Fetch updated project
        updated_project = repo.get_project(project_id)
        
        logger.info(f"Project updated successfully: {project_id}")
        
        return ProjectResponse(
            id=updated_project['id'],
            name=updated_project['name'],
            description=updated_project['description'],
            created_at=updated_project['created_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    repo: ProjectRepo = Depends(get_project_repo)
):
    """
    Delete a project
    
    Args:
        project_id: Project ID
        repo: Project repository dependency
        
    Returns:
        dict: Success message
        
    Raises:
        HTTPException: If project not found or deletion fails
    """
    try:
        logger.info(f"Deleting project: {project_id}")
        
        # Check if project exists
        existing_project = repo.get_project(project_id)
        if not existing_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Delete project
        repo.delete_project(project_id)
        
        logger.info(f"Project deleted successfully: {project_id}")
        
        return {"message": "Project deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")
