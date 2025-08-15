# Project Handler

This module provides a complete CRUD (Create, Read, Update, Delete) API for managing projects in the RAG Eval system.

## Features

- **Create Projects**: Create new projects with name and description
- **Get Project**: Retrieve a specific project by ID
- **Get All Projects**: List all projects in the system
- **Update Project**: Modify existing project details
- **Delete Project**: Remove projects from the system

## API Endpoints

### Create Project
```
POST /projects/
```

**Request Body:**
```json
{
  "name": "My Project",
  "description": "A description of my project"
}
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "My Project",
  "description": "A description of my project",
  "created_at": "2024-01-01T00:00:00"
}
```

### Get Project
```
GET /projects/{project_id}
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "My Project",
  "description": "A description of my project",
  "created_at": "2024-01-01T00:00:00"
}
```

### Get All Projects
```
GET /projects/
```

**Response:**
```json
[
  {
    "id": "uuid-string-1",
    "name": "Project 1",
    "description": "Description 1",
    "created_at": "2024-01-01T00:00:00"
  },
  {
    "id": "uuid-string-2",
    "name": "Project 2",
    "description": "Description 2",
    "created_at": "2024-01-02T00:00:00"
  }
]
```

### Update Project
```
PUT /projects/{project_id}
```

**Request Body:**
```json
{
  "name": "Updated Project Name",
  "description": "Updated description"
}
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "Updated Project Name",
  "description": "Updated description",
  "created_at": "2024-01-01T00:00:00"
}
```

### Delete Project
```
DELETE /projects/{project_id}
```

**Response:**
```json
{
  "message": "Project deleted successfully"
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `404`: Project not found
- `500`: Internal server error

Error responses include a detail message explaining the issue.

## Database Schema

Projects are stored in the `project` table with the following structure:

```sql
CREATE TABLE project (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  description   TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Usage Example

### Using curl

```bash
# Create a project
curl -X POST "http://localhost:8000/projects/" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Test Project", "description": "A test project"}'

# Get all projects
curl "http://localhost:8000/projects/"

# Get specific project
curl "http://localhost:8000/projects/{project_id}"

# Update project
curl -X PUT "http://localhost:8000/projects/{project_id}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Name", "description": "Updated description"}'

# Delete project
curl -X DELETE "http://localhost:8000/projects/{project_id}"
```

### Using Python requests

```python
import requests

# Create project
response = requests.post("http://localhost:8000/projects/", json={
    "name": "My Project",
    "description": "Project description"
})
project = response.json()

# Get all projects
projects = requests.get("http://localhost:8000/projects/").json()

# Update project
updated = requests.put(f"http://localhost:8000/projects/{project['id']}", json={
    "name": "Updated Name"
}).json()

# Delete project
requests.delete(f"http://localhost:8000/projects/{project['id']}")
```

## Testing

Run the test script to verify the handler functionality:

```bash
cd server
python test_project_handler.py
```

Make sure the server is running on `http://localhost:8000` before running the tests.

## Dependencies

- FastAPI
- Pydantic
- SQLite3 (via Python standard library)
- UUID (via Python standard library)

## Installation

Install the required dependencies:

```bash
cd server
uv sync
```

## Running the Server

```bash
cd server
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000` with automatic API documentation at `http://localhost:8000/docs`.
