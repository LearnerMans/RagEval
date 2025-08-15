# RAG Eval Server Integration Guide

This guide explains how to run the server and test the frontend-backend integration for the RAG Eval project.

## Prerequisites

- Python 3.12 or higher
- Node.js and npm (for the frontend)

## Setup and Running

### 1. Install Server Dependencies

```bash
cd server
uv sync
```

### 2. Start the Server

```bash
cd server
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### 3. Test the Server

Run the test script to verify the server is working:

```bash
cd server
uv run python test_server.py
```

This will test:
- Health endpoint
- Project creation
- Project listing

### 4. Start the Frontend

In a new terminal:

```bash
cd client
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`

## API Endpoints

The server provides the following endpoints:

- `GET /health` - Health check
- `GET /projects/` - Get all projects
- `POST /projects/` - Create a new project
- `GET /projects/{id}` - Get a specific project
- `PUT /projects/{id}` - Update a project
- `DELETE /projects/{id}` - Delete a project

## Project Data Structure

Projects have the following structure:

```json
{
  "id": "uuid-string",
  "name": "Project Name",
  "description": "Project description",
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Frontend Integration

The frontend uses the `projectAPI` service in `client/src/services/api.js` to communicate with the backend. The integration includes:

1. **Project Creation**: The `ProjectModal` component sends project data to the backend
2. **Project Listing**: The `ProjectsDashboard` component fetches and displays projects
3. **Real-time Updates**: New projects are added to the list immediately after creation

## Troubleshooting

### CORS Issues
If you see CORS errors, make sure the server is running and the CORS middleware is properly configured in `app.py`.

### Database Issues
The database file is created automatically at `server/data/db.db`. Make sure the server has write permissions to the `server/data/` directory.

### Port Conflicts
If port 8000 is in use, you can change it in the uvicorn command:
```bash
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8001
```
Then update the `API_BASE_URL` in `client/src/services/api.js` to match.

## Development Workflow

1. Start the server first
2. Start the frontend
3. Test project creation through the UI
4. Check the server logs for any errors
5. Use the test script to verify API functionality
