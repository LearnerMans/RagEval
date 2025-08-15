#!/usr/bin/env python3
"""
Simple script to run the FastAPI server
"""
import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting RAG Eval Server...")
    print("Server will be available at: http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )
