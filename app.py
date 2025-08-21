import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from db.db import open, close
from contextlib import asynccontextmanager
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up...")
    try:
        # Initialize database connection
        logger.info("Initializing database connection...")
        global conn
        base_dir = Path(__file__).resolve().parent
        db_path = base_dir / "data" / "db.db"
        conn = open(str(db_path))
        logger.info("Database connection initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise
    finally:
        # Shutdown logic
        logger.info("Application shutting down...")
        try:
            close()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error during application shutdown: {str(e)}")
            raise



# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connection variable
conn = None

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code} for {request.method} {request.url}")
        return response
    except Exception as e:
        logger.error(f"Error processing request {request.method} {request.url}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "message": "Service is running"}

# Import handlers
from handlers.project_handler import router as project_router
from handlers.corpus_handler import router as corpus_router

# Include routers
app.include_router(project_router)
app.include_router(corpus_router)

# Root endpoint
@app.get("/")
async def home():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to RAG Eval API"}



