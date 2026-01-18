"""Main FastAPI application."""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings
from core.model_manager import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    
    Startup:
    - Load ML model
    
    Shutdown:
    - Clean up resources
    """
    # Startup
    logger.info("Starting up FL Client API...")
    
    try:
        model_path = Path("model2.pt")
        model_manager.load_model(model_path)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.warning("API will start without model (predictions will fail)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FL Client API...")


# Create FastAPI app
app = FastAPI(
    title="Federated Learning Client API",
    description="API for collecting swipe gestures and federated learning",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FL Client API",
        "version": "1.0.0",
        "client_id": settings.client_id,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting FL Client API on {settings.api_host}:{settings.api_port}")
    logger.info(f"Client ID: {settings.client_id}")
    logger.info(f"Data directory: {settings.data_dir}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
