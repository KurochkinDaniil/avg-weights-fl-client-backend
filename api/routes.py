"""API routes for swipe gesture collection."""
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from typing import Dict
import logging

from api.models import SwipeRequest, SwipeResponse
from services import PredictionService, StorageService, TrainingService
from core.exceptions import (
    ModelNotLoadedException,
    PredictionException,
    StorageException,
    TrainingException
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["swipes"])

# Initialize services
prediction_service = PredictionService()
storage_service = StorageService()
training_service = TrainingService()


@router.post("/swipes", response_model=SwipeResponse, status_code=status.HTTP_202_ACCEPTED)
async def receive_swipe(
    swipe: SwipeRequest,
    background_tasks: BackgroundTasks
) -> SwipeResponse:
    """
    Receive and store swipe gesture from frontend.
    
    Uses background task for storage to avoid blocking the response.
    
    Args:
        swipe: Swipe gesture data
        background_tasks: FastAPI background tasks
    
    Returns:
        Response with status and gesture ID (202 Accepted)
    """
    try:
        # Convert to list of dicts
        coords = [{"x": p.x, "y": p.y, "t": p.t} for p in swipe.coords]
        
        # Add storage task to background
        background_tasks.add_task(
            storage_service.save_swipe,
            gesture_id=swipe.gesture_id,
            coords=coords,
            word=swipe.word
        )
        
        logger.info(
            f"Accepted swipe: {swipe.gesture_id}, "
            f"word: '{swipe.word}', points: {len(coords)} (saving in background)"
        )
        
        return SwipeResponse(
            status="accepted",
            gesture_id=swipe.gesture_id,
            message="Swipe gesture accepted, saving in background"
        )
    
    except Exception as e:
        logger.error(f"Error accepting swipe: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to accept swipe gesture: {str(e)}"
        )


@router.post("/predict", response_model=Dict)
async def predict_swipe(swipe: SwipeRequest) -> Dict:
    """
    Predict word from swipe gesture.
    
    Args:
        swipe: Swipe gesture data
    
    Returns:
        Dictionary with predicted word
    """
    try:
        coords = [{"x": p.x, "y": p.y, "t": p.t} for p in swipe.coords]
        predicted_word = prediction_service.predict(coords)
        
        return {
            "gesture_id": swipe.gesture_id,
            "predicted_word": predicted_word
        }
    
    except ModelNotLoadedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    except PredictionException as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e.message)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/stats", response_model=Dict)
async def get_stats() -> Dict:
    """
    Get statistics about stored swipes.
    
    Returns:
        Dictionary with statistics
    """
    try:
        return storage_service.get_stats()
    
    except StorageException as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e.message)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/train", response_model=Dict, status_code=status.HTTP_202_ACCEPTED)
async def start_training(background_tasks: BackgroundTasks) -> Dict:
    """
    Start federated learning training cycle in background.
    
    Training includes:
    1. Download global weights from server (MinIO)
    2. Train on local data
    3. Upload delta to server
    4. Hot reload model with new weights
    
    Args:
        background_tasks: FastAPI background tasks
    
    Returns:
        Response indicating training has started
    """
    try:
        # Add training task to background
        background_tasks.add_task(training_service.run_training_cycle)
        
        logger.info("FL training cycle started in background")
        
        return {
            "status": "training_started",
            "message": "Federated learning training cycle started in background"
        }
    
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )
