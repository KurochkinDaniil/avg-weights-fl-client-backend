"""API routes for swipe gesture collection."""
from fastapi import APIRouter, HTTPException, status
from typing import Dict
import logging
from pathlib import Path

from api.models import SwipeRequest, SwipeResponse
from storage.local_storage import LocalStorage
from config import settings
from ml.inference import SwipePredictor
from ml.preprocessing import build_char_mappings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["swipes"])

# Initialize storage
storage = LocalStorage(settings.data_dir)

# Initialize predictor
char2idx, idx2char = build_char_mappings(settings.alphabet)
model_path = Path(__file__).parent.parent / "model2.pt"

try:
    predictor = SwipePredictor(model_path, (char2idx, idx2char))
    logger.info(f"Loaded model from {model_path}")
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    predictor = None


@router.post("/swipes", response_model=SwipeResponse, status_code=status.HTTP_201_CREATED)
async def receive_swipe(swipe: SwipeRequest) -> SwipeResponse:
    """
    Receive and store swipe gesture from frontend.
    
    Args:
        swipe: Swipe gesture data
    
    Returns:
        Response with status and gesture ID
    """
    try:
        # Convert to dict for storage
        swipe_data = {
            "gesture_id": swipe.gesture_id,
            "coords": [{"x": p.x, "y": p.y, "t": p.t} for p in swipe.coords],
            "word": swipe.word
        }
        
        # Save to local storage
        storage.save_swipe(swipe_data)
        
        logger.info(f"Received swipe: {swipe.gesture_id}, word: {swipe.word}, points: {len(swipe.coords)}")
        
        return SwipeResponse(
            status="success",
            gesture_id=swipe.gesture_id,
            message=f"Swipe gesture saved successfully"
        )
    
    except Exception as e:
        logger.error(f"Error saving swipe: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save swipe gesture: {str(e)}"
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
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        coords = [{"x": p.x, "y": p.y, "t": p.t} for p in swipe.coords]
        predicted_word = predictor.predict(coords)
        
        logger.info(f"Predicted: {predicted_word} for gesture {swipe.gesture_id}")
        
        return {
            "gesture_id": swipe.gesture_id,
            "predicted_word": predicted_word
        }
    
    except Exception as e:
        logger.error(f"Error predicting swipe: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/stats", response_model=Dict)
async def get_stats() -> Dict:
    """
    Get statistics about stored swipes.
    
    Returns:
        Dictionary with statistics
    """
    try:
        total_samples = storage.count_samples()
        jsonl_files = storage.get_all_jsonl_files()
        
        return {
            "total_samples": total_samples,
            "total_files": len(jsonl_files),
            "files": [str(f.relative_to(settings.data_dir)) for f in jsonl_files]
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
