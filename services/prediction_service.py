"""Prediction service for swipe gesture recognition."""
import logging
from typing import List, Dict

from core.model_manager import model_manager
from core.exceptions import PredictionException, ModelNotLoadedException

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions on swipe gestures."""
    
    def predict(self, coords: List[Dict[str, float]]) -> str:
        """
        Predict word from swipe coordinates.
        
        Args:
            coords: List of coordinate points [{x, y, t}, ...]
            
        Returns:
            Predicted word
            
        Raises:
            PredictionException: If prediction fails
            ModelNotLoadedException: If model not loaded
        """
        try:
            predictor = model_manager.get_predictor()
            
            logger.info(f"Predicting swipe with {len(coords)} points")
            predicted_word = predictor.predict(coords)
            
            logger.info(f"Predicted: '{predicted_word}'")
            return predicted_word
            
        except ModelNotLoadedException:
            logger.error("Model not loaded")
            raise
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise PredictionException(str(e))

