"""Model Manager for loading and hot-reloading models."""
import logging
import torch
from pathlib import Path
from typing import Optional, Dict
from threading import Lock

from ml.model import SwipeLSTM
from ml.inference import SwipePredictor
from ml.preprocessing import build_char_mappings
from config import settings
from .exceptions import ModelNotLoadedException, ModelDownloadException

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for ML model with hot reload support.
    
    Features:
    - Thread-safe model loading
    - Hot reload from local file or MinIO
    - Automatic device selection (CPU/GPU)
    """
    
    _instance: Optional['ModelManager'] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model manager (called once)."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._model: Optional[SwipeLSTM] = None
        self._predictor: Optional[SwipePredictor] = None
        self._device = self._get_device()
        self._char_mappings = build_char_mappings(settings.alphabet)
        
        logger.info(f"ModelManager initialized on device: {self._device}")
    
    @staticmethod
    def _get_device() -> torch.device:
        """Get available device (cuda/cpu)."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def load_model(self, model_path: Path = Path("model2.pt")) -> None:
        """
        Load model from local file.
        
        Args:
            model_path: Path to model weights file
            
        Raises:
            ModelNotLoadedException: If model cannot be loaded
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Create model
            self._model = SwipeLSTM(
                input_size=settings.input_size,
                hidden_size=settings.hidden_size,
                alphabet_size=settings.alphabet_size
            )
            
            # Load weights
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                self._model.to(self._device)
                self._model.eval()
                logger.info(f"✓ Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}, using random weights")
                self._model.to(self._device)
                self._model.eval()
            
            # Create predictor
            char2idx, idx2char = self._char_mappings
            self._predictor = SwipePredictor(
                model_path=model_path,
                char_mappings=(char2idx, idx2char),
                device=str(self._device),
                keyboard_width=1080.0,
                keyboard_height=631.0
            )
            
            logger.info("✓ Predictor initialized")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelNotLoadedException()
    
    def reload_from_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        Hot reload model from state dict (e.g., from MinIO).
        
        Args:
            weights: Model state dict
            
        This is thread-safe and allows updating the model without downtime.
        """
        with self._lock:
            try:
                logger.info("Hot reloading model from new weights")
                
                if self._model is None:
                    # Initialize model first time
                    self._model = SwipeLSTM(
                        input_size=settings.input_size,
                        hidden_size=settings.hidden_size,
                        alphabet_size=settings.alphabet_size
                    )
                
                # Move weights to correct device
                weights_on_device = {
                    k: v.to(self._device) for k, v in weights.items()
                }
                
                # Load new weights
                self._model.load_state_dict(weights_on_device)
                self._model.eval()
                
                # Update predictor's model
                if self._predictor:
                    self._predictor.model = self._model
                
                logger.info("✓ Model hot reloaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to reload model: {e}")
                raise ModelNotLoadedException()
    
    def get_model(self) -> SwipeLSTM:
        """
        Get current model instance.
        
        Returns:
            Current model
            
        Raises:
            ModelNotLoadedException: If model not loaded
        """
        if self._model is None:
            raise ModelNotLoadedException()
        return self._model
    
    def get_predictor(self) -> SwipePredictor:
        """
        Get current predictor instance.
        
        Returns:
            Current predictor
            
        Raises:
            ModelNotLoadedException: If predictor not loaded
        """
        if self._predictor is None:
            raise ModelNotLoadedException()
        return self._predictor
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._predictor is not None


# Global singleton instance
model_manager = ModelManager()

