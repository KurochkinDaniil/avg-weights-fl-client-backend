import torch
import numpy as np
from typing import List, Dict
from pathlib import Path

from .model import SwipeLSTM
from .preprocessing import preprocess_swipe


class SwipePredictor:
    """Predictor for swipe gesture recognition."""
    
    def __init__(
        self, 
        model_path: Path, 
        char_mappings: tuple, 
        device: str = "cpu",
        keyboard_width: float = 1080.0,
        keyboard_height: float = 631.0
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model weights
            char_mappings: Tuple of (char2idx, idx2char)
            device: Device to run inference on
            keyboard_width: Keyboard width in pixels (default: 1080)
            keyboard_height: Keyboard height in pixels (default: 631)
        """
        self.device = torch.device(device)
        self.char2idx, self.idx2char = char_mappings
        self.keyboard_width = keyboard_width
        self.keyboard_height = keyboard_height
        
        self.model = SwipeLSTM(
            input_size=7,
            hidden_size=512,
            alphabet_size=len(self.idx2char)
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, coords: List[Dict[str, float]]) -> str:
        """
        Predict word from swipe coordinates.
        
        Args:
            coords: List of coordinate dicts [{"x": 342.3, "y": 263.1, "t": 0.0}, ...]
                    where x, y are in PIXELS
        
        Returns:
            Predicted word
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Validation
        if len(coords) < 2:
            raise ValueError("Need at least 2 points for prediction")
        
        logger.info(f"Received {len(coords)} points")
        logger.info(f"First point: {coords[0] if coords else 'none'}")
        logger.info(f"Last point: {coords[-1] if coords else 'none'}")
        
        features = preprocess_swipe(coords, self.keyboard_width, self.keyboard_height)
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features min/max: {features.min():.3f}/{features.max():.3f}")
        
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        
        pred_ids = self._ctc_greedy_decode(log_probs)
        logger.info(f"Predicted indices: {pred_ids[0][:10]}")
        
        # FIXED: Filter out special tokens and handle space
        skip_tokens = {'_', 'shift', 'backspace', 'toNumberState', 'globe', 'enter'}
        
        decoded_chars = []
        for i in pred_ids[0]:
            char = self.idx2char.get(i, '')
            if char and char not in skip_tokens:
                # Convert 'space' token to actual space
                decoded_chars.append(' ' if char == 'space' else char)
        
        pred_word = ''.join(decoded_chars).strip()
        
        return pred_word
    
    def _ctc_greedy_decode(self, log_probs: torch.Tensor, blank: int = 0) -> List[List[int]]:
        """
        CTC greedy decoding.
        
        Args:
            log_probs: Log probabilities (T, B, C)
            blank: Blank token index
        
        Returns:
            List of decoded sequences
        """
        T, B, C = log_probs.shape
        preds = log_probs.argmax(dim=-1)
        
        results = []
        for b in range(B):
            seq = []
            prev = blank
            
            for t in range(T):
                cur = preds[t, b].item()
                if cur != blank and cur != prev:
                    seq.append(cur)
                prev = cur
            
            results.append(seq)
        
        return results
