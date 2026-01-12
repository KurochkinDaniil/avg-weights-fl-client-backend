import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path


class SwipeLSTM(nn.Module):
    """LSTM model for swipe gesture recognition."""
    
    def __init__(self, input_size: int = 7, hidden_size: int = 512, alphabet_size: int = 40):
        """
        Initialize SwipeLSTM model.
        
        Args:
            input_size: Number of input features (x, y, dt, vx, vy, ax, ay)
            hidden_size: LSTM hidden state size
            alphabet_size: Number of output classes (characters)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, alphabet_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
        
        Returns:
            Logits of shape (seq_len, batch_size, alphabet_size)
        """
        outputs, (h_n, c_n) = self.lstm(x)
        logits = self.fc(outputs)
        return logits
    
    def save_weights(self, path: Path) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: Path) -> None:
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def get_state_dict_bytes(self) -> bytes:
        """Get state_dict as bytes for gRPC transmission."""
        import io
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getvalue()
    
    def load_state_dict_bytes(self, data: bytes) -> None:
        """Load state_dict from bytes."""
        import io
        buffer = io.BytesIO(data)
        self.load_state_dict(torch.load(buffer, map_location='cpu'))
    
    def compute_delta(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute delta between current weights and global weights.
        
        Args:
            global_weights: Global model weights
        
        Returns:
            Delta weights (local - global)
        """
        local_weights = self.state_dict()
        delta = {}
        for key in local_weights:
            delta[key] = local_weights[key] - global_weights[key]
        return delta
    
    def apply_delta(self, global_weights: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> None:
        """
        Apply delta to global weights and load into model.
        
        Args:
            global_weights: Global model weights
            delta: Delta weights to apply
        """
        new_weights = {}
        for key in global_weights:
            new_weights[key] = global_weights[key] + delta[key]
        self.load_state_dict(new_weights)
