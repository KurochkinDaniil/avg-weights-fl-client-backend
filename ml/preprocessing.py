import numpy as np
from typing import List, Dict


def preprocess_swipe(coords: List[Dict[str, float]]) -> np.ndarray:
    """
    Preprocess swipe coordinates into model features.
    
    Converts raw coordinates into features:
    - x, y: normalized coordinates (0..1)
    - dt: time delta
    - vx, vy: velocity
    - ax, ay: acceleration
    
    Args:
        coords: List of coordinate dicts [{"x": 0.3, "y": 0.4, "t": 0.0}, ...]
    
    Returns:
        Feature array of shape (seq_len, 7) with columns [x, y, dt, vx, vy, ax, ay]
    """
    if not coords:
        raise ValueError("Empty coordinates list")
    
    # Extract arrays
    x = np.array([p["x"] for p in coords], dtype=np.float32)
    y = np.array([p["y"] for p in coords], dtype=np.float32)
    t = np.array([p["t"] for p in coords], dtype=np.float32)
    
    # Compute dt (time delta)
    dt = np.diff(t, prepend=t[0])
    
    # Compute velocity with proper normalization
    dx = np.gradient(x)
    dy = np.gradient(y)
    dt_grad = np.gradient(t)
    dt_grad = np.where(dt_grad == 0, 1e-8, dt_grad)  # avoid division by zero
    
    vx = dx / dt_grad
    vy = dy / dt_grad
    
    # Compute acceleration
    ax = np.gradient(vx) / dt_grad
    ay = np.gradient(vy) / dt_grad
    
    # Clip extreme values to match training data range
    vx = np.clip(vx, -10, 10)
    vy = np.clip(vy, -10, 10)
    ax = np.clip(ax, -10, 10)
    ay = np.clip(ay, -10, 10)
    
    # Stack features
    features = np.column_stack([x, y, dt, vx, vy, ax, ay])
    
    return features.astype(np.float32)


def word_to_indices(word: str, char2idx: Dict[str, int]) -> List[int]:
    """
    Convert word to list of character indices.
    
    Args:
        word: Target word
        char2idx: Character to index mapping
    
    Returns:
        List of character indices
    """
    return [char2idx[c] for c in word if c in char2idx]


def build_char_mappings(alphabet: str) -> tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character to index and index to character mappings.
    
    Args:
        alphabet: String with tokens separated by '|' (e.g., "_|й|ц|shift|space")
    
    Returns:
        Tuple of (char2idx, idx2char) dictionaries
    """
    # Split by pipe delimiter to handle multi-char tokens properly
    chars = alphabet.split('|')
    
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    
    return char2idx, idx2char
