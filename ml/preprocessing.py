import numpy as np
from typing import List, Dict


def preprocess_swipe(
    coords: List[Dict[str, float]], 
    keyboard_width: float = 1080.0, 
    keyboard_height: float = 631.0
) -> np.ndarray:
    """
    Preprocess swipe coordinates into model features.
    
    Compute derivatives from PIXEL coordinates, then normalize all features.
    This matches the approach where velocities/accelerations are computed from raw pixels.
    
    Converts pixel coordinates into normalized features:
    - x, y: normalized coordinates (0..1)
    - dt: time delta
    - vx, vy: normalized velocity
    - ax, ay: normalized acceleration
    
    Args:
        coords: List of coordinate dicts [{"x": 342.3, "y": 263.1, "t": 0.0}, ...]
                where x, y are in PIXELS (x: 0-1080, y: 0-631)
        keyboard_width: Keyboard width in pixels (default: 1080)
        keyboard_height: Keyboard height in pixels (default: 631)
    
    Returns:
        Feature array of shape (seq_len, 7) with columns [x, y, dt, vx, vy, ax, ay]
    """
    if not coords:
        raise ValueError("Empty coordinates list")
    
    # Extract PIXEL arrays
    x_pixel = np.array([p["x"] for p in coords], dtype=np.float32)
    y_pixel = np.array([p["y"] for p in coords], dtype=np.float32)
    t = np.array([p["t"] for p in coords], dtype=np.float32)
    
    # Compute dt (time delta)
    dt = np.diff(t, prepend=t[0])
    
    # Compute velocity FROM PIXELS
    dx_pixel = np.gradient(x_pixel)
    dy_pixel = np.gradient(y_pixel)
    dt_grad = np.gradient(t)
    dt_grad = np.where(dt_grad == 0, 1e-8, dt_grad)  # avoid division by zero
    
    vx_pixel = dx_pixel / dt_grad  # pixels/sec
    vy_pixel = dy_pixel / dt_grad  # pixels/sec
    
    # Compute acceleration FROM PIXEL VELOCITIES
    ax_pixel = np.gradient(vx_pixel) / dt_grad  # pixels/sec²
    ay_pixel = np.gradient(vy_pixel) / dt_grad  # pixels/sec²
    
    # NOW NORMALIZE EVERYTHING
    x = x_pixel / keyboard_width      # 0-1
    y = y_pixel / keyboard_height     # 0-1
    vx = vx_pixel / keyboard_width    # normalized_units/sec
    vy = vy_pixel / keyboard_height   # normalized_units/sec
    ax = ax_pixel / keyboard_width    # normalized_units/sec²
    ay = ay_pixel / keyboard_height   # normalized_units/sec²
    
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
