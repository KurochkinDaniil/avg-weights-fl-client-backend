import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler


def preprocess_swipe(
    coords: List[Dict[str, float]], 
    keyboard_width: float = 1080.0, 
    keyboard_height: float = 631.0
) -> np.ndarray:
    """
    Preprocess swipe coordinates into model features.
    
    Updated version from colleague:
    - Uses pandas for cleaner data manipulation
    - Uses MinMaxScaler for normalization of x, y, dt ONLY
    - Velocities and accelerations are NOT normalized (only clipped)
    
    Converts pixel coordinates into features:
    - x, y, dt: normalized (0..1) using MinMaxScaler
    - vx, vy: velocity in pixels/time, clipped to [-10, 10]
    - ax, ay: acceleration in pixels/time², clipped to [-10, 10]
    
    Args:
        coords: List of coordinate dicts [{"x": 342.3, "y": 263.1, "t": 0.0}, ...]
                where x, y are in PIXELS (x: 0-1080, y: 0-631)
        keyboard_width: Keyboard width in pixels (default: 1080, not used in new version)
        keyboard_height: Keyboard height in pixels (default: 631, not used in new version)
    
    Returns:
        Feature array of shape (seq_len, 7) with columns [x, y, dt, vx, vy, ax, ay]
    """
    if not coords:
        raise ValueError("Empty coordinates list")
    
    df = pd.DataFrame(coords)

    # Вычисляем dt (разницу во времени)
    df['dt'] = df['t'].diff()
    df.loc[0, 'dt'] = 0  # первое значение
    df['dt'] = df['dt'].replace(0, 1e-8)  # избегаем деления на ноль

    # Вычисляем скорость
    df['vx'] = df['x'].diff() / df['dt']
    df['vy'] = df['y'].diff() / df['dt']

    # Вычисляем ускорение
    df['ax'] = df['vx'].diff() / df['dt']
    df['ay'] = df['vy'].diff() / df['dt']

    # Заменяем NaN значения (первые строки после diff)
    df.fillna(0, inplace=True)

    # Опционально: обрезаем экстремальные значения
    clip_limit = 10
    df['vx'] = df['vx'].clip(-clip_limit, clip_limit)
    df['vy'] = df['vy'].clip(-clip_limit, clip_limit)
    df['ax'] = df['ax'].clip(-clip_limit, clip_limit)
    df['ay'] = df['ay'].clip(-clip_limit, clip_limit)

    # Нормализация отдельно по каждой оси (ТОЛЬКО x, y, dt!)
    scaler = MinMaxScaler()
    df[['x', 'y', 'dt']] = scaler.fit_transform(df[['x', 'y', 'dt']])
    df = df.drop(columns=['t'])

    # Создаем DataFrame с признаками
    features = np.column_stack([
        df['x'].values,
        df['y'].values,
        df['dt'].values,
        df['vx'].values,
        df['vy'].values,
        df['ax'].values,
        df['ay'].values
    ])

    return features


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
