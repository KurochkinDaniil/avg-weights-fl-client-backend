"""Local storage for swipe gestures."""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class LocalStorage:
    """Local storage for swipe gestures in JSONL format."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize LocalStorage.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def save_swipe(self, swipe_data: Dict) -> None:
        """
        Save swipe gesture to JSONL file.
        
        Args:
            swipe_data: Swipe data dict with keys: gesture_id, coords, word
        """
        # Create date-based directory
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.raw_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL file
        jsonl_file = day_dir / "swipes.jsonl"
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            json.dump(swipe_data, f, ensure_ascii=False)
            f.write('\n')
        
        logger.info(f"Saved swipe {swipe_data.get('gesture_id')} to {jsonl_file}")
    
    def get_all_jsonl_files(self) -> List[Path]:
        """
        Get all JSONL files in the raw directory.
        
        Returns:
            List of JSONL file paths
        """
        jsonl_files = list(self.raw_dir.glob("*/swipes.jsonl"))
        return sorted(jsonl_files)
    
    def get_recent_jsonl_files(self, days: int = 7) -> List[Path]:
        """
        Get JSONL files from recent days.
        
        Args:
            days: Number of recent days to include
        
        Returns:
            List of JSONL file paths
        """
        all_files = self.get_all_jsonl_files()
        
        # Filter by date (simple approach: take last N files)
        return all_files[-days:] if len(all_files) > days else all_files
    
    def count_samples(self) -> int:
        """
        Count total number of samples across all JSONL files.
        
        Returns:
            Total number of samples
        """
        total = 0
        for jsonl_file in self.get_all_jsonl_files():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                total += sum(1 for line in f if line.strip())
        return total
