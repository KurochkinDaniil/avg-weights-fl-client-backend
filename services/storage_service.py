"""Storage service for managing swipe data."""
import logging
from typing import List, Dict
from pathlib import Path

from storage.local_storage import LocalStorage
from config import settings
from core.exceptions import StorageException

logger = logging.getLogger(__name__)


class StorageService:
    """Service for storing and retrieving swipe data."""
    
    def __init__(self):
        """Initialize storage service."""
        self._storage = LocalStorage(settings.data_dir)
    
    def save_swipe(
        self,
        gesture_id: str,
        coords: List[Dict[str, float]],
        word: str
    ) -> None:
        """
        Save swipe gesture to storage.
        
        Args:
            gesture_id: Unique gesture ID
            coords: List of coordinate points
            word: Ground truth word
            
        Raises:
            StorageException: If save fails
        """
        try:
            logger.info(f"Saving swipe: gesture_id={gesture_id}, word='{word}'")
            
            self._storage.save_swipe(
                gesture_id=gesture_id,
                coords=coords,
                word=word
            )
            
            logger.info(f"✓ Swipe saved: {gesture_id}")
            
        except Exception as e:
            logger.error(f"Failed to save swipe: {e}")
            raise StorageException(f"Failed to save swipe: {e}")
    
    def get_all_jsonl_files(self) -> List[Path]:
        """
        Get all JSONL files with swipe data.
        
        Returns:
            List of JSONL file paths
            
        Raises:
            StorageException: If retrieval fails
        """
        try:
            files = self._storage.get_all_jsonl_files()
            logger.info(f"Found {len(files)} JSONL files")
            return files
            
        except Exception as e:
            logger.error(f"Failed to get JSONL files: {e}")
            raise StorageException(f"Failed to get JSONL files: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with stats (total_swipes, files, etc.)
        """
        try:
            jsonl_files = self.get_all_jsonl_files()
            
            total_swipes = 0
            for file_path in jsonl_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_swipes += sum(1 for _ in f)
            
            stats = {
                "total_swipes": total_swipes,
                "total_files": len(jsonl_files),
                "data_directory": str(settings.data_dir)
            }
            
            logger.info(f"Storage stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise StorageException(f"Failed to get stats: {e}")

