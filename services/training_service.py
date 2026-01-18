"""Training service for federated learning."""
import logging
import torch
from typing import Optional, Dict
from pathlib import Path

from ml.model import SwipeLSTM
from ml.trainer import Trainer
from ml.dataset import SwipeDataset
from ml.preprocessing import build_char_mappings
from grpc_client.fl_client import FederatedLearningClient
from core.model_manager import model_manager
from core.exceptions import TrainingException, ServerConnectionException
from config import settings
from .storage_service import StorageService

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for federated learning training cycle."""
    
    def __init__(self):
        """Initialize training service."""
        self._storage_service = StorageService()
    
    async def run_training_cycle(self) -> Dict:
        """
        Run one federated learning cycle.
        
        Steps:
        1. Download global weights from server (MinIO)
        2. Load local training data
        3. Train model locally
        4. Compute delta
        5. Upload delta to server
        6. Hot reload model with new weights
        
        Returns:
            Training results dict with metrics
            
        Raises:
            TrainingException: If training fails
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting FL Training Cycle")
            logger.info("=" * 60)
            
            results = {
                "status": "started",
                "steps_completed": [],
                "metrics": {}
            }
            
            # Step 1: Download global weights
            logger.info("[Step 1/5] Downloading global weights from server...")
            global_weights = await self._download_global_weights()
            results["steps_completed"].append("download_weights")
            
            # Step 2: Load local data
            logger.info("[Step 2/5] Loading local training data...")
            dataset = self._load_training_data()
            
            if len(dataset) == 0:
                raise TrainingException("No training data available")
            
            results["metrics"]["num_samples"] = len(dataset)
            results["steps_completed"].append("load_data")
            
            # Step 3: Train model
            logger.info(f"[Step 3/5] Training on {len(dataset)} samples...")
            trained_weights, num_examples = await self._train_model(
                dataset, global_weights
            )
            results["metrics"]["num_examples"] = num_examples
            results["steps_completed"].append("train_model")
            
            # Step 4: Compute delta
            logger.info("[Step 4/5] Computing weight delta...")
            model = model_manager.get_model()
            delta = model.compute_delta(global_weights)
            results["steps_completed"].append("compute_delta")
            
            # Step 5: Upload delta
            logger.info("[Step 5/5] Uploading delta to server...")
            upload_success = await self._upload_delta(delta, num_examples)
            
            if upload_success:
                results["steps_completed"].append("upload_delta")
                logger.info("✓ Delta uploaded successfully")
            else:
                logger.warning("⚠ Delta upload failed (server unavailable)")
            
            # Step 6: Hot reload model
            logger.info("[Step 6/6] Hot reloading model...")
            model_manager.reload_from_weights(trained_weights)
            results["steps_completed"].append("hot_reload")
            
            results["status"] = "completed"
            logger.info("=" * 60)
            logger.info("FL Training Cycle Completed Successfully")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Training cycle failed: {e}")
            raise TrainingException(str(e))
    
    async def _download_global_weights(self) -> Dict[str, torch.Tensor]:
        """
        Download global weights from FL server (MinIO).
        
        Returns:
            Global weights state dict
        """
        try:
            with FederatedLearningClient(
                settings.server_grpc_url,
                settings.client_id
            ) as fl_client:
                global_weights = fl_client.download_global_weights()
                
                if global_weights:
                    logger.info("✓ Downloaded global weights from server")
                    return global_weights
        
        except Exception as e:
            logger.warning(f"Cannot connect to server: {e}")
        
        # Fallback: use current model weights
        logger.info("Using current model weights as baseline")
        model = model_manager.get_model()
        return model.state_dict()
    
    def _load_training_data(self) -> SwipeDataset:
        """
        Load training data from local storage.
        
        Returns:
            PyTorch dataset with swipe data
        """
        jsonl_files = self._storage_service.get_all_jsonl_files()
        
        if not jsonl_files:
            logger.warning("No local data found")
            return SwipeDataset(
                jsonl_files=[],
                char2idx={},
                max_length=settings.max_seq_len,
                keyboard_width=1080.0,
                keyboard_height=631.0
            )
        
        char2idx, _ = build_char_mappings(settings.alphabet)
        
        dataset = SwipeDataset(
            jsonl_files=jsonl_files,
            char2idx=char2idx,
            max_length=settings.max_seq_len,
            keyboard_width=1080.0,
            keyboard_height=631.0
        )
        
        logger.info(f"✓ Loaded {len(dataset)} training samples from {len(jsonl_files)} files")
        return dataset
    
    async def _train_model(
        self,
        dataset: SwipeDataset,
        global_weights: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Train model on local data.
        
        Args:
            dataset: Training dataset
            global_weights: Initial weights from server
            
        Returns:
            Tuple of (trained_weights, num_examples)
        """
        # Get current model and load global weights
        model = model_manager.get_model()
        model.load_state_dict({
            k: v.to(model_manager.device) for k, v in global_weights.items()
        })
        
        # Train
        trainer = Trainer(
            model=model,
            device=model_manager.device,
            learning_rate=settings.learning_rate,
            max_seq_len=settings.max_seq_len
        )
        
        trained_weights, num_examples = trainer.train(
            dataset=dataset,
            batch_size=settings.batch_size,
            num_epochs=settings.num_epochs
        )
        
        logger.info(f"✓ Training completed on {num_examples} examples")
        return trained_weights, num_examples
    
    async def _upload_delta(
        self,
        delta: Dict[str, torch.Tensor],
        num_examples: int
    ) -> bool:
        """
        Upload delta to FL server.
        
        Args:
            delta: Weight delta
            num_examples: Number of training examples
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            with FederatedLearningClient(
                settings.server_grpc_url,
                settings.client_id
            ) as fl_client:
                success = fl_client.upload_weights(delta, num_examples)
                return success
        
        except Exception as e:
            logger.warning(f"Upload failed: {e}")
            return False

