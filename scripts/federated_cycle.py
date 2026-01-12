import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from typing import Optional

from config import settings
from ml.model import SwipeLSTM
from ml.dataset import SwipeDataset
from ml.trainer import Trainer
from ml.preprocessing import build_char_mappings
from storage.local_storage import LocalStorage
from grpc_client.fl_client import FederatedLearningClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get available device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS создаёт проблемы с CTC Loss - используем CPU
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')


def run_federated_cycle():
    """
    Run one federated learning cycle:
    1. Download global weights from server
    2. Load local data
    3. Train model locally
    4. Compute delta
    5. Upload delta to server
    """
    logger.info("=" * 60)
    logger.info("Starting Federated Learning Cycle")
    logger.info("=" * 60)
    
    # Initialize components
    device = get_device()
    logger.info(f"Using device: {device}")
    
    storage = LocalStorage(settings.data_dir)
    char2idx, idx2char = build_char_mappings(settings.alphabet)
    
    # Initialize model
    model = SwipeLSTM(
        input_size=settings.input_size,
        hidden_size=settings.hidden_size,
        alphabet_size=settings.alphabet_size
    )
    
    # Step 1: Download global weights
    logger.info("\n[Step 1/5] Downloading global weights from server...")
    global_weights: Optional[dict] = None
    
    try:
        with FederatedLearningClient(settings.server_grpc_url, settings.client_id) as fl_client:
            global_weights = fl_client.download_global_weights()
    except Exception as e:
        logger.warning(f"Could not connect to server: {e}")
        logger.info("Will train from scratch (no global weights)")
    
    if global_weights:
        model.load_state_dict(global_weights)
        logger.info("✓ Loaded global weights")
    else:
        logger.info("✓ Using initial random weights")
        global_weights = model.state_dict()  # Use initial weights as baseline
    
    # Step 2: Load local data
    logger.info("\n[Step 2/5] Loading local data...")
    jsonl_files = storage.get_all_jsonl_files()
    
    if not jsonl_files:
        logger.error("✗ No local data found. Please collect some swipes first.")
        return
    
    dataset = SwipeDataset(
        jsonl_files=jsonl_files,
        char2idx=char2idx,
        max_length=settings.max_seq_len
    )
    
    logger.info(f"✓ Loaded {len(dataset)} samples from {len(jsonl_files)} files")
    
    if len(dataset) == 0:
        logger.error("✗ Dataset is empty")
        return
    
    # Step 3: Train model locally
    logger.info("\n[Step 3/5] Training model on local data...")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=settings.learning_rate,
        max_seq_len=settings.max_seq_len
    )
    
    trained_weights, num_examples = trainer.train(
        dataset=dataset,
        batch_size=settings.batch_size,
        num_epochs=settings.num_epochs
    )
    
    logger.info(f"✓ Training completed on {num_examples} examples")
    
    # Step 4: Compute delta
    logger.info("\n[Step 4/5] Computing weight delta...")
    delta = model.compute_delta(global_weights)
    logger.info(f"✓ Computed delta with {len(delta)} layers")
    
    # Step 5: Upload delta to server
    logger.info("\n[Step 5/5] Uploading delta to server...")
    
    try:
        with FederatedLearningClient(settings.server_grpc_url, settings.client_id) as fl_client:
            success = fl_client.upload_weights(delta, num_examples)
            
            if success:
                logger.info("✓ Successfully uploaded delta to server")
            else:
                logger.error("✗ Failed to upload delta")
    except Exception as e:
        logger.error(f"✗ Error uploading to server: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Federated Learning Cycle Completed")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        run_federated_cycle()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error in federated cycle: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
