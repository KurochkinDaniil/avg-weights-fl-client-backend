import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import logging

from .model import SwipeLSTM
from .dataset import SwipeDataset, pad_collate

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for SwipeLSTM model."""
    
    def __init__(
        self,
        model: SwipeLSTM,
        device: torch.device,
        learning_rate: float = 0.001,
        max_seq_len: int = 300
    ):
        """
        Initialize Trainer.
        
        Args:
            model: SwipeLSTM model
            device: Device to train on (cpu/cuda/mps)
            learning_rate: Learning rate for optimizer
            max_seq_len: Maximum sequence length
        """
        self.model = model.to(device)
        self.device = device
        self.max_seq_len = max_seq_len
        
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(
        self,
        dataset: SwipeDataset,
        batch_size: int = 32,
        num_epochs: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Train model on local dataset.
        
        Args:
            dataset: SwipeDataset
            batch_size: Batch size
            num_epochs: Number of training epochs
        
        Returns:
            Tuple of (trained_weights, num_examples)
        """
        if len(dataset) == 0:
            logger.warning("Empty dataset, skipping training")
            return self.model.state_dict(), 0
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: pad_collate(batch, max_length=self.max_seq_len)
        )
        
        self.model.train()
        num_examples = len(dataset)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, input_lengths, targets, target_lengths) in enumerate(dataloader):
                # Move to device
                inputs = inputs.to(self.device)
                input_lengths = input_lengths.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Transpose for LSTM: (batch, seq, features) -> (seq, batch, features)
                inputs = inputs.transpose(0, 1)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(inputs)  # (seq, batch, alphabet_size)
                
                # Log softmax for CTC loss
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                
                # CTC loss
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Training completed on {num_examples} examples")
        return self.model.state_dict(), num_examples
