import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from .preprocessing import preprocess_swipe, word_to_indices


class SwipeDataset(Dataset):
    """Dataset for swipe gesture recognition."""
    
    def __init__(
        self,
        jsonl_files: List[Path],
        char2idx: Dict[str, int],
        max_length: int = 300
    ):
        """
        Initialize SwipeDataset.
        
        Args:
            jsonl_files: List of JSONL file paths
            char2idx: Character to index mapping
            max_length: Maximum sequence length (padding/truncation)
        """
        self.max_length = max_length
        self.char2idx = char2idx
        self.samples = []
        
        # Load all samples from JSONL files
        for jsonl_file in jsonl_files:
            if jsonl_file.exists():
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (sequence, label, sequence_length)
        """
        sample = self.samples[idx]
        
        # Preprocess coordinates
        features = preprocess_swipe(sample["coords"])
        seq = torch.tensor(features, dtype=torch.float32)
        
        # Convert word to indices
        label = word_to_indices(sample["word"], self.char2idx)
        
        return seq, label, seq.size(0)


def pad_collate(batch: List[Tuple], max_length: int = 300) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader with padding.
    
    Args:
        batch: List of (sequence, label, length) tuples
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (padded_sequences, seq_lengths, labels_flat, label_lengths)
    """
    sequences, labels, lengths = zip(*batch)
    
    # Pad sequences
    padded = []
    seqlens = []
    for seq in sequences:
        seqlens.append(min(len(seq), max_length))
        if seq.size(0) < max_length:
            # Pad with zeros
            pad = F.pad(seq, (0, 0, 0, max_length - seq.size(0)))
        else:
            # Truncate
            pad = seq[:max_length]
        padded.append(pad)
    
    padded = torch.stack(padded)  # (batch_size, seq_len, features)
    seqlens = torch.tensor(seqlens, dtype=torch.long)
    
    # Flatten labels for CTC loss
    labelslen = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labelsflat = []
    for label in labels:
        labelsflat.extend(label)
    labels_tensor = torch.tensor(labelsflat, dtype=torch.long)
    
    return padded, seqlens, labels_tensor, labelslen
