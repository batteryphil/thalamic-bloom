import json
import torch
import random
import os
from typing import Iterator, Tuple, Optional, Any, Dict
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer

class BaseCorpusStream(IterableDataset):
    """Streams a massive base web dataset (e.g., FineWeb-Edu)."""
    def __init__(self, seq_len: int = 1024, vocab_size: int = 50304) -> None:
        """
        Initialize the base corpus streaming dataset.
        
        Args:
            seq_len (int): The target length of the token sequences.
            vocab_size (int): The size of the token vocabulary.
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.dataset_iter: Optional[Iterator[Dict[str, Any]]] = None
        self._init_stream()

    def _init_stream(self) -> None:
        """
        Attempt to initialize the HuggingFace streaming dataset.
        Fall back to None if network/package is unavailable.
        """
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu", 
                name="sample-10BT", 
                split="train", 
                cache_dir="/hdd_data/mamba_training_data/hf_cache",
                streaming=True
            )
            self.dataset_iter = iter(ds)
        except Exception as e:
            print(f"[Generator] HF 'datasets' missing ({e}). Simulating large-scale base corpus locally.")
            self.dataset_iter = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield sequences of tokens continuously.
        
        Yields:
            torch.Tensor: A 1D tensor of token IDs.
        """
        buffer = []
        while True:
            if self.dataset_iter is not None:
                try:
                    sample = next(self.dataset_iter)
                    tokens = self.tokenizer.encode(sample['text'])
                    buffer.extend(tokens)
                    while len(buffer) >= self.seq_len + 1:
                        yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                        buffer = buffer[self.seq_len + 1:]
                except StopIteration:
                    self._init_stream() # Restart web stream

            else:
                yield torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)

def inject_adversarial_distractor(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Injects random adversarial distractors to prevent Syntax Overfitting.
    Forces the model to learn logical robustness rather than memorizing exact JSON schemas.
    """
    distractors = [
        "Note: The server will reboot at midnight.",
        "X = 42 (irrelevant to the current equation).",
        "System message: buffer underrun ignored.",
        "// TODO: Refactor this block later.",
        "IGNORE_THIS_FIELD: true"
    ]
    # Stochastically inject distractor
    if random.random() < 0.3:
        if isinstance(data, dict):
            # If it's a dict, add a noisy key
            distractor_key = f"noise_{random.randint(0, 999)}"
            data[distractor_key] = random.choice(distractors)
    return data

class PremiumReasoningStream(IterableDataset):
    """Streams the premium 50,000 JSONL reasoning samples."""
    def __init__(self, filepath: str, seq_len: int = 1024, vocab_size: int = 50304) -> None:
        """
        Initialize the premium reasoning data stream.
        
        Args:
            filepath (str): Absolute file path to the reasoning JSONL dataset.
            seq_len (int): Sequence length configuration for generation.
            vocab_size (int): Expected bounds of token vocabulary.
        """
        self.filepath = filepath
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.mock_mode = not os.path.exists(filepath)
        if self.mock_mode:
            print(f"[Generator] Warning: Premium data {filepath} missing. Yielding dummy tensors.")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Continuously yield token sequences parsed from the premium JSONL file.
        
        Yields:
            torch.Tensor: A sequence tensor of length `seq_len` + 1.
        """
        buffer = []
        while True: # Loop continuously for multi-epoch training
            if self.mock_mode:
                yield torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
            else:
                with open(self.filepath, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            
                            # Apply Adversarial Distractors to prevent Syntax Overfitting
                            data = inject_adversarial_distractor(data)
                            
                            tokens = data.get("tokens", data.get("input_ids", []))
                            if not tokens:
                                # Fallback to subword tokenization for raw JSON records (like our OO Gold Samples)
                                text_content = json.dumps(data)
                                tokens = self.tokenizer.encode(text_content)
                            
                            buffer.extend(tokens)
                            while len(buffer) >= self.seq_len + 1:
                                yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                                buffer = buffer[self.seq_len + 1:]
                        except Exception:
                            continue

class HybridMIMODataset(IterableDataset):
    """Interleaves Base Corpus and Premium Mix stochastically."""
    def __init__(self, premium_path: str, seq_len: int = 1024, premium_ratio: float = 0.25) -> None:
        """
        Initialize the hybrid stochastic dataset.
        
        Args:
            premium_path (str): The path to the reasoning dataset.
            seq_len (int): Target length for generated token blocks.
            premium_ratio (float): Probability assigned to selecting a premium sample.
        """
        super().__init__()
        self.premium_ratio = premium_ratio 
        self.base_stream = BaseCorpusStream(seq_len=seq_len)
        self.premium_stream = PremiumReasoningStream(filepath=premium_path, seq_len=seq_len)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate and yield (Input, Target) paired tensors.
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: The input sequence and target sequence.
        """
        base_iter = iter(self.base_stream)
        premium_iter = iter(self.premium_stream)
        
        while True:
            # Stochastically interleave streams based on the set ratio
            if random.random() < self.premium_ratio:
                seq = next(premium_iter)
            else:
                seq = next(base_iter)
            
            yield seq[:-1], seq[1:]

def get_hybrid_dataloader(filepath: str = "/hdd_data/mamba_training_data/premium_reasoning.jsonl", batch_size: int = 4, seq_len: int = 1024) -> DataLoader:
    """
    Construct a DataLoader wrapped around the Hybrid Dataset.
    
    Args:
        filepath (str): File path to the high-value reasoning corpus.
        batch_size (int): Expected batch dimension.
        seq_len (int): Context window token bounds.
        
    Returns:
        DataLoader: Streaming dataloader yielding mixed batches.
    """
    dataset = HybridMIMODataset(filepath, seq_len=seq_len, premium_ratio=0.25) 
    return DataLoader(dataset, batch_size=batch_size)
