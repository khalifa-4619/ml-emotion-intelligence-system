"""
Data loading and preprocessing for emotion classification.

Single entry point for dataset access across training, evaluation, and inference.
All functions are independently testable. prepare_dataset() provides a convenience
facade for the full pipeline.
"""
from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizerFast
from typing import Tuple, Dict, List, Any, Optional
import torch

# ============================================================
# Model Configuration
# Will migrate to config/ when consumed by multiple modules.
# Currently only 2 consumers (training script, evaluation notebook).
# ============================================================
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 128

# Module-level cache for dataset metadata.
# Avoids repeated network calls after first load.
_LABEL_NAMES: Optional[List[str]] = None
_NUM_LABELS: Optional[int] = None


def load_emotion_dataset(use_official_splits: bool = True) -> DatasetDict:
    """
    Load the dair-ai/emotion dataset.
    
    Args:
        use_official_splits: If True, use dataset's official train/validation/test splits.
                           If False, create validation split from training data (80/20).
    
    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
        Labels are integer-encoded (0-5 by default).
    
    Raises:
        RuntimeError: If dataset cannot be loaded (network issue, corrupt cache).
        ValueError: If expected splits are missing from the dataset.
    """
    try:
        dataset = load_dataset("dair-ai/emotion")
    except Exception as e:
        raise RuntimeError(
            "Failed to load dair-ai/emotion dataset. "
            "Verify network connectivity and dataset cache at ~/.cache/huggingface/datasets/"
        ) from e
    
    if use_official_splits:
        expected_splits = {'train', 'validation', 'test'}
        actual_splits = set(dataset.keys())
        if not expected_splits.issubset(actual_splits):
            raise ValueError(
                f"Dataset structure unexpected. "
                f"Expected splits: {expected_splits}, Found: {actual_splits}"
            )
        return dataset
    
    # Custom validation split from training data
    train_val_split = dataset['train'].train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column='label'
    )
    
    return DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': dataset['test']
    })


def get_label_names(dataset: Optional[DatasetDict] = None) -> List[str]:
    """
    Get label names from dataset metadata (authoritative source).
    
    Queries the dataset's feature definitions rather than using hardcoded values.
    This prevents silent bugs if the dataset label ordering or names change.
    
    Args:
        dataset: Optional DatasetDict. If None, loads a minimal dataset slice.
    
    Returns:
        List of label name strings in index order.
    """
    global _LABEL_NAMES
    if _LABEL_NAMES is not None:
        return _LABEL_NAMES
    
    if dataset is None:
        raw_dataset = load_dataset("dair-ai/emotion", split='train')
    elif hasattr(dataset, 'features'):
        # Raw dataset (single split) passed directly
        raw_dataset = dataset
    else:
        # DatasetDict passed - extract train split
        raw_dataset = dataset['train']
    
    _LABEL_NAMES = raw_dataset.features['label'].names
    return _LABEL_NAMES


def get_num_labels(dataset: Optional[DatasetDict] = None) -> int:
    """Get the number of unique emotion labels from dataset metadata."""
    global _NUM_LABELS
    if _NUM_LABELS is not None:
        return _NUM_LABELS
    _NUM_LABELS = len(get_label_names(dataset))
    return _NUM_LABELS


def get_tokenizer() -> DistilBertTokenizerFast:
    """
    Get the DistilBERT tokenizer instance.
    
    Hugging Face handles caching automatically - repeated calls
    return the same cached tokenizer.
    """
    return DistilBertTokenizerFast.from_pretrained(TOKENIZER_NAME)


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: DistilBertTokenizerFast
) -> DatasetDict:
    """
    Tokenize text field without padding.
    
    Padding is deferred to DataCollatorWithPadding during training,
    which dynamically pads to the longest sequence in each batch.
    This is more efficient than padding all sequences to MAX_SEQUENCE_LENGTH.
    
    Args:
        dataset: Raw dataset with 'text' and 'label' fields.
        tokenizer: Hugging Face tokenizer instance.
    
    Returns:
        Tokenized dataset with 'input_ids', 'attention_mask', and 'label' fields.
        Original 'text' field removed to reduce memory usage.
    """
    def tokenize(batch: Dict[str, Any]) -> Dict[str, List[int]]:
        return tokenizer(
            batch['text'],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding=False,  # DataCollatorWithPadding handles this
            return_tensors=None
        )
    
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing dataset"
    )
    
    return tokenized


def compute_class_weights(dataset: DatasetDict) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced training.
    
    Formula: n_samples / (n_classes * n_samples_per_class)
    Minority classes (surprise, love) receive higher weights.
    
    Args:
        dataset: DatasetDict containing 'train' split with 'label' field.
    
    Returns:
        Tensor of shape (num_labels,) with per-class loss weights.
    """
    import numpy as np
    
    train_labels = dataset['train']['label']
    num_labels = get_num_labels(dataset)
    
    class_counts = np.bincount(train_labels, minlength=num_labels)
    total_samples = len(train_labels)
    
    # Avoid division by zero if a class is completely missing
    class_counts = np.maximum(class_counts, 1)
    
    weights = total_samples / (num_labels * class_counts)
    
    return torch.tensor(weights, dtype=torch.float32)


def prepare_dataset(
    use_official_splits: bool = True
) -> Tuple[DatasetDict, DistilBertTokenizerFast]:
    """
    Convenience facade: load, preprocess, and return dataset + tokenizer.
    
    This is the recommended entry point for training scripts.
    Individual functions remain available for fine-grained control.
    
    Args:
        use_official_splits: Forwarded to load_emotion_dataset().
    
    Returns:
        Tuple of (tokenized_dataset, tokenizer).
    """
    dataset = load_emotion_dataset(use_official_splits=use_official_splits)
    tokenizer = get_tokenizer()
    tokenized = preprocess_dataset(dataset, tokenizer)
    return tokenized, tokenizer