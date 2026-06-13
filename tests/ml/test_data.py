"""Tests for data loading module. Run with: pytest tests/ -v"""
import pytest
import torch
import numpy as np
from datasets import load_dataset

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.data import (
    load_emotion_dataset,
    get_tokenizer,
    preprocess_dataset,
    get_label_names,
    get_num_labels,
    compute_class_weights
)


class TestDatasetLoading:
    """Verify dataset loads with correct structure."""
    
    def test_dataset_splits_exist(self):
        dataset = load_emotion_dataset()
        assert 'train' in dataset
        assert 'validation' in dataset
        assert 'test' in dataset
    
    def test_label_count(self):
        dataset = load_emotion_dataset()
        num_labels = get_num_labels(dataset)
        
        for split in ['train', 'validation', 'test']:
            labels = dataset[split]['label']
            assert min(labels) >= 0, f"{split} has negative label"
            assert max(labels) < num_labels, \
                f"{split} has label {max(labels)} but num_labels={num_labels}"
    
    def test_all_labels_present(self):
        """Critical: ensure no class disappears during split."""
        dataset = load_emotion_dataset()
        num_labels = get_num_labels(dataset)
        
        for split in ['train', 'validation', 'test']:
            unique_labels = set(dataset[split]['label'])
            missing = set(range(num_labels)) - unique_labels
            assert len(missing) == 0, \
                f"{split} missing labels: {missing}"
    
    def test_label_names_match_dataset(self):
        """Verify get_label_names returns dataset's actual label names."""
        dataset = load_emotion_dataset()
        names = get_label_names(dataset)
        
        raw = load_dataset("dair-ai/emotion", split='train')
        expected = raw.features['label'].names
        
        assert names == expected, f"Label names mismatch: {names} vs {expected}"
        assert len(names) == 6, f"Expected 6 labels, got {len(names)}"
        assert 'surprise' in names, "Minority class 'surprise' missing"
        assert 'love' in names, "Minority class 'love' missing"
    
    def test_official_splits_flag(self):
        """Verify use_official_splits parameter works correctly."""
        official = load_emotion_dataset(use_official_splits=True)
        custom = load_emotion_dataset(use_official_splits=False)
        
        for split in ['train', 'validation', 'test']:
            assert split in official
            assert split in custom
        
        official_train_size = len(official['train'])
        custom_train_size = len(custom['train'])
        assert custom_train_size < official_train_size, \
            "Custom split should have smaller training set"


class TestTokenizer:
    """Verify tokenizer loads and produces valid structures."""
    
    def test_tokenizer_loads(self):
        tokenizer = get_tokenizer()
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0
        assert tokenizer.vocab_size == 30522, \
            f"Expected DistilBERT vocab size 30522, got {tokenizer.vocab_size}"
    
    def test_tokenization_structure(self):
        """Verify tokenized samples have required fields and valid lengths."""
        dataset = load_emotion_dataset()
        tokenizer = get_tokenizer()
        tokenized = preprocess_dataset(dataset, tokenizer)
        
        sample = tokenized['train'][0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'label' in sample
        
        seq_len = len(sample['input_ids'])
        assert seq_len > 0, "input_ids should not be empty"
        assert seq_len <= 128, f"Sequence length {seq_len} exceeds MAX_SEQUENCE_LENGTH"
        assert len(sample['attention_mask']) == seq_len, \
            "attention_mask must have same length as input_ids"
    
    def test_variable_length_sequences(self):
        """Verify dynamic padding preserves variable sequence lengths."""
        dataset = load_emotion_dataset()
        tokenizer = get_tokenizer()
        tokenized = preprocess_dataset(dataset, tokenizer)
        
        lengths = [len(tokenized['train'][i]['input_ids']) for i in range(100)]
        unique_lengths = set(lengths)
        
        assert len(unique_lengths) > 1, \
            f"Expected variable-length sequences but all 100 samples have same length. " \
            f"Check if padding is incorrectly enabled."
        
        assert all(0 < l <= 128 for l in lengths), \
            f"Some sequences outside valid range [1, 128]"
    
    def test_tokenizer_consistency(self):
        """Verify all token IDs are within vocabulary range."""
        dataset = load_emotion_dataset()
        tokenizer = get_tokenizer()
        tokenized = preprocess_dataset(dataset, tokenizer)
        
        vocab_size = tokenizer.vocab_size
        for i in range(50):
            sample = tokenized['train'][i]
            invalid_ids = [tid for tid in sample['input_ids'] if tid < 0 or tid >= vocab_size]
            assert len(invalid_ids) == 0, \
                f"Sample {i} contains invalid token IDs: {invalid_ids}"


class TestClassWeights:
    """Verify class weight computation for imbalanced dataset."""
    
    def test_class_weights_shape(self):
        dataset = load_emotion_dataset()
        weights = compute_class_weights(dataset)
        num_labels = get_num_labels(dataset)
        
        assert len(weights) == num_labels, \
            f"Expected {num_labels} weights, got {len(weights)}"
        assert weights.dtype == torch.float32, \
            f"Expected float32 weights, got {weights.dtype}"
    
    def test_minority_classes_weighted_higher(self):
        """Minority classes (surprise, love) should have higher weights."""
        dataset = load_emotion_dataset()
        weights = compute_class_weights(dataset)
        names = get_label_names(dataset)
        
        weight_dict = dict(zip(names, weights.tolist()))
        
        assert weight_dict['surprise'] > weight_dict['joy'], \
            f"surprise weight {weight_dict['surprise']:.3f} should exceed joy {weight_dict['joy']:.3f}"
        assert weight_dict['love'] > weight_dict['sadness'], \
            f"love weight {weight_dict['love']:.3f} should exceed sadness {weight_dict['sadness']:.3f}"
    
    def test_weights_sum_to_expected(self):
        """
        Verify weighted contribution equals total samples.
        
        Formula: sum(class_count[i] * weight[i]) = total_samples
        This is a mathematical invariant of inverse-frequency weighting.
        
        Note: Uses relaxed tolerance (1e-3) because float32 operations
        across 16k+ samples accumulate rounding error beyond 1e-5.
        """
        dataset = load_emotion_dataset()
        weights = compute_class_weights(dataset)
        
        train_labels = np.array(dataset['train']['label'])
        num_labels = len(weights)
        class_counts = np.bincount(train_labels, minlength=num_labels)
        total_samples = len(train_labels)
        
        # Each sample's contribution = weight of its class
        weighted_sum = (class_counts * weights.numpy()).sum()
        
        # Float32 tolerance: 1e-3 accounts for 16k * epsilon
        assert abs(weighted_sum - total_samples) < 1e-3, \
            f"Weighted sum {weighted_sum:.6f} differs from expected {total_samples} " \
            f"by {abs(weighted_sum - total_samples):.6f} (tolerance: 1e-3)"
    
    def test_all_weights_positive(self):
        """All class weights should be positive and finite."""
        dataset = load_emotion_dataset()
        weights = compute_class_weights(dataset)
        
        assert torch.all(weights > 0), "All weights must be positive"
        assert torch.all(torch.isfinite(weights)), "All weights must be finite"