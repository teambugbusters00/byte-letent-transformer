# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Entropy-based sentence boundary detection using word-level tokenization.
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter

from ..tokenizers.word_level_tokenizer import WordLevelTokenizer

class EntropyBoundaryDetector:
    """
    Implements entropy-based sentence boundary detection using word-level tokenization.
    
    This class processes text to identify sentence boundaries by calculating
    the entropy of token transitions, with higher entropy typically indicating
    sentence boundaries.
    """
    
    def __init__(self, window_size: int = 5, threshold: float = 0.7):
        """
        Initialize the boundary detector.
        
        Args:
            window_size: Size of the context window for entropy calculation
            threshold: Entropy threshold for detecting boundaries (0-1)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.tokenizer = WordLevelTokenizer(lowercase=True)
        self.transition_counts = defaultdict(Counter)
        self.total_transitions = 0
        self.vocab = set()
    
    def train(self, text: str) -> None:
        """
        Train the model on a text corpus.
        
        Args:
            text: Training text
        """
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.inverse_vocab[tid] for tid in token_ids 
                 if tid in self.tokenizer.inverse_vocab]
        
        # Update vocabulary
        self.vocab.update(tokens)
        
        # Calculate transition counts
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            self.transition_counts[current][next_token] += 1
            self.total_transitions += 1
    
    def calculate_entropy(self, token: str) -> float:
        """
        Calculate the entropy of transitions from a given token.
        
        Args:
            token: The token to calculate entropy for
            
        Returns:
            Normalized entropy value (0-1)
        """
        if token not in self.transition_counts:
            return 0.0
            
        transitions = self.transition_counts[token]
        total = sum(transitions.values())
        if total == 0:
            return 0.0
            
        # Calculate entropy
        entropy = 0.0
        for count in transitions.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(transitions)) if transitions else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def detect_boundaries(self, text: str) -> List[Tuple[int, float]]:
        """
        Detect sentence boundaries in the given text.
        
        Args:
            text: Input text
            
        Returns:
            List of (position, entropy) tuples for potential boundaries
        """
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.inverse_vocab[tid] for tid in token_ids 
                 if tid in self.tokenizer.inverse_vocab]
        
        # Calculate entropy for each position
        boundaries = []
        for i in range(len(tokens) - self.window_size + 1):
            window = tokens[i:i + self.window_size]
            
            # Calculate average entropy in the window
            entropy_sum = 0.0
            for token in window:
                entropy_sum += self.calculate_entropy(token)
            avg_entropy = entropy_sum / len(window)
            
            # Check if this is a boundary
            if avg_entropy >= self.threshold:
                # Position is the end of the current window
                boundaries.append((i + self.window_size - 1, avg_entropy))
        
        return boundaries
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences based on detected boundaries.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Tokenize the text
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.inverse_vocab[tid] for tid in token_ids 
                 if tid in self.tokenizer.inverse_vocab]
        
        # Detect boundaries
        boundaries = [0] + [pos for pos, _ in self.detect_boundaries(text)] + [len(tokens)]
        
        # Split tokens into sentences
        sentences = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            sentence_tokens = tokens[start:end]
            sentence = ' '.join(sentence_tokens)
            sentences.append(sentence)
        
        return sentences
