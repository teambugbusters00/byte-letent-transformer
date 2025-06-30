# Copyright (c) Meta Platforms, Inc. and affiliates.
import re
from typing import Optional

from .abstract_tokenizer import Tokenizer


class WordLevelTokenizer(Tokenizer):
    """
    A simple word-level tokenizer that splits text into words and punctuation.
    Each word and punctuation mark becomes a separate token.
    """
    
    def __init__(self, lowercase: bool = False):
        """
        Initialize the word-level tokenizer.
        
        Args:
            lowercase: If True, converts all text to lowercase before tokenization
        """
        self.lowercase = lowercase
        self.word_pattern = re.compile(r"\w+|[^\w\s]")
        self.vocab = {}
        self.inverse_vocab = {}
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        
        # Add special tokens to vocab
        self._add_to_vocab(self.unk_token)
        self._add_to_vocab(self.bos_token)
        self._add_to_vocab(self.eos_token)
        self._add_to_vocab(self.pad_token)
    
    def _add_to_vocab(self, token: str) -> None:
        """Add a token to the vocabulary if it doesn't exist."""
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    def _tokenize(self, text: str) -> list[str]:
        """Split text into tokens."""
        if self.lowercase:
            text = text.lower()
        # Split into words and punctuation, preserving whitespace after tokens
        tokens = []
        pos = 0
        for match in self.word_pattern.finditer(text):
            # Add any leading whitespace
            start = match.start()
            if start > pos:
                tokens.append(text[pos:start])
            # Add the token
            tokens.append(match.group())
            pos = match.end()
        # Add any trailing whitespace
        if pos < len(text):
            tokens.append(text[pos:])
        return tokens
    
    def encode(
        self, 
        text: str, 
        add_bos: bool = False, 
        add_eos: bool = False
    ) -> list[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sentence token
            add_eos: Whether to add end-of-sentence token
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        token_ids = []
        
        if add_bos:
            token_ids.append(self.vocab[self.bos_token])
            
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Add new tokens to vocab on the fly
                self._add_to_vocab(token)
                token_ids.append(self.vocab[token])
                
        if add_eos:
            token_ids.append(self.vocab[self.eos_token])
            
        return token_ids
    
    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        return "".join(
            self.inverse_vocab.get(token, self.unk_token) 
            for token in tokens
            if token in self.inverse_vocab  # Skip unknown tokens
        )
    
    def get_token_offsets(
        self, 
        text: str, 
        tokens: Optional[list[int]] = None
    ) -> tuple[list[str], list[int]]:
        """
        Get the offsets of tokens in the original text.
        
        Args:
            text: Original text
            tokens: Optional list of token IDs. If None, will encode the text.
            
        Returns:
            Tuple of (tokens, offsets)
        """
        if tokens is None:
            tokens = self.encode(text)
            
        token_texts = [self.inverse_vocab.get(token, self.unk_token) for token in tokens]
        
        # Simple implementation: tokens are space-separated in the output
        # For a more accurate implementation, you'd track actual character positions
        offsets = []
        pos = 0
        for token in token_texts:
            offsets.append(pos)
            pos += len(token)
            
        return token_texts, offsets
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)
