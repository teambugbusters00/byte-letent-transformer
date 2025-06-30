# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Text cleaning utilities for preprocessing text data before tokenization.
"""
import re
import string
from typing import Callable, List, Optional, Union

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

class TextCleaner:
    """
    A configurable text cleaner that applies a series of cleaning functions to text.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        custom_patterns: Optional[List[tuple[str, str]]] = None
    ):
        """
        Initialize the text cleaner with specified cleaning options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_extra_whitespace: Remove extra whitespace
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            custom_patterns: List of (pattern, replacement) tuples for custom regex replacements
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.custom_patterns = custom_patterns or []
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def clean_text(self, text: str) -> str:
        """
        Apply all enabled cleaning operations to the input text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        cleaned = text
        
        # Apply standard cleaning operations
        if self.remove_urls:
            cleaned = self._remove_urls(cleaned)
        if self.remove_emails:
            cleaned = self._remove_emails(cleaned)
        if self.remove_numbers:
            cleaned = self._remove_numbers(cleaned)
        if self.remove_punctuation:
            cleaned = self._remove_punctuation(cleaned)
        if self.lowercase:
            cleaned = cleaned.lower()
        if self.remove_extra_whitespace:
            cleaned = self._remove_extra_whitespace(cleaned)
            
        # Apply custom patterns
        for pattern, replacement in self.custom_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
            
        return cleaned.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = re.compile(r'\S+@\S+\.\S+')
        return email_pattern.sub('', text)
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        # Keep apostrophes for contractions
        return text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        return ' '.join(text.split())
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words after cleaning.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        cleaned = self.clean_text(text)
        return word_tokenize(cleaned)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences after cleaning.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Clean but preserve sentence boundaries
        cleaned = self.clean_text(text)
        return sent_tokenize(cleaned)
    
    def clean_and_tokenize(self, text: str, level: str = "word") -> Union[List[str], List[List[str]]]:
        """
        Clean and tokenize text at the specified level.
        
        Args:
            text: Input text to process
            level: Tokenization level ('word' or 'sentence')
            
        Returns:
            List of tokens or list of sentences (each a list of words)
        """
        if level == "word":
            return self.tokenize_words(text)
        elif level == "sentence":
            sentences = self.tokenize_sentences(text)
            return [self.tokenize_words(sent) for sent in sentences]
        else:
            raise ValueError(f"Unsupported tokenization level: {level}")


def create_default_cleaner() -> TextCleaner:
    """Create a TextCleaner with default settings for general text processing."""
    return TextCleaner(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        remove_extra_whitespace=True,
        remove_urls=True,
        remove_emails=True,
        custom_patterns=[
            (r'\s+', ' '),  # Replace multiple spaces with single space
            (r'\n+', ' '),  # Replace newlines with space
            (r'\t+', ' '),  # Replace tabs with space
        ]
    )
