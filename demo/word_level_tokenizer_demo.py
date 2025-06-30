#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Demo script for WordLevelTokenizer.

This script demonstrates how to use the WordLevelTokenizer with text from Project Gutenberg.
It downloads "War and Peace" by Leo Tolstoy and performs word-level tokenization on it.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import requests
from rich.console import Console
from rich.progress import Progress

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.tokenizers.word_level_tokenizer import WordLevelTokenizer

# Constants
WAR_AND_PEACE_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
DOWNLOAD_DIR = Path("demo_data")
DOWNLOAD_PATH = DOWNLOAD_DIR / "war_and_peace.txt"
SAMPLE_SIZE = 1000  # Number of characters to display in the demo


def download_file(url: str, path: Path) -> None:
    """Download a file from a URL to the specified path."""
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip download if file already exists
    if path.exists():
        print(f"File already exists at {path}. Skipping download.")
        return
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total size for progress tracking
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(path, 'wb') as f, Progress() as progress:
        task = progress.add_task("Downloading...", total=total_size)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
                progress.update(task, advance=len(chunk))
    
    print(f"Downloaded to {path}")


def load_text(path: Path, max_chars: int = None) -> str:
    """Load text from a file, optionally limiting the number of characters."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)
    return text


def analyze_text(text: str, tokenizer: WordLevelTokenizer) -> Tuple[list[int], list[str]]:
    """Tokenize text and return tokens and token IDs."""
    # Time the tokenization
    start_time = time.time()
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.inverse_vocab[tid] for tid in token_ids]
    elapsed = time.time() - start_time
    
    return token_ids, tokens, elapsed


def display_tokenization_sample(
    console: Console,
    text: str,
    tokens: list[str],
    token_ids: list[int],
    max_chars: int = 500
) -> None:
    """Display a sample of the tokenization results."""
    console.print("\n[bold blue]=== Tokenization Sample ===[/]")
    
    # Show original text sample
    console.print("\n[bold]Original Text (first 500 chars):[/]")
    console.print(f"{text[:max_chars]}...")
    
    # Show tokens sample (first 20 tokens)
    console.print("\n[bold]First 20 Tokens:[/]")
    sample_tokens = tokens[:20]
    sample_ids = token_ids[:20]
    
    for i, (token, tid) in enumerate(zip(sample_tokens, sample_ids)):
        # Replace newlines for display
        display_token = token.replace("\n", "\\n")
        console.print(f"  {i:2d}: {display_token!r:<15} (ID: {tid})")


def main():
    console = Console()
    
    # Download the text if needed
    try:
        download_file(WAR_AND_PEACE_URL, DOWNLOAD_PATH)
    except Exception as e:
        console.print(f"[red]Error downloading file: {e}[/]")
        return
    
    # Load the text
    try:
        text = load_text(DOWNLOAD_PATH)
        console.print(f"\n[green]Successfully loaded text ({len(text):,} characters)[/]")
    except Exception as e:
        console.print(f"[red]Error loading text: {e}[/]")
        return
    
    # Initialize tokenizer
    tokenizer = WordLevelTokenizer(lowercase=True)
    
    # Analyze the text
    console.print("\n[bold]Analyzing text with WordLevelTokenizer...[/]")
    token_ids, tokens, elapsed = analyze_text(text, tokenizer)
    
    # Display statistics
    console.print("\n[bold green]=== Tokenization Statistics ===[/]")
    console.print(f"Original text length: {len(text):,} characters")
    console.print(f"Number of tokens: {len(tokens):,}")
    console.print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    console.print(f"Tokenization time: {elapsed:.4f} seconds")
    console.print(f"Tokens per second: {len(tokens) / elapsed:,.0f}" if elapsed > 0 else "")
    
    # Show a sample of the tokenization
    display_tokenization_sample(console, text, tokens, token_ids)
    
    # Show some interesting tokens from the vocabulary
    console.print("\n[bold]Vocabulary Sample:[/]")
    vocab_items = list(tokenizer.vocab.items())
    # Show first 5, last 5, and some in between
    sample_indices = list(range(5)) + list(range(len(vocab_items)//2, len(vocab_items)//2 + 5)) + list(range(-5, 0))
    sample_indices = sorted(set(sample_indices))
    
    for i in sample_indices:
        if 0 <= i < len(vocab_items):
            token, tid = vocab_items[i]
            display_token = token.replace("\n", "\\n").replace("\r", "\\r")
            console.print(f"  {display_token!r:<20} → {tid}")
    
    console.print("\n[bold green]Demo completed successfully![/]")


if __name__ == "__main__":
    main()
