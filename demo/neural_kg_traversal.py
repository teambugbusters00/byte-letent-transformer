#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Demo of neural entropy-based knowledge graph traversal for sentence boundary detection.
"""

import os
import sys
import torch
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.neural_entropy import (
    KnowledgeGraphConstructor,
    NeuralEntropyModel,
    KGGraphTraverser
)
from bytelatent.tokenizers.word_level_tokenizer import WordLevelTokenizer

def load_and_preprocess_text(filepath: str, max_chars: int = 10000) -> str:
    """Load and preprocess text from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)
    
    # Simple preprocessing - in a real scenario, you'd want more robust cleaning
    text = ' '.join(text.split())  # Normalize whitespace
    return text

def train_neural_model(kg_constructor: KnowledgeGraphConstructor, 
                     num_epochs: int = 10,
                     batch_size: int = 32,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> NeuralEntropyModel:
    """
    Train the neural entropy model on the knowledge graph.
    
    Note: This is a simplified training loop. In practice, you'd want:
    1. More sophisticated negative sampling
    2. Proper train/validation split
    3. Learning rate scheduling
    4. Early stopping
    5. Model checkpointing
    """
    print("Initializing neural model...")
    model = NeuralEntropyModel(
        node_feature_dim=128,
        graph_hidden_dim=256,
        lstm_hidden_dim=256,
        num_relations=len(kg_constructor.rel_to_idx),
        dropout=0.2
    ).to(device)
    
    # Create training data (simplified - in practice, you'd want more sophisticated sampling)
    print("Preparing training data...")
    train_data = []
    
    # Sample random paths from the graph
    num_nodes = len(kg_constructor.graph.nodes())
    node_list = list(kg_constructor.graph.nodes())
    
    for _ in tqdm(range(1000), desc="Generating training samples"):
        # Sample a random path of length 3-5
        path_length = torch.randint(3, 6, (1,)).item()
        start_node = node_list[torch.randint(0, num_nodes, (1,)).item()]
        
        path = [start_node]
        current = start_node
        
        for _ in range(path_length - 1):
            neighbors = list(kg_constructor.graph.neighbors(current))
            if not neighbors:
                break
            current = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
            path.append(current)
        
        if len(path) >= 2:  # Need at least 2 nodes for a valid path
            train_data.append(path)
    
    # Convert to PyG Data format
    graph_data = kg_constructor._create_graph_data()
    
    # Training loop (simplified)
    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle training data
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch_paths = train_data[i:i+batch_size]
            
            # Prepare batch
            max_len = max(len(path) for path in batch_paths)
            batch_indices = torch.zeros((len(batch_paths), max_len), dtype=torch.long, device=device)
            batch_lengths = torch.tensor([len(path) for path in batch_paths], dtype=torch.long, device=device)
            
            for j, path in enumerate(batch_paths):
                batch_indices[j, :len(path)] = torch.tensor(
                    [kg_constructor.node_to_idx[node] for node in path], 
                    device=device
                )
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(graph_data, batch_indices, batch_lengths)
            
            # Create targets (boundary at the end of each path)
            targets = torch.zeros_like(outputs['boundary_probs'])
            for j, length in enumerate(batch_lengths):
                targets[j, length-1] = 1.0  # Boundary at the last position
            
            # Calculate loss
            loss = criterion(outputs['boundary_probs'], targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / (len(train_data) / batch_size):.4f}")
    
    return model

def main():
    # Configuration
    data_dir = Path("demo_data")
    output_dir = Path("demo_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Load and preprocess the text
    print("Loading and preprocessing text...")
    text_file = data_dir / "war_and_peace_chapter1.txt"
    if not text_file.exists():
        print(f"Error: Could not find {text_file}")
        print("Please download the book text first using the download script.")
        return
    
    text = load_and_preprocess_text(text_file)
    
    # Step 2: Initialize KG constructor and process the text
    print("Building knowledge graph...")
    kg_constructor = KnowledgeGraphConstructor()
    kg_constructor.add_document(text, doc_id="war_and_peace")
    
    # Save the graph for inspection
    kg_constructor.save_graph(output_dir / "knowledge_graph.json")
    print(f"Knowledge graph saved to {output_dir}/knowledge_graph.json")
    
    # Step 3: Train the neural model
    print("\nTraining neural entropy model...")
    model = train_neural_model(kg_constructor, num_epochs=5, device=device)
    
    # Save the trained model
    model_path = output_dir / "neural_entropy_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
    
    # Step 4: Initialize the graph traverser
    print("\nInitializing graph traverser...")
    traverser = KGGraphTraverser(
        model=model,
        graph=kg_constructor.graph,
        device=device,
        max_steps=10,
        boundary_threshold=0.7,
        max_entropy=0.9,
        beam_size=3
    )
    
    # Step 5: Perform traversal from some starting nodes
    print("\nPerforming traversals...")
    
    # Get some entities to start from (nodes with highest degree)
    degrees = dict(kg_constructor.graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    start_nodes = [node for node, _ in top_nodes]
    
    print(f"Starting traversals from: {start_nodes[:3]}...")
    
    # Perform traversals
    results = traverser.find_sentence_boundaries(start_nodes[:3])  # Limit to 3 for demo
    
    # Print results
    print("\nTraversal Results:")
    print("-" * 80)
    
    for start_node, paths in results.items():
        print(f"\nStarting from: {start_node}")
        print(f"Found {len(paths)} paths")
        
        for i, path in enumerate(paths[:2]):  # Show first 2 paths per node
            print(f"\nPath {i+1} (Score: {path['score']:.2f}):")
            
            # Print the path with nodes and edges
            path_str = path['nodes'][0]
            for j in range(1, len(path['nodes'])):
                path_str += f" --[{path['edges'][j-1]}]--> {path['nodes'][j]}"
                if 'entropies' and j-1 < len(path['entropies']):
                    path_str += f" (entropy: {path['entropies'][j-1]:.2f})"
            
            print(path_str)
            
            # Print boundary probabilities
            print("Boundary probabilities:", 
                 [f"{p:.2f}" for p in path['boundary_probabilities']])
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
