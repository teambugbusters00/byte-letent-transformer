#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Process text with word-level tokenization before building knowledge graph.
"""

import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.preprocess.pipeline import create_default_pipeline
from bytelatent.tokenizers.word_level_tokenizer import WordLevelTokenizer
from bytelatent.preprocess.text_cleaner import TextCleaner

def read_war_and_peace() -> str:
    """Read the War and Peace text from the demo_data directory."""
    input_file = Path("demo_data") / "war_and_peace.txt"
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Read the first 10,000 characters for processing
            return f.read(10000)  # Limit size for demo purposes
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        print("Please make sure the file exists in the demo_data directory.")
        sys.exit(1)

def tokenize_text(text: str) -> List[str]:
    """Tokenize text using the word-level tokenizer."""
    # Initialize the tokenizer
    tokenizer = WordLevelTokenizer(lowercase=False)  # Already lowercased in cleaning
    
    # Tokenize the text - this returns a list of token IDs
    token_ids = tokenizer.encode(text)
    
    # Convert token IDs back to tokens using the tokenizer's inverse_vocab
    tokens = []
    for token_id in token_ids:
        if token_id in tokenizer.inverse_vocab:
            tokens.append(tokenizer.inverse_vocab[token_id])
    
    return tokens

def build_knowledge_graph(tokens: List[str]) -> Dict[str, Any]:
    """Build a knowledge graph from tokens."""
    import networkx as nx
    from collections import defaultdict
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes for each unique token
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    
    # Add nodes with frequency as a property
    for token, count in token_counts.items():
        G.add_node(token, type='token', frequency=count)
    
    # Add edges based on token co-occurrence within a window
    window_size = 3
    for i in range(len(tokens) - 1):
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            if tokens[i] != tokens[j]:  # Skip self-loops
                if G.has_edge(tokens[i], tokens[j]):
                    G[tokens[i]][tokens[j]]['weight'] += 1
                else:
                    G.add_edge(tokens[i], tokens[j], weight=1, relation='co-occurs_with')
    
    return nx.node_link_data(G)

def visualize_kg(kg_data: Dict[str, Any], output_file: Path):
    """Visualize a knowledge graph using pyvis."""
    try:
        from pyvis.network import Network
        import networkx as nx
        
        # Create a pyvis network
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )
        
        # Convert node-link data to NetworkX graph
        G = nx.node_link_graph(kg_data)
        
        # Limit the number of nodes for better visualization
        nodes = list(G.nodes(data=True))
        if len(nodes) > 100:
            print(f"Graph has {len(nodes)} nodes, limiting to top 100 for visualization")
            # Sort nodes by degree (most connected first)
            nodes_sorted = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)
            top_nodes = [n[0] for n in nodes_sorted[:100]]
            G = G.subgraph(top_nodes).copy()
        
        # Add nodes and edges to the pyvis network
        for node, node_attrs in G.nodes(data=True):
            # Set node properties
            node_label = node if len(str(node)) < 20 else f"{str(node)[:17]}..."
            node_title = f"{node}\nType: {node_attrs.get('type', 'token')}"
            if 'frequency' in node_attrs:
                node_title += f"\nFrequency: {node_attrs['frequency']}"
            
            # Add node with properties
            net.add_node(
                str(node),
                label=node_label,
                title=node_title,
                color="#97c2fc",
                size=10 + min(30, G.degree(node, weight='weight') * 0.5)  # Scale node size by weighted degree
            )
        
        # Add edges with properties
        for source, target, edge_attrs in G.edges(data=True):
            weight = edge_attrs.get('weight', 1)
            net.add_edge(
                str(source),
                str(target),
                title=f"Co-occurs ({weight}x)",
                label=f"{weight}",
                arrows='to',
                color='#808080',
                width=0.5 + min(weight * 0.5, 3)  # Scale edge width by weight
            )
        
        # Generate the visualization
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        
        # Save to HTML file
        net.save_graph(str(output_file))
        print(f"\nGraph visualization saved to: {output_file.absolute()}")
        
        # Open in default web browser
        webbrowser.open(f'file://{output_file.absolute()}')
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install the required packages for visualization:")
        print("pip install networkx pyvis")

def main():
    # Set up output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Read War and Peace text
    print("Reading War and Peace text...")
    text = read_war_and_peace()
    
    # Clean the text
    print("Cleaning text...")
    cleaner = TextCleaner(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_extra_whitespace=True,
        remove_urls=True,
        remove_emails=True
    )
    cleaned_text = cleaner.clean_text(text)
    
    # Save cleaned text
    with open(output_dir / "war_and_peace_cleaned.txt", 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Tokenize the cleaned text
    print("Tokenizing text...")
    tokens = tokenize_text(cleaned_text)
    
    # Save tokens
    with open(output_dir / "war_and_peace_tokens.json", 'w', encoding='utf-8') as f:
        json.dump(tokens, f, indent=2)
    
    # Build knowledge graph from tokens
    print("Building knowledge graph from tokens...")
    kg_data = build_knowledge_graph(tokens)
    
    # Save knowledge graph
    kg_file = output_dir / "war_and_peace_token_kg.json"
    with open(kg_file, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2)
    
    print(f"\n=== Processing Complete ===")
    print(f"Original text length: {len(text)} characters")
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Unique tokens: {len(set(tokens))}")
    
    # Visualize the knowledge graph
    print("\nGenerating knowledge graph visualization...")
    output_html = output_dir / "war_and_peace_token_kg.html"
    visualize_kg(kg_data, output_html)
    
    print("\n=== Output Files ===")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"  - war_and_peace_cleaned.txt: Cleaned text")
    print(f"  - war_and_peace_tokens.json: Extracted tokens")
    print(f"  - war_and_peace_token_kg.json: Token-based knowledge graph")
    print(f"  - war_and_peace_token_kg.html: Interactive visualization")

if __name__ == "__main__":
    try:
        import networkx as nx
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "pyvis"])
    
    main()
