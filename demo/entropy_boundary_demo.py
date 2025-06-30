#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Demo of entropy-based sentence boundary detection using word-level tokenization
and knowledge graph traversal with stable visualization.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rich.console import Console
from rich.panel import Panel
from rich.progress import track

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.entropy import EntropyBoundaryDetector, KnowledgeGraphTraverser
from bytelatent.tokenizers.word_level_tokenizer import WordLevelTokenizer
from bytelatent.preprocess.text_cleaner import TextCleaner

console = Console()

def build_knowledge_graph(tokens: List[str], window_size: int = 3) -> nx.Graph:
    """
    Build a co-occurrence graph from a list of tokens.
    
    Args:
        tokens: List of tokens
        window_size: Size of the sliding window for co-occurrence
        
    Returns:
        NetworkX graph representing token co-occurrences
    """
    G = nx.Graph()
    
    # Add nodes
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    for token, count in token_counts.items():
        G.add_node(token, count=count)
    
    # Add edges based on co-occurrence within window
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            if tokens[i] != tokens[j]:  # No self-loops
                if G.has_edge(tokens[i], tokens[j]):
                    G[tokens[i]][tokens[j]]['weight'] += 1
                else:
                    G.add_edge(tokens[i], tokens[j], weight=1)
    
    return G

def create_stable_visualization(graph: nx.Graph, token_entropy: Dict[str, float], 
                              output_path: Path, max_nodes: int = 30):
    """
    Create a stable, non-animated graph visualization using matplotlib.
    
    Args:
        graph: NetworkX graph to visualize
        token_entropy: Dictionary mapping tokens to entropy scores
        output_path: Path to save the visualization
        max_nodes: Maximum number of nodes to display
    """
    # Filter to most connected nodes for cleaner visualization
    node_degrees = dict(graph.degree())
    # Get nodes with at least 2 connections to avoid isolated nodes
    connected_nodes = {node: degree for node, degree in node_degrees.items() if degree >= 2}
    top_nodes = sorted(connected_nodes.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    top_node_names = [node for node, degree in top_nodes]
    
    # Create subgraph with top nodes
    subgraph = graph.subgraph(top_node_names).copy()
    
    # Use a stable layout algorithm with better spacing
    plt.figure(figsize=(20, 16))
    
    # Spring layout with fixed seed for reproducibility and better spacing
    pos = nx.spring_layout(subgraph, k=5, iterations=200, seed=42)
    
    # Prepare node colors based on entropy
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        entropy = token_entropy.get(node, 0)
        # Color based on entropy: blue (low) to red (high)
        if entropy < 0.3:
            color = 'lightblue'
        elif entropy < 0.6:
            color = 'yellow'
        else:
            color = 'lightcoral'
        node_colors.append(color)
        
        # Size based on degree with better scaling
        degree = subgraph.degree(node)
        node_sizes.append(max(300, degree * 100))
    
    # Prepare edge weights for visualization
    edge_weights = []
    for u, v in subgraph.edges():
        weight = subgraph[u][v].get('weight', 1)
        edge_weights.append(min(weight * 0.8, 4))  # Better edge visibility
    
    # Draw the graph with stable parameters
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors, 
                          node_size=node_sizes,
                          alpha=0.9,
                          linewidths=2,
                          edgecolors='black')
    
    nx.draw_networkx_edges(subgraph, pos,
                          width=edge_weights,
                          alpha=0.4,
                          edge_color='gray')
    
    # Smart label selection: show labels for important nodes
    labels_to_show = {}
    for node in subgraph.nodes():
        degree = subgraph.degree(node)
        entropy = token_entropy.get(node, 0)
        
        # Show labels for:
        # 1. High degree nodes (well connected)
        # 2. High entropy nodes (interesting)
        # 3. Medium entropy nodes with good connections
        if (degree >= 3 or entropy > 0.4 or 
            (entropy > 0.2 and degree >= 2)):
            # Clean up the label - remove empty strings and whitespace
            clean_label = str(node).strip()
            if clean_label and clean_label != '' and len(clean_label) > 0:
                labels_to_show[node] = clean_label
    
    # Draw labels with better formatting
    nx.draw_networkx_labels(subgraph, pos, 
                           labels=labels_to_show,
                           font_size=12,
                           font_weight='bold',
                           font_color='darkblue',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   alpha=0.8,
                                   edgecolor='black'))
    
    # Create legend
    low_patch = mpatches.Patch(color='lightblue', label='Low Entropy (< 0.3)')
    med_patch = mpatches.Patch(color='yellow', label='Medium Entropy (0.3-0.6)')
    high_patch = mpatches.Patch(color='lightcoral', label='High Entropy (> 0.6)')
    plt.legend(handles=[low_patch, med_patch, high_patch], 
              loc='upper right', fontsize=12)
    
    plt.title('Token Co-occurrence Graph with Entropy Scores\n(Node size = connection degree, Color = entropy)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save as static image
    plt.savefig(output_path / 'stable_graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'stable_graph_visualization.pdf', bbox_inches='tight')
    plt.show()
    
    # Print debug info about labels
    console.print(f"\n[bold]Graph Visualization Info:[/]")
    console.print(f"Nodes displayed: {len(subgraph.nodes())}")
    console.print(f"Labels shown: {len(labels_to_show)}")
    console.print(f"Sample labels: {list(labels_to_show.values())[:10]}")

def create_detailed_node_view(graph: nx.Graph, token_entropy: Dict[str, float], output_path: Path):
    """Create a detailed view showing the most important tokens and their relationships."""
    # Get the most important tokens (high degree or high entropy)
    node_degrees = dict(graph.degree())
    
    # Score nodes by combination of degree and entropy
    node_scores = {}
    for node in graph.nodes():
        degree = node_degrees.get(node, 0)
        entropy = token_entropy.get(node, 0)
        # Filter out empty/whitespace tokens
        clean_node = str(node).strip()
        if clean_node and len(clean_node) > 0 and clean_node != '':
            # Combined score: degree importance + entropy importance
            node_scores[node] = (degree * 0.6) + (entropy * 10)  # Weight entropy higher
    
    # Get top 20 most important tokens
    top_tokens = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    top_token_names = [token for token, score in top_tokens]
    
    # Create subgraph
    subgraph = graph.subgraph(top_token_names).copy()
    
    # Create circular layout for better label visibility
    plt.figure(figsize=(16, 16))
    pos = nx.circular_layout(subgraph)
    
    # Adjust positions to spread out nodes more
    for node in pos:
        pos[node] = pos[node] * 2  # Expand the circle
    
    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        entropy = token_entropy.get(node, 0)
        degree = subgraph.degree(node)
        
        # Color gradient from blue to red based on entropy
        if entropy < 0.2:
            color = '#ADD8E6'  # Light blue
        elif entropy < 0.4:
            color = '#90EE90'  # Light green
        elif entropy < 0.6:
            color = '#FFD700'  # Gold
        elif entropy < 0.8:
            color = '#FFA500'  # Orange
        else:
            color = '#FF6347'  # Tomato red
        
        node_colors.append(color)
        node_sizes.append(max(500, degree * 150))
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          linewidths=3,
                          edgecolors='black')
    
    # Draw edges with transparency
    nx.draw_networkx_edges(subgraph, pos,
                          alpha=0.3,
                          edge_color='gray',
                          width=1)
    
    # Add all labels with better positioning
    labels = {}
    for node in subgraph.nodes():
        clean_label = str(node).strip()
        if clean_label and clean_label != '':
            labels[node] = clean_label
    
    # Draw labels with offset for better readability
    label_pos = {}
    for node in pos:
        x, y = pos[node]
        # Offset labels slightly outside the nodes
        offset = 0.15
        if x >= 0:
            label_pos[node] = (x + offset, y)
        else:
            label_pos[node] = (x - offset, y)
    
    nx.draw_networkx_labels(subgraph, label_pos,
                           labels=labels,
                           font_size=14,
                           font_weight='bold',
                           font_color='darkblue',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   alpha=0.9,
                                   edgecolor='navy'))
    
    plt.title('Top 20 Most Important Tokens\n(Size = connections, Color = entropy score)', 
              fontsize=18, fontweight='bold', pad=30)
    
    # Add entropy scale
    entropy_colors = ['#ADD8E6', '#90EE90', '#FFD700', '#FFA500', '#FF6347']
    entropy_labels = ['Very Low (0-0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 
                     'High (0.6-0.8)', 'Very High (0.8+)']
    
    patches = [mpatches.Patch(color=color, label=label) 
              for color, label in zip(entropy_colors, entropy_labels)]
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path / 'detailed_token_view.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'detailed_token_view.pdf', bbox_inches='tight')
    plt.show()
    
    # Print token information
    console.print(f"\n[bold]Top Important Tokens:[/]")
    for i, (token, score) in enumerate(top_tokens[:10], 1):
        degree = node_degrees.get(token, 0)
        entropy = token_entropy.get(token, 0)
        console.print(f"{i:2d}. '{token}' - Score: {score:.2f}, Connections: {degree}, Entropy: {entropy:.3f}")

def create_entropy_histogram(token_entropy: Dict[str, float], output_path: Path):
    """Create a histogram of entropy scores."""
    entropy_values = list(token_entropy.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(entropy_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Entropy Score')
    plt.ylabel('Number of Tokens')
    plt.title('Distribution of Token Entropy Scores')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path / 'entropy_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    console.print("[bold blue]Entropy-Based Sentence Boundary Detection Demo[/]")
    console.print("=" * 60)
    
    # Read War and Peace text
    input_file = Path("demo_data") / "war_and_peace.txt"
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read(10000)  # First 10,000 chars for demo
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {input_file}[/]")
        console.print("Please make sure the file exists in the demo_data directory.")
        
        # Create sample text for demo if file not found
        console.print("[yellow]Using sample text for demonstration...[/]")
        text = """
        In the world of natural language processing, entropy plays a crucial role in understanding 
        text structure. Words with high entropy often mark important boundaries in discourse. 
        The concept of information theory helps us identify these patterns. When we analyze text, 
        we look for sudden changes in predictability. These changes often correspond to sentence 
        boundaries or topic shifts. Machine learning algorithms can learn to detect these patterns 
        automatically. The process involves building statistical models of language structure.
        """
    
    # Clean the text
    console.print("\n[bold]1. Cleaning text...[/]")
    cleaner = TextCleaner(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_extra_whitespace=True,
        remove_urls=True,
        remove_emails=True
    )
    cleaned_text = cleaner.clean_text(text)
    
    # Initialize tokenizer and entropy detector
    console.print("\n[bold]2. Tokenizing text and training entropy model...[/]")
    tokenizer = WordLevelTokenizer(lowercase=True)
    token_ids = tokenizer.encode(cleaned_text)
    tokens = [tokenizer.inverse_vocab[tid] for tid in token_ids 
             if tid in tokenizer.inverse_vocab]
    
    # Train entropy model
    entropy_detector = EntropyBoundaryDetector(window_size=3, threshold=0.7)
    entropy_detector.train(cleaned_text)
    
    # Build knowledge graph
    console.print("\n[bold]3. Building knowledge graph...[/]")
    graph = build_knowledge_graph(tokens)
    
    # Calculate entropy scores for each token
    token_entropy = {}
    for token in set(tokens):
        token_entropy[token] = entropy_detector.calculate_entropy(token)
    
    # Initialize graph traverser
    traverser = KnowledgeGraphTraverser(graph, token_entropy)
    
    # Detect sentence boundaries
    console.print("\n[bold]4. Detecting sentence boundaries...[/]")
    boundaries = traverser.find_sentence_boundaries(tokens)
    
    # Split text into sentences
    sentences = []
    start = 0
    for boundary in boundaries:
        sentence_tokens = tokens[start:boundary+1]
        if sentence_tokens:  # Only add non-empty sentences
            sentences.append(' '.join(sentence_tokens))
        start = boundary + 1
    
    # Add the last sentence if any tokens remain
    if start < len(tokens):
        sentence_tokens = tokens[start:]
        if sentence_tokens:
            sentences.append(' '.join(sentence_tokens))
    
    # Display results
    console.print("\n[bold green]Detected Sentences:[/]")
    for i, sentence in enumerate(sentences[:5], 1):  # Show first 5 sentences
        console.print(f"\n[bold]{i}.[/] {sentence}")
    
    if len(sentences) > 5:
        console.print(f"\n... and {len(sentences) - 5} more sentences.")
    
    # Show entropy scores for some tokens
    console.print("\n[bold]Sample Token Entropy Scores:[/]")
    sample_tokens = tokens[:10] + [tokens[i] for i in boundaries[:3] if i < len(tokens)]
    for token in set(sample_tokens):
        console.print(f"  - '{token}': {token_entropy.get(token, 0):.3f}")
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create stable visualizations
    console.print("\n[bold]5. Creating stable visualizations...[/]")
    create_stable_visualization(graph, token_entropy, output_dir)
    create_detailed_node_view(graph, token_entropy, output_dir)
    create_entropy_histogram(token_entropy, output_dir)
    
    # Save results
    # Save sentences
    with open(output_dir / "detected_sentences.txt", 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences, 1):
            f.write(f"{i}. {sentence}\n\n")
    
    # Save token entropy
    with open(output_dir / "token_entropy.json", 'w', encoding='utf-8') as f:
        json.dump(token_entropy, f, indent=2, ensure_ascii=False)
    
    # Save graph statistics
    with open(output_dir / "graph_stats.txt", 'w', encoding='utf-8') as f:
        f.write(f"Graph Statistics:\n")
        f.write(f"Number of nodes: {graph.number_of_nodes()}\n")
        f.write(f"Number of edges: {graph.number_of_edges()}\n")
        f.write(f"Graph density: {nx.density(graph):.4f}\n")
        f.write(f"Average clustering coefficient: {nx.average_clustering(graph):.4f}\n")
        
        # Top 10 most connected nodes
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        f.write(f"\nTop 10 most connected tokens:\n")
        for token, degree in top_nodes:
            entropy = token_entropy.get(token, 0)
            f.write(f"  {token}: {degree} connections, entropy: {entropy:.3f}\n")
    
    console.print(f"\n[bold green]Results saved to {output_dir.absolute()}[/]")
    console.print("  - detected_sentences.txt: Detected sentences")
    console.print("  - token_entropy.json: Entropy scores for tokens")
    console.print("  - stable_graph_visualization.png/pdf: Network structure visualization")
    console.print("  - detailed_token_view.png/pdf: Clear view of top important tokens")
    console.print("  - entropy_histogram.png: Entropy distribution")
    console.print("  - graph_stats.txt: Graph statistics and top nodes")

if __name__ == "__main__":
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        console.print("[yellow]Installing required packages...[/]")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "matplotlib"])
    
    main()