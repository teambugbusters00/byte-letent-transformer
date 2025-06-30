#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Demo script for the text preprocessing and knowledge graph pipeline.

This script demonstrates how to use the preprocessing pipeline to clean text,
extract entities and relationships, and build a knowledge graph.
"""

import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.preprocess.pipeline import create_default_pipeline
from bytelatent.preprocess.text_cleaner import TextCleaner


def visualize_graph(kg_data: dict, output_file: Optional[Path] = None) -> None:
    """
    Visualize a knowledge graph using pyvis.
    
    Args:
        kg_data: Knowledge graph data in node-link format
        output_file: Path to save the visualization (HTML)
    """
    try:
        from pyvis.network import Network
        import networkx as nx
        
        # Create a pyvis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )
        
        # Convert node-link data back to a NetworkX graph
        G = nx.node_link_graph(kg_data)
        
        # Add nodes and edges to the pyvis network
        for node, node_attrs in G.nodes(data=True):
            # Set node properties
            node_label = node
            node_title = f"{node}\nType: {node_attrs.get('type', 'node')}"
            if 'entity_type' in node_attrs:
                node_title += f"\nEntity: {node_attrs['entity_type']}"
            
            # Add node with properties
            net.add_node(
                str(node),  # Node ID must be a string
                label=node_label,
                title=node_title,
                color="#97c2fc" if node_attrs.get('type') == 'entity' else "#fcba03"
            )
        
        # Add edges with properties
        for source, target, edge_attrs in G.edges(data=True):
            net.add_edge(
                str(source),
                str(target),
                title=edge_attrs.get('relation', 'related'),
                label=edge_attrs.get('relation', ''),
                arrows='to',
                color='#808080'
            )
        
        # Generate the visualization
        net.toggle_physics(True)
        
        # Save to HTML file
        if output_file is None:
            output_file = Path("kg_visualization.html")
        
        net.save_graph(str(output_file))
        print(f"\nGraph visualization saved to: {output_file.absolute()}")
        
        # Open in default web browser
        webbrowser.open(f'file://{output_file.absolute()}')
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install the required packages for visualization:")
        print("pip install networkx matplotlib pyvis")


def plot_graph(kg_data: dict) -> None:
    """
    Plot a simple static visualization of the knowledge graph using matplotlib.
    
    Args:
        kg_data: Knowledge graph data in node-link format
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Convert to NetworkX graph
        G = nx.node_link_graph(kg_data)
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Define node colors based on type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('type') == 'entity':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgreen')
        
        # Draw the graph
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)
        
        # Draw node labels
        node_labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(u, v): d.get('relation', '') for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        
        # Save the figure
        output_file = Path("kg_plot.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGraph plot saved to: {output_file.absolute()}")
        
        # Show the plot
        plt.show()
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install the required packages for plotting:")
        print("pip install matplotlib networkx")


def main():
    # Example text to process
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. Apple is known for
    its innovative products like the iPhone, iPad, and Mac computers. The company's CEO is Tim Cook.
    Apple has over 147,000 employees and operates in more than 25 countries.
    """

    print("=== Text Preprocessing and Knowledge Graph Demo ===\n")
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create a default pipeline
    print("Creating preprocessing pipeline...")
    pipeline = create_default_pipeline(output_dir=output_dir)
    
    # Process the sample text
    print("\nProcessing sample text...")
    result = pipeline.process_text(
        sample_text,
        clean_text=True,
        extract_kg=True,
        save_output=True,
        output_prefix="sample"
    )
    
    # Display results
    print("\n=== Processing Results ===")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Cleaned text length: {len(result.get('cleaned_text', ''))} characters")
    
    # Display entities
    entities = result.get('entities', [])
    if entities:
        print("\n=== Extracted Entities ===")
        for i, entity in enumerate(entities[:10], 1):  # Show first 10 entities
            print(f"{i}. {entity['text']} ({entity['type']})")
    
    # Display relations
    relations = result.get('relations', [])
    if relations:
        print("\n=== Extracted Relations ===")
        for i, rel in enumerate(relations[:5], 1):  # Show first 5 relations
            print(f"{i}. {rel['subject']} --[{rel['relation']}]--> {rel['object']}")
    
    # Display knowledge graph info
    kg_file = output_dir / "sample_kg.json"
    if kg_file.exists():
        try:
            import networkx as nx
            with open(kg_file, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            kg = nx.node_link_graph(kg_data)
            
            print(f"\n=== Knowledge Graph ===")
            print(f"Nodes: {len(kg.nodes())}")
            print(f"Edges: {len(kg.edges())}")
            
            # Show a few nodes and edges
            print("\nSample nodes:")
            nodes = list(kg.nodes(data=True))
            for i, node in enumerate(nodes[:3]):
                node_attrs = {k: v for k, v in node[1].items() if not k.startswith('_')}
                print(f"  {i+1}. {node[0]} {f'({node_attrs})' if node_attrs else ''}")
            
            print("\nSample edges:")
            edges = list(kg.edges(data=True))
            for i, edge in enumerate(edges[:3]):
                edge_attrs = {k: v for k, v in edge[2].items() if not k.startswith('_')}
                print(f"  {i+1}. {edge[0]} --[{edge_attrs}]--> {edge[1]}")
            
            # Ask user if they want to visualize the graph
            try:
                visualize = input("\nWould you like to visualize the knowledge graph? (y/n): ").strip().lower()
                if visualize == 'y':
                    # Visualize with pyvis (interactive)
                    output_html = output_dir / "kg_visualization.html"
                    visualize_graph(kg_data, output_html)
                    
                    # Also create a static plot
                    plot_choice = input("Would you like to see a static plot as well? (y/n): ").strip().lower()
                    if plot_choice == 'y':
                        plot_graph(kg_data)
            except Exception as e:
                print(f"Error during visualization: {e}")
                
        except Exception as e:
            print(f"\nError loading knowledge graph: {e}")
    else:
        print("\nNo knowledge graph data found in output.")
    
    print("\n=== Output Files ===")
    print(f"Results saved to: {output_dir.absolute()}")
    print("  - sample_cleaned.txt: Cleaned text")
    print("  - sample_entities.json: Extracted entities")
    print("  - sample_relations.json: Extracted relations")
    print("  - sample_full.json: Complete processing results")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import spacy
        import nltk
    except ImportError:
        import subprocess
        import sys
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "nltk"])
        
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
        except:
            pass
    
    main()
