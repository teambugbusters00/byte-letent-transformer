#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Process War and Peace text and visualize the knowledge graph.
"""

import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.preprocess.pipeline import create_default_pipeline

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

def visualize_kg(kg_data, output_file: Path):
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
            node_title = f"{node}\nType: {node_attrs.get('type', 'node')}"
            if 'entity_type' in node_attrs:
                node_title += f"\nEntity: {node_attrs['entity_type']}"
            
            # Add node with properties
            net.add_node(
                str(node),
                label=node_label,
                title=node_title,
                color="#97c2fc" if node_attrs.get('type') == 'entity' else "#fcba03",
                size=10 + min(30, G.degree(node) * 2)  # Scale node size by degree
            )
        
        # Add edges with properties
        for source, target, edge_attrs in G.edges(data=True):
            relation = edge_attrs.get('relation', 'related')
            net.add_edge(
                str(source),
                str(target),
                title=relation,
                label=relation[:15],
                arrows='to',
                color='#808080',
                width=0.5 + min(edge_attrs.get('weight', 1) * 0.5, 3)  # Scale edge width by weight
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
    
    # Create and configure the pipeline
    print("Creating preprocessing pipeline...")
    pipeline = create_default_pipeline(output_dir=output_dir)
    
    # Process the text
    print("Processing text (this may take a minute)...")
    result = pipeline.process_text(
        text,
        clean_text=True,
        extract_kg=True,
        save_output=True,
        output_prefix="war_and_peace"
    )
    
    # Save the knowledge graph data
    kg_file = output_dir / "war_and_peace_kg.json"
    with open(kg_file, 'w', encoding='utf-8') as f:
        json.dump(nx.node_link_data(result['knowledge_graph']), f, indent=2)
    
    print(f"\n=== Processing Complete ===")
    print(f"Original text length: {len(text)} characters")
    print(f"Cleaned text length: {len(result.get('cleaned_text', ''))} characters")
    
    # Display entity statistics
    entities = result.get('entities', [])
    if entities:
        print(f"\nExtracted {len(entities)} entities")
        print("Top 10 entities:")
        from collections import Counter
        entity_types = Counter([e['type'] for e in entities])
        for etype, count in entity_types.most_common(10):
            print(f"  - {etype}: {count}")
    
    # Visualize the knowledge graph
    print("\nGenerating knowledge graph visualization...")
    output_html = output_dir / "war_and_peace_kg.html"
    visualize_kg(nx.node_link_data(result['knowledge_graph']), output_html)
    
    print("\n=== Output Files ===")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"  - war_and_peace_cleaned.txt: Cleaned text")
    print(f"  - war_and_peace_entities.json: Extracted entities")
    print(f"  - war_and_peace_relations.json: Extracted relations")
    print(f"  - war_and_peace_kg.json: Knowledge graph data")
    print(f"  - war_and_peace_kg.html: Interactive visualization")

if __name__ == "__main__":
    try:
        import networkx as nx
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "pyvis", "matplotlib"])
    
    main()
