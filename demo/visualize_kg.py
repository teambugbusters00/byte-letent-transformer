#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Standalone script to visualize a knowledge graph from saved JSON files.
"""

import json
import sys
import webbrowser
from pathlib import Path

def load_kg_data(kg_file: Path):
    """Load knowledge graph data from a JSON file."""
    try:
        with open(kg_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {kg_file}: {e}")
        sys.exit(1)

def visualize_kg(kg_data, output_file: Path):
    """Visualize a knowledge graph using pyvis."""
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
            node_label = node if len(str(node)) < 20 else f"{str(node)[:17]}..."
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
                label=edge_attrs.get('relation', '')[:15],  # Limit label length
                arrows='to',
                color='#808080'
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
    # Default input and output paths
    default_kg_file = Path("demo_output/sample_kg.json")
    default_output = Path("kg_visualization.html")
    
    # Get input file path
    kg_file = input(f"Enter path to knowledge graph JSON file [{default_kg_file}]: ").strip()
    kg_file = Path(kg_file) if kg_file else default_kg_file
    
    if not kg_file.exists():
        print(f"Error: File not found: {kg_file}")
        sys.exit(1)
    
    # Get output file path
    output_file = input(f"Enter output HTML file path [{default_output}]: ").strip()
    output_file = Path(output_file) if output_file else default_output
    
    # Load and visualize the knowledge graph
    print(f"Loading knowledge graph from {kg_file}...")
    kg_data = load_kg_data(kg_file)
    
    print("Generating visualization...")
    visualize_kg(kg_data, output_file)

if __name__ == "__main__":
    main()
