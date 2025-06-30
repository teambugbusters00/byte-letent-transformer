#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Visualize the sample knowledge graph from demo_output/sample_kg.json
"""

import json
import webbrowser
from pathlib import Path

def main():
    # Paths
    kg_file = Path("demo_output/sample_kg.json")
    output_file = Path("demo_output/kg_visualization.html")
    
    print(f"Loading knowledge graph from {kg_file}...")
    
    try:
        # Load the knowledge graph data
        with open(kg_file, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        print("Generating visualization...")
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a pyvis network
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
        
        # Convert node-link data to NetworkX graph
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
            relation = edge_attrs.get('relation', 'related')
            net.add_edge(
                str(source),
                str(target),
                title=relation,
                label=relation[:15],  # Limit label length
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
        
    except FileNotFoundError:
        print(f"Error: File not found: {kg_file}")
        print("Please run 'python demo/preprocess_demo.py' first to generate the knowledge graph.")
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install the required packages for visualization:")
        print("pip install networkx pyvis")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
