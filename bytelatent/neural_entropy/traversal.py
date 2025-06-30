# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Knowledge Graph traversal with neural boundary detection.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import networkx as nx
from tqdm import tqdm

from torch_geometric.data import Data

class KGGraphTraverser:
    """
    Traverses a knowledge graph using a neural model to detect sentence boundaries.
    """
    
    def __init__(self, 
                 model: 'NeuralEntropyModel',
                 graph: nx.Graph,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_steps: int = 20,
                 boundary_threshold: float = 0.7,
                 max_entropy: float = 0.9,
                 beam_size: int = 3):
        """
        Initialize the graph traverser.
        
        Args:
            model: Trained NeuralEntropyModel
            graph: NetworkX knowledge graph
            device: Device to run the model on
            max_steps: Maximum number of steps to traverse
            boundary_threshold: Probability threshold to consider a boundary
            max_entropy: Maximum allowed entropy before stopping
            beam_size: Number of paths to keep in beam search
        """
        self.model = model.to(device)
        self.graph = graph
        self.device = device
        self.max_steps = max_steps
        self.boundary_threshold = boundary_threshold
        self.max_entropy = max_entropy
        self.beam_size = beam_size
        
        # Preprocess graph for faster lookups
        self._preprocess_graph()
    
    def _preprocess_graph(self) -> None:
        """Preprocess the graph for faster traversal."""
        # Create node and relation indices
        self.node_to_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        
        # Create relation indices
        all_relations = set()
        for _, _, data in self.graph.edges(data=True):
            all_relations.add(data.get('relation', 'unknown'))
        self.relation_to_idx = {rel: i for i, rel in enumerate(all_relations)}
        
        # Create edge index and edge attributes for PyG
        edge_indices = []
        edge_attrs = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_indices.append((self.node_to_idx[u], self.node_to_idx[v]))
            rel = data.get('relation', 'unknown')
            edge_attrs.append(self.relation_to_idx[rel])
        
        self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    
    def _create_graph_data(self) -> Data:
        """Create a PyG Data object from the graph."""
        # Create node features (simple one-hot for now)
        x = torch.eye(len(self.node_to_idx))
        
        return Data(
            x=x.to(self.device),
            edge_index=self.edge_index.to(self.device),
            edge_attr=self.edge_attr.to(self.device)
        )
    
    def traverse_from_node(self, 
                         start_node: str,
                         max_steps: Optional[int] = None,
                         return_entropy: bool = True) -> List[Dict[str, Any]]:
        """
        Traverse the graph starting from a node, using beam search.
        
        Args:
            start_node: Starting node for traversal
            max_steps: Maximum number of steps to traverse (overrides init if provided)
            return_entropy: Whether to include entropy in the output
            
        Returns:
            List of traversal paths, each with nodes, edges, and boundary probabilities
        """
        if max_steps is None:
            max_steps = self.max_steps
        
        # Initialize beam with the start node
        start_idx = self.node_to_idx[start_node]
        beam = [{
            'nodes': [start_idx],
            'edges': [],
            'probs': [],
            'score': 0.0,
            'stopped': False
        }]
        
        graph_data = self._create_graph_data()
        
        for step in range(max_steps):
            new_beam = []
            
            for path in beam:
                if path['stopped']:
                    new_beam.append(path)
                    continue
                
                current_node = path['nodes'][-1]
                
                # Get neighbors
                neighbors = []
                for _, v, data in self.graph.edges(self.idx_to_node[current_node], data=True):
                    if v != current_node:  # Avoid self-loops
                        neighbors.append((self.node_to_idx[v], 
                                       self.relation_to_idx[data.get('relation', 'unknown')]))
                
                if not neighbors:
                    path['stopped'] = True
                    new_beam.append(path)
                    continue
                
                # Create batch for model
                batch_size = len(neighbors)
                path_indices = torch.zeros((batch_size, len(path['nodes'])), 
                                         dtype=torch.long, device=self.device)
                path_lengths = torch.full((batch_size,), len(path['nodes']), 
                                        dtype=torch.long, device=self.device)
                
                for i in range(batch_size):
                    path_indices[i, :len(path['nodes'])] = torch.tensor(path['nodes'], 
                                                                      device=self.device)
                
                # Predict boundary probabilities
                with torch.no_grad():
                    outputs = self.model(graph_data, path_indices, path_lengths)
                    boundary_probs = outputs['boundary_probs'][:, -1].cpu().numpy()
                
                # Add new paths to beam
                for i, (neighbor_idx, rel_idx) in enumerate(neighbors):
                    boundary_prob = boundary_probs[i]
                    
                    new_path = {
                        'nodes': path['nodes'] + [neighbor_idx],
                        'edges': path['edges'] + [rel_idx],
                        'probs': path['probs'] + [boundary_prob],
                        'score': path['score'] - np.log(1e-10 + 1 - boundary_prob),
                        'stopped': boundary_prob > self.boundary_threshold
                    }
                    new_beam.append(new_path)
            
            # Keep top-k paths
            new_beam.sort(key=lambda x: x['score'], reverse=True)
            beam = new_beam[:self.beam_size]
            
            # Stop if all paths are complete
            if all(path['stopped'] for path in beam):
                break
        
        # Convert indices back to node/relation names
        result = []
        for path in beam:
            nodes = [self.idx_to_node[idx] for idx in path['nodes']]
            edges = [list(self.relation_to_idx.keys())[list(self.relation_to_idx.values()).index(idx)] 
                    for idx in path['edges']]
            
            result_path = {
                'nodes': nodes,
                'edges': edges,
                'boundary_probabilities': path['probs'],
                'score': path['score']
            }
            
            if return_entropy:
                # Calculate entropy for each step
                entropies = []
                for i in range(1, len(nodes)):
                    # Simple entropy calculation based on transition probabilities
                    neighbors = list(self.graph.neighbors(nodes[i-1]))
                    if len(neighbors) <= 1:
                        entropies.append(0.0)
                        continue
                    
                    # Count transitions (simplified - could use model's predictions)
                    transition_counts = defaultdict(int)
                    for _, v, data in self.graph.edges(nodes[i-1], data=True):
                        rel = data.get('relation', 'unknown')
                        transition_counts[rel] += 1
                    
                    # Calculate entropy
                    total = sum(transition_counts.values())
                    entropy = 0.0
                    for count in transition_counts.values():
                        p = count / total
                        entropy -= p * np.log2(p) if p > 0 else 0
                    
                    # Normalize by max possible entropy
                    max_entropy = np.log2(len(transition_counts)) if len(transition_counts) > 1 else 1
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    entropies.append(min(normalized_entropy, 1.0))
                
                result_path['entropies'] = entropies
            
            result.append(result_path)
        
        return result
    
    def find_sentence_boundaries(self, 
                               start_nodes: List[str],
                               max_steps: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Find sentence boundaries starting from multiple nodes.
        
        Args:
            start_nodes: List of starting nodes
            max_steps: Maximum number of steps to traverse
            
        Returns:
            Dictionary mapping start nodes to their traversal results
        """
        results = {}
        
        for node in tqdm(start_nodes, desc="Traversing from start nodes"):
            if node not in self.node_to_idx:
                print(f"Warning: Node '{node}' not found in graph")
                results[node] = []
                continue
                
            paths = self.traverse_from_node(node, max_steps=max_steps)
            results[node] = paths
        
        return results
