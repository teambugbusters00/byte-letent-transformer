# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Knowledge graph traversal with entropy-based guidance.
"""

import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from collections import defaultdict, deque

class KnowledgeGraphTraverser:
    """
    Traverses a knowledge graph using entropy-based guidance.
    
    This class implements methods to traverse a knowledge graph while considering
    the entropy of transitions between nodes, which can be used to identify
    meaningful paths or boundaries in the underlying text.
    """
    
    def __init__(self, graph: nx.Graph, entropy_scores: Dict[str, float]):
        """
        Initialize the graph traverser.
        
        Args:
            graph: A NetworkX graph where nodes are tokens
            entropy_scores: Dictionary mapping tokens to their entropy scores
        """
        self.graph = graph
        self.entropy_scores = entropy_scores
        self._precompute_node_entropy()
    
    def _precompute_node_entropy(self) -> None:
        """Precompute entropy scores for all nodes in the graph."""
        self.node_entropy = {}
        for node in self.graph.nodes():
            self.node_entropy[node] = self.entropy_scores.get(node, 0.0)
    
    def get_entropy_weighted_neighbors(self, node: str) -> List[Tuple[str, float]]:
        """
        Get neighbors of a node, weighted by entropy.
        
        Args:
            node: The node to get neighbors for
            
        Returns:
            List of (neighbor, weight) tuples, where weight is based on edge weight and entropy
        """
        neighbors = []
        for neighbor in self.graph.neighbors(node):
            # Get edge weight (default to 1 if no weight)
            weight = self.graph.get_edge_data(node, neighbor, {}).get('weight', 1.0)
            
            # Adjust weight by neighbor's entropy
            entropy = self.node_entropy.get(neighbor, 0.0)
            adjusted_weight = weight * (1.0 + entropy)  # Higher entropy = more exploration
            
            neighbors.append((neighbor, adjusted_weight))
        
        return neighbors
    
    def traverse_from_node(
        self, 
        start_node: str, 
        max_steps: int = 10,
        entropy_threshold: float = 0.7,
        max_entropy_jumps: int = 3
    ) -> List[str]:
        """
        Traverse the graph starting from a given node.
        
        Args:
            start_node: The starting node
            max_steps: Maximum number of steps to traverse
            entropy_threshold: Threshold for considering a high-entropy transition
            max_entropy_jumps: Maximum number of high-entropy transitions to allow
            
        Returns:
            List of nodes in the traversal path
        """
        if start_node not in self.graph:
            return []
        
        path = [start_node]
        current_node = start_node
        entropy_jumps = 0
        
        for _ in range(max_steps - 1):
            # Get entropy-weighted neighbors
            neighbors = self.get_entropy_weighted_neighbors(current_node)
            if not neighbors:
                break
                
            # Sort by weight (descending)
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Check if the best move is a high-entropy transition
            next_node, weight = neighbors[0]
            
            # Check if this is a high-entropy transition
            if self.node_entropy.get(next_node, 0) > entropy_threshold:
                entropy_jumps += 1
                if entropy_jumps > max_entropy_jumps:
                    break  # Stop after too many high-entropy transitions
            
            # Move to the next node
            path.append(next_node)
            current_node = next_node
        
        return path
    
    def find_high_entropy_paths(
        self, 
        start_node: str,
        max_depth: int = 3,
        min_entropy: float = 0.7
    ) -> List[List[str]]:
        """
        Find all paths from start_node that end with a high-entropy node.
        
        Args:
            start_node: The starting node
            max_depth: Maximum path length
            min_entropy: Minimum entropy for a node to be considered a boundary
            
        Returns:
            List of paths, where each path ends with a high-entropy node
        """
        if start_node not in self.graph:
            return []
        
        paths = []
        queue = deque([([start_node], 0)])
        
        while queue:
            path, depth = queue.popleft()
            current_node = path[-1]
            
            # Check if current node is a high-entropy boundary
            if depth > 0 and self.node_entropy.get(current_node, 0) >= min_entropy:
                paths.append(path)
                continue  # Don't extend past a boundary
            
            # Don't extend beyond max depth
            if depth >= max_depth:
                continue
            
            # Add neighbors to the queue
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in path:  # Avoid cycles
                    queue.append((path + [neighbor], depth + 1))
        
        return paths
    
    def find_sentence_boundaries(
        self, 
        token_sequence: List[str],
        window_size: int = 3,
        min_entropy: float = 0.7
    ) -> List[int]:
        """
        Identify sentence boundaries in a sequence of tokens.
        
        Args:
            token_sequence: List of tokens
            window_size: Size of the sliding window to consider
            min_entropy: Minimum entropy to consider a boundary
            
        Returns:
            List of indices where sentence boundaries occur
        """
        boundaries = []
        
        for i in range(len(token_sequence) - window_size + 1):
            window = token_sequence[i:i + window_size]
            
            # Calculate average entropy in the window
            entropy_sum = 0.0
            for token in window:
                entropy_sum += self.node_entropy.get(token, 0.0)
            avg_entropy = entropy_sum / len(window)
            
            # If average entropy exceeds threshold, mark as boundary
            if avg_entropy >= min_entropy:
                boundary_pos = i + window_size // 2  # Middle of window
                boundaries.append(boundary_pos)
        
        return boundaries
