# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Neural entropy model for sentence boundary detection during KG traversal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np

class GraphEncoder(nn.Module):
    """
    Graph encoder that processes the local neighborhood around a node.
    """
    def __init__(self, 
                 node_feature_dim: int = 100,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_embedding = nn.Embedding(10000, node_feature_dim)  # Fixed vocab size for demo
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = node_feature_dim if i == 0 else hidden_dim * heads
            self.convs.append(
                GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
            )
        
        # Edge type embeddings (for relation types)
        self.edge_type_embeddings = nn.Embedding(100, hidden_dim * heads)  # Fixed relation types
        
        # Output projection
        self.proj = nn.Linear(hidden_dim * heads, hidden_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Process a batch of graph neighborhoods.
        
        Args:
            data: PyG Data object with:
                - x: Node indices [num_nodes]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge type indices [num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            Node representations [batch_size, hidden_dim]
        """
        x = self.node_embedding(data.x)  # [num_nodes, node_feature_dim]
        
        # Get edge type embeddings
        edge_attr = self.edge_type_embeddings(data.edge_attr)  # [num_edges, hidden_dim * heads]
        
        # Apply GAT layers
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, data.edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, data.edge_index, edge_attr=edge_attr)
        
        # Pool to get graph-level representation
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_dim * heads]
        
        # Project to hidden_dim
        x = self.proj(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x

class TraversalLSTM(nn.Module):
    """
    LSTM that processes the traversal path.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Will be bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            lengths: Actual sequence lengths for packed sequence
            
        Returns:
            Tuple of (outputs, (h_n, c_n))
        """
        if lengths is not None:
            # Sort by length
            lengths, perm_idx = lengths.sort(0, descending=True)
            x = x[perm_idx]
            
            # Pack sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # Apply LSTM
        outputs, (h_n, c_n) = self.lstm(x)
        
        if lengths is not None:
            # Unpack and unsort
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            _, unperm_idx = perm_idx.sort(0)
            outputs = outputs[unperm_idx]
            h_n = h_n[:, unperm_idx]
            c_n = c_n[:, unperm_idx]
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        return outputs, (h_n, c_n)

class NeuralEntropyModel(nn.Module):
    """
    Neural model for predicting traversal decisions and boundary detection.
    """
    def __init__(self,
                 node_feature_dim: int = 100,
                 graph_hidden_dim: int = 256,
                 lstm_hidden_dim: int = 256,
                 num_relations: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=graph_hidden_dim // 4,  # Will be scaled by heads in GAT
            num_layers=2,
            heads=4,
            dropout=dropout
        )
        
        # Path encoder (bidirectional LSTM)
        self.path_encoder = TraversalLSTM(
            input_dim=graph_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # Boundary classifier
        self.boundary_classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Next node predictor (for training)
        self.next_node_predictor = nn.Sequential(
            nn.Linear(lstm_hidden_dim + graph_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_relations)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, 
               graph_data: Data,
               path_indices: torch.Tensor,
               path_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            graph_data: PyG Data object for the current graph
            path_indices: Indices of nodes in the current path [batch_size, max_path_len]
            path_lengths: Actual lengths of each path [batch_size]
            
        Returns:
            Dict with:
                - boundary_probs: Probability of boundary after each step [batch_size, max_path_len]
                - next_node_logits: Logits for next node prediction [batch_size, num_relations]
                - path_embeddings: Final path embeddings [batch_size, lstm_hidden_dim]
        """
        batch_size, max_path_len = path_indices.size()
        
        # Encode the graph
        graph_embeddings = self.graph_encoder(graph_data)  # [num_nodes, hidden_dim]
        
        # Get node embeddings for the path
        path_embeddings = []
        for i in range(max_path_len):
            # Get embeddings for the i-th node in each path
            node_embs = graph_embeddings[path_indices[:, i]]  # [batch_size, hidden_dim]
            path_embeddings.append(node_embs.unsqueeze(1))
        
        # Stack to [batch_size, max_path_len, hidden_dim]
        path_embeddings = torch.cat(path_embeddings, dim=1)
        
        # Encode the path
        path_outputs, (h_n, _) = self.path_encoder(
            path_embeddings, path_lengths)  # [batch_size, max_path_len, lstm_hidden_dim]
        
        # Predict boundary probabilities
        boundary_probs = self.boundary_classifier(path_outputs).squeeze(-1)  # [batch_size, max_path_len]
        
        # Predict next node (for training)
        if self.training and path_lengths is not None:
            # Get the last valid hidden state for each sequence
            last_indices = (path_lengths - 1).view(-1, 1, 1).expand(-1, 1, path_outputs.size(-1))
            last_hidden = path_outputs.gather(1, last_indices).squeeze(1)  # [batch_size, lstm_hidden_dim]
            
            # Get the last node's graph embedding
            last_node_indices = path_indices.gather(1, (path_lengths - 1).view(-1, 1)).squeeze(1)
            last_node_embs = graph_embeddings[last_node_indices]  # [batch_size, hidden_dim]
            
            # Predict next node
            next_node_logits = self.next_node_predictor(
                torch.cat([last_hidden, last_node_embs], dim=1))  # [batch_size, num_relations]
        else:
            next_node_logits = None
        
        return {
            'boundary_probs': boundary_probs,
            'next_node_logits': next_node_logits,
            'path_embeddings': h_n.permute(1, 0, 2).reshape(batch_size, -1)  # [batch_size, num_layers * hidden_dim]
        }
    
    def predict_boundary(self, 
                        graph_data: Data, 
                        path_indices: torch.Tensor,
                        path_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict boundary probability for the last node in each path.
        
        Args:
            graph_data: PyG Data object for the current graph
            path_indices: Indices of nodes in the current path [batch_size, max_path_len]
            path_lengths: Actual lengths of each path [batch_size]
            
        Returns:
            Boundary probabilities [batch_size]
        """
        self.eval()
        with torch.no_grad():
            outputs = self(graph_data, path_indices, path_lengths)
            if path_lengths is not None:
                # Get the boundary probability at the last valid position
                last_indices = (path_lengths - 1).unsqueeze(1)
                boundary_probs = outputs['boundary_probs'].gather(1, last_indices).squeeze(1)
            else:
                # If no lengths provided, assume all paths are full length
                boundary_probs = outputs['boundary_probs'][:, -1]
        return boundary_probs
