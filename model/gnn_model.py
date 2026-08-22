import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

class GCNconvConv1d(nn.Module):
    """
    GNN-based RNA activity predictor.
    Architecture: GCN layers -> 1D CNN -> MLP -> Activity prediction
    """
    
    def __init__(self, node_dim=641, hidden_dim=128, gnn_layers=3, 
                 cnn_channels=64, cnn_kernel=3, output_dim=1):
        super(GCNconvConv1d, self).__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # GCN layers for graph convolution
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(gnn_layers)
        ])
        
        # 1D CNN for sequence-like processing
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2),
            nn.ReLU(),
            # nn.AdaptiveAvgPool1d(1)
        )
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(cnn_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x (node features), edge_index, and edge_attr
        Returns:
            predictions: Activity predictions
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # GCN forward pass with edge weights
        for conv in self.gnn_layers:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        x_dense, mask = to_dense_batch(
            x, data.batch
        )
        mask_3d = mask.unsqueeze(1)
        x_cnn = x_dense.transpose(1, 2)
        # 1D CNN: reshape to (batch=1, channels, length)
        x_cnn = self.conv1d(x_cnn)  # (batch, cnn_channels, 1)
        x_cnn = x_cnn * mask_3d

        seq_lengths = mask_3d.sum(dim=-1).clamp(min=1)
        
        x_cnn = x_cnn.sum(dim=-1) / seq_lengths
        
        
        
        
        # MLP prediction
        out = self.mlp(x_cnn)
        
        return out.view(-1) if out.size(-1) == 1 else out


class GCNconv(nn.Module):
    """
    Batch-aware GNN model for predicting ribozyme activity.
    """
    
    def __init__(self, node_dim=641, hidden_dim=128, gnn_layers=3,
                 cnn_channels=64, cnn_kernel=3, output_dim=1):
        super(GCNconv, self).__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(gnn_layers)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object or list of Data objects
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # GCN layers with edge weights
        for conv in self.gnn_layers:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        # MLP
        out = self.mlp(x)
        
        return out.squeeze()

