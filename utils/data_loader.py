import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_predata(filepath='data/predata.pkl'):
    """
    Load preprocessed RNA data from pickle file.
    
    Returns:
        rna_seqs: List of RNA sequences
        bpp_matrices: List of base pair probability matrices
        embeddings: List of RNA embeddings
        activities: List of activity labels
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    rna_seqs,  embeddings, bpp_matrices, activities = data
    
    return rna_seqs, bpp_matrices, embeddings, activities


def construct_graph_from_bpp(embedding, bpp_matrix, threshold=1e-6):
    """
    Construct PyG Data object from RNA embedding and BPP matrix.
    
    Args:
        embedding: Node features (N, node_dim)
        bpp_matrix: Base pair probability matrix (N, N)
        threshold: Minimum edge weight to include
    
    Returns:
        Data: PyG Data object with node features and edges
    """
    # Node features include diagonal BPP values as unpaired probability.
    diag = torch.FloatTensor(np.diag(bpp_matrix).copy()).unsqueeze(-1)
    x = torch.cat([torch.FloatTensor(embedding.copy()), diag], dim=1)
    
    # Build graph edges from sequence order and pair probability.
    edge_list = []
    edge_weights = []
    
    # Sequence adjacency: each nucleotide connects to neighbors in RNA order.
    seq_weight = 1.0
    for i in range(bpp_matrix.shape[0] - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
        edge_weights.extend([seq_weight, seq_weight])
    
    # Pairing edges from non-zero BPP entries.
    for i in range(bpp_matrix.shape[0]):
        for j in range(i + 1, bpp_matrix.shape[1]):
            weight = float(bpp_matrix[i, j])
            if weight > threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_weights.extend([weight, weight])
    
    edge_index = torch.LongTensor(edge_list).t().contiguous() if len(edge_list) > 0 else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.FloatTensor(edge_weights) if len(edge_weights) > 0 else None
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    return data


def create_dataset(rna_seqs, bpp_matrices, embeddings, activities, threshold=1e-6):
    """
    Create PyG dataset from preprocessed RNA data.
    
    Args:
        rna_seqs: List of RNA sequences or None
        bpp_matrices: List of BPP matrices
        embeddings: List of embeddings
        activities: List of activity labels
        threshold: Edge weight threshold
    
    Returns:
        List of PyG Data objects
    """
    dataset = []
    
    for i, (emb, bpp, activity) in enumerate(zip(embeddings, bpp_matrices, activities)):
        try:
            data = construct_graph_from_bpp(emb, bpp, threshold=threshold)
            data.y = torch.FloatTensor([activity])
            data.sample_index = i
            if rna_seqs is not None:
                data.rna_seq = rna_seqs[i]
            dataset.append(data)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    return dataset


def get_dataloader(rna_seqs, bpp_matrices, embeddings, activities,
                   batch_size=32, train_ratio=0.8, shuffle=True):
    """
    Create train and test dataloaders.
    
    Args:
        embeddings: List of embeddings
        bpp_matrices: List of BPP matrices
        activities: List of activities
        batch_size: Batch size
        train_ratio: Train/test split ratio
        shuffle: Whether to shuffle data
    
    Returns:
        train_loader, test_loader: PyG DataLoaders
    """
    dataset = create_dataset(rna_seqs, bpp_matrices, embeddings, activities)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_data(rna_seqs, bpp_matrices, embeddings, activities,
               train_ratio=0.8, seed=42):
    """
    Split data into train and test sets.
    """
    np.random.seed(seed)
    n_samples = len(activities)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_data = {
        'rna_seqs': [rna_seqs[i] for i in train_idx],
        'bpp_matrices': [bpp_matrices[i] for i in train_idx],
        'embeddings': [embeddings[i] for i in train_idx],
        'activities': [activities[i] for i in train_idx]
    }
    
    test_data = {
        'rna_seqs': [rna_seqs[i] for i in test_idx],
        'bpp_matrices': [bpp_matrices[i] for i in test_idx],
        'embeddings': [embeddings[i] for i in test_idx],
        'activities': [activities[i] for i in test_idx]
    }
    
    return train_data, test_data
