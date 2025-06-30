import argparse
import numpy as np
import torch
import os.path as osp
import logging
import dgl
from dgl.data import DGLDataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, matmul
import warnings
warnings.filterwarnings("ignore")

class IGL260MDataset(object):
    def __init__(self, root: str, size: str, in_memory: int, classes: int, cache_size: int):
        self.dir = root
        self.size = size
        self.in_memory = in_memory
        self.num_classes = classes
        self.cache_size = cache_size
        self.cache = {}
        self.node_id = 0
        #self.initialize_cache()

    def num_features(self) -> int:
        return 1024

    def num_classes(self, type_of_class: str) -> int:
        return 19

    @property
    def paper_feat(self) -> np.ndarray:
        # define the path to be relative to the directory of this file and then dataset/large/processed/paper/node_feat.npy
        code_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(code_path, 'dataset', 'large', 'processed', 'paper', 'node_feat.npy')
        node_features = np.memmap(path, dtype='float32', mode='r',  shape=(100000000,1024))
        return node_features

    @property
    def paper_label(self) -> np.ndarray:
        if self.num_classes == 19:
            code_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(code_path, 'dataset', 'large', 'processed', 'paper', 'node_label_19l.npy')
        else:
            code_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(code_path, 'dataset', 'large', 'processed', 'paper', 'node_label_2K.npy')

        
        return np.load(path, allow_pickle = True)

    @property
    def paper_edge(self) -> np.ndarray:
        code_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(code_path, 'dataset', 'large', 'processed', 'paper__cites__paper', 'edge_index.npy')
        return np.load(path)

    def initialize_cache(self):
        """Preload high PageRank score node features into cache."""
        print("Initializing cache based on PageRank...")
        edge_index = torch.from_numpy(self.paper_edge).t()
        num_nodes = self.paper_feat.shape[0]
    
        # Calculate adjacency matrix
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    
        # Calculate PageRank scores
        pagerank_scores = self.calculate_pagerank(adj)
    
        # Get top nodes based on PageRank scores
        top_pagerank_nodes = torch.topk(pagerank_scores, self.cache_size).indices
    
        for node_id in top_pagerank_nodes:
            self.cache[int(node_id)] = torch.tensor(self.paper_feat[int(node_id)])
        self.node_id = node_id
        print(f"Cache initialized with {len(self.cache)} nodes based on PageRank.")

    def calculate_pagerank(self, adj: SparseTensor, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100):
        """Calculate PageRank scores using power iteration."""
        num_nodes = adj.size(0)
        rank = torch.ones(num_nodes, dtype=torch.float32).view(-1, 1) / num_nodes  # Make rank 2D
    
        # Row-normalize adjacency matrix
        adj = adj.set_value(None)  # Treat as unweighted graph
        row_sum = adj.sum(dim=1).to(torch.float32)
        row_sum_inv = 1.0 / torch.clamp(row_sum, min=1e-10)
        normalized_adj = adj.mul(row_sum_inv.view(-1, 1))
    
        for _ in range(max_iter):
            prev_rank = rank.clone()
            rank = alpha * normalized_adj.matmul(prev_rank) + (1 - alpha) / num_nodes
            if torch.norm(rank - prev_rank, p=1) < tol:
                break
    
        return rank.view(-1)  # Return rank as a 1D tensor
    
    def get_node_feature(self, node_id: int):
        """Retrieve node features with caching logic."""
        if node_id in self.cache:
            return self.cache[node_id]
        else:
            return torch.tensor(self.paper_feat[node_id])


class IGL260M(DGLDataset):
    def __init__(self, cache_size):
        super().__init__(name='IGB260M')
        self.cache_size = cache_size
        self.dataset = None

    def process(self):
        self.dataset = IGL260MDataset(root='/scratch/meghbal/gnnSZ/dataset', size='small', in_memory=0, classes=19, cache_size=10000000)
        node_features = torch.from_numpy(self.dataset.paper_feat)
        node_edges = torch.from_numpy(self.dataset.paper_edge).t()
        node_labels = torch.from_numpy(self.dataset.paper_label).to(torch.long)
        n_nodes = node_features.shape[0]
        print(f"edge shape: {node_edges.shape}")
        print(f"{self.dataset}")
        remove_self_loops(node_edges)
        add_self_loops(node_edges)

        adj_t = SparseTensor.from_edge_index(node_edges, sparse_sizes=(n_nodes, n_nodes))
        self.graph_g = Data(x=node_features, y=node_labels, adj_t=adj_t)
        self.graph_g.num_node_features = 1024

        #self.graph.ndata['feat'] = node_features
        ##check the type of ndata 
        #print(f"ndata type: {type(self.graph.ndata)}")
        #self.graph.ndata['label'] = node_labels

        #self.graph = dgl.remove_self_loop(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)

        #n_train = int(n_nodes * 0.6)
        #n_val = int(n_nodes * 0.2)

        #train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        #train_mask[:n_train] = True
        #val_mask[n_train:n_train + n_val] = True
        #test_mask[n_train + n_val:] = True

        #self.graph.ndata['train_mask'] = train_mask
        #self.graph.ndata['val_mask'] = val_mask
        #self.graph.ndata['test_mask'] = test_mask
    
    def get_node_feature(self, node_id: int):
        """Delegate to the IGL260MDataset instance."""
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized. Call `process()` first.")
        return self.dataset.get_node_feature(node_id)
    
    def __getitem__(self, i):
        return self.graph_g

    def __len__(self):
        return 1

