import argparse
import numpy as np
import torch
import os.path as osp

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")
class IGL260MDataset(object):
    def __init__(self, root: str, size: str, in_memory: int, classes: int):
        self.dir = root
        self.size = size
        self.in_memory = in_memory
        self.num_classes = classes
    

    def num_features(self) -> int:
        return 1024
    

    def num_classes(self, type_of_class: str) -> int:
        if type_of_class == 'small':
            return 19
        else:
            return 2983

    @property
    def paper_feat(self) -> np.ndarray:
        path = '/home/meghbal/ibmb/datasets/small/processed/paper/node_feat.npy'
        node_features = np.memmap(path, dtype='float32', mode='r',  shape=(1000000,1024))
        # node_features_unlabelled =  np.memmap('/mnt/nvme14/IGB260M_part_1/processed/paper/node_feat.npy', dtype='float32', mode='r',  shape=(111005234,1024))
        # node_features[157675969:,:] = node_features_unlabelled
        return node_features

    @property
    def paper_label(self) -> np.ndarray:
        if self.num_classes == 19:
            path = '/home/meghbal/ibmb/datasets/small/processed/paper/node_label_19.npy'
        else:
            path = '/home/meghbal/ibmb/datasets/small/processed/paper/node_label_2K.npy'

        
        node_features = np.memmap(path, dtype='float32', mode='r',  shape=(1000000))
        return node_features
        # if self.in_memory:
        #     return np.load(path)
        # else:
        #     return np.load(path, mmap_mode='r')

    @property
    def paper_edge(self) -> np.ndarray:
        # path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        path = '/home/meghbal/ibmb/datasets/small/processed/paper__cites__paper/edge_index.npy'
        return np.load(path, mmap_mode='r')
        # if self.in_memory:
        #     return np.load(path)
        # else:
        #     return np.load(path, mmap_mode='r')

class IGL260M(DGLDataset):
    def __init__(self, args):
        super().__init__(name='IGB260M')

    def process(self):
        dataset = IGL260MDataset(root='/home/meghbal/ibmb/datasets/', size='small', in_memory=0, classes=19)
        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])

        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)
        
        n_nodes = node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
