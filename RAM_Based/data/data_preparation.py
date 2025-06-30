import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from dataloader import IGL260M
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
import logging

def check_consistence(mode: str, batch_order: str):
    assert mode in ['ppr', 'rand', 'randfix', 'part',
                    'clustergcn', 'n_sampling', 'rw_sampling', 'ladies', 'ppr_shadow']
    if mode in ['ppr', 'part', 'randfix',]:
        assert batch_order in ['rand', 'sample', 'order']
    else:
        assert batch_order == 'rand'

def find_isolated_nodes(edge_index, num_nodes):
    # Step 1: Collect all unique nodes from edge_index
    # edge_index has shape (2, num_edges), so we collect all unique node ids
    nodes_in_edges = torch.unique(edge_index)

    # Step 2: Create a tensor of all node indices from 0 to num_nodes-1
    all_nodes = torch.arange(num_nodes, device=edge_index.device)

    # Step 3: Find isolated nodes by checking which nodes are not in nodes_in_edges
    isolated_nodes = torch.setdiff1d(all_nodes, nodes_in_edges)

    return isolated_nodes


def load_data(dataset_name: str,
              small_trainingset: float,
              pretransform):
    """

    :param dataset_name:
    :param small_trainingset:
    :param pretransform:
    :return:
    """
    if dataset_name.lower() in ['arxiv', 'products', 'papers100m']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name),
                                         root='./datasets',
                                         pre_transform=pretransform)
        split_idx = dataset.get_idx_split()
        graph = dataset[0]
        logging.info(f" graph is {graph}")
    elif dataset_name.lower().startswith('reddit'):
        if dataset_name == 'reddit2':
            dataset = Reddit2('./datasets/reddit2', pre_transform=pretransform)
        elif dataset_name == 'reddit':
            dataset = Reddit('./datasets/reddit', pre_transform=pretransform)
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        graph.train_mask, graph.val_mask, graph.test_mask = None, None, None
    else:
        args = []
        if dataset_name == 'tiny':
            dataset = IGL260M(args)
        else:
            dataset = IGL260M(args)
        print(dataset[0])
        paper_label = torch.tensor(dataset.paper_label, dtype=torch.long).unsqueeze(1)
        paper_feat = torch.tensor(dataset.paper_feat, dtype=torch.float)
        logging.info(f" x shape is {paper_feat.shape} with max being {paper_feat.max()} and min being {paper_feat.min()}")
        paper_edge = torch.tensor(dataset.paper_edge, dtype=torch.long)
        edge_index = paper_edge.t()     
        num_nodes = paper_feat.size(0)
        logging.info(f"num_nodes is {num_nodes}")
        logging.info(f" edge_indx shape is {edge_index.shape}")
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
        data = Data(x=paper_feat, y=paper_label, adj_t=adj_t, edge_index = edge_index)
        graph = data
        split_idx = {}
        n_nodes = data.num_nodes
        if ('full' in dataset_name):
            n_labeled_idx = 227130858
        else:
            n_labeled_idx = data.y.size(0)
        n_train = int(n_labeled_idx * 0.6)
        n_val = int(n_labeled_idx * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask[n_train:n_train + n_val] = True
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask[n_train + n_val:n_labeled_idx] = True
        nodes_in_edge_index = torch.unique(edge_index)
        
        # Find the set of all nodes in the graph
        all_nodes = torch.arange(num_nodes)
        
        isolated_mask = ~torch.isin(all_nodes, nodes_in_edge_index)
        isolated_nodes = all_nodes[isolated_mask]
        #set the mask to 0 for all isolated nodes
        train_mask[isolated_nodes] = False
        val_mask[isolated_nodes] = False
        test_mask[isolated_nodes] = False
        split_idx['train'] = torch.where(train_mask)[0]
        split_idx['valid'] = torch.where(val_mask)[0]
        split_idx['test'] = torch.where(test_mask)[0]
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        
        # Find isolated nodes (nodes that are not in edge_index)
        
        print("Isolated nodes:", isolated_nodes.size(0)) 

    train_indices = split_idx["train"].numpy()

    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices,
                                                 size=int(len(train_indices) * small_trainingset),
                                                 replace=False,
                                                 p=None))

    train_indices = torch.from_numpy(train_indices)

    val_indices = split_idx["valid"]
    test_indices = split_idx["test"]
    return graph, (train_indices, val_indices, test_indices,)


class GraphPreprocess:
    def __init__(self,
                 self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        if self.self_loop:
            edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        else:
            edge_index = graph.edge_index

        if self.to_undirected:
            edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

        graph.edge_index = edge_index
        return graph
