import logging
import resource
import time
import traceback
import pandas as pd
import os.path as osp
import numpy as np
import seml
import torch
from sacred import Experiment
import os
from dataloaders.get_loaders import get_loaders
from data.data_preparation import check_consistence, load_data, GraphPreprocess
from models.get_model import get_model
from train.trainer import Trainer
import networkx as nx
import networkit as nk 
import gc
import sys
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.typing import SparseTensor
from dataloaders.BaseLoader import BaseLoader
ex = Experiment()
seml.setup_logger(ex)
os.environ["OMP_NUM_THREADS"] = "20"
def which_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def add_to_edge_list(edge_list,u, i):
   edge_list[i].append(u)
   return


@ex.automain
def run(dataset_name,
        mode,
        batch_size,
        micro_batch,
        batch_order,
        inference,
        LBMB_val,
        small_trainingset,

        ppr_params,
        batch_params,
        n_sampling_params=None,
        rw_sampling_params=None,
        ladies_params=None,
        shadow_ppr_params=None,
        rand_ppr_params=None,

        graphmodel='gcn',
        hidden_channels=256,
        reg=0.,
        num_layers=3,
        heads=None,

        epoch_min=1,
        epoch_max=3,
        patience=10,
        lr=1e-3,
        sprs = 0,
        sprs_method = 'SCAN',
        sprs_rate = 1,
        seed=None, ):
    try:
        gc.enable()
        check_consistence(mode, batch_order)
        logging.info(f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}')
        
        graph, (train_indices, val_indices, test_indices) = load_data(dataset_name,
                                                                      small_trainingset,
                                                                      GraphPreprocess(True, True))

        # convert the undirected edge list to directed edge list
        logging.info(f"Graph: {graph}")
        is_graph_directed = 1-is_undirected(graph.edge_index)
        logging.info(f"Input Graph directed ? {is_graph_directed}")
        if (sprs >= 2):
            #convert the graph to networkx graph
            start_time = time.time()
            #check the type of edge_index
            edge_coo = graph.edge_index.cpu().numpy()
            edge_size_before = edge_coo.shape[1]
            G = nk.GraphFromCoo((edge_coo[0], edge_coo[1]), directed= True , weighted=False, n=graph.num_nodes)
            G.indexEdges()

            if sprs_method == 'SCAN':
                scansp = nk.sparsification.SCANSparsifier()
            elif sprs_method == 'LSIM':
                scansp = nk.sparsification.LocalSimilaritySparsifier()
            logging.info(f"SPRS method: {sprs_method}")
            scanG = scansp.getSparsifiedGraphOfSize(G, sprs_rate)
            logging.info(f"Graph number of edges before sparsification: {graph.edge_index.size(1)}")
            logging.info(f"Graph numberOfEdfes: {scanG.numberOfEdges()}")
            # create an empty list of edges with size of [2, num_edges]
            # create a 2d tensor 
            # iterate over the edges of the graph
            # create a list of edgges with size of [2, num_edges] and append it to the edge_list
            #define edge list as a 2d list
            edge_list = [[], []]
            scanG.forEdges(lambda u, v, w, eid: add_to_edge_list(edge_list, u, 0))
            scanG.forEdges(lambda u, v, w, eid: add_to_edge_list(edge_list, v, 1))
            logging.info(f"Graph before sparsification: {graph}")
            #create a tensor from edge_list
            tensor_edge_list = torch.tensor(edge_list, dtype=torch.int64)
            graph.edge_index = tensor_edge_list
            #convert the graph to undirected graph
            logging.info(f"Achieved sprs rate: {graph.edge_index.size(1)/edge_size_before}")
            graph.edge_index = to_undirected(graph.edge_index)
            end_time = time.time()
            logging.info(f"G spar time: {end_time - start_time} seconds")
            logging.info(f"graph is: {graph}")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logging.info(f"After Graph loaded!\n")
        trainer = Trainer(mode,
                          batch_params['num_batches'][0],
                          micro_batch=micro_batch,
                          batch_size=batch_size,
                          epoch_max=epoch_max,
                          epoch_min=epoch_min,
                          patience=patience,
                          sprs=sprs, 
                          sprs_rate=sprs_rate,
                          sprs_method=sprs_method)

        comment = '_'.join([dataset_name,
                            graphmodel,
                            mode,
                            str(sprs),
                            str(sprs_rate),
                            sprs_method,])
        sampling_start = time.time()
        (train_loader,
         self_val_loader,
         ppr_val_loader,
         batch_val_loader,
         self_test_loader,
         ppr_test_loader,
         batch_test_loader) = get_loaders(
            graph,
            (train_indices, val_indices, test_indices),
            batch_size,
            mode,
            batch_order,
            ppr_params,
            batch_params,
            rw_sampling_params,
            shadow_ppr_params,
            rand_ppr_params,
            ladies_params,
            n_sampling_params,
            inference,
            LBMB_val)
        sampling_end = time.time()
        logging.info(f"Sampling time: {sampling_end - sampling_start} seconds")

        stamp = ''.join(str(time.time()).split('.')) + str(seed)

        logging.info(f'model info: {comment}/model_{stamp}.pt')
        model = get_model(graphmodel,
                          graph.num_node_features,
                          graph.y.max().item() + 1,
                          hidden_channels,
                          num_layers,
                          heads,
                          device)
       
        if ('arxiv' in dataset_name or 'products' in dataset_name):
            adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
            adj = BaseLoader.normalize_adjmat(adj, normalization='sym')
            graph.edge_index = adj
        logging.info(f" model model : {graphmodel}")
        #check if model is a GAT model 
        if ('gat' in graphmodel):
            graph.edge_index = torch.stack([graph.edge_index.coo()[0], graph.edge_index.coo()[1]], dim=0)
        tmp = trainer.train(graph,
                      val_indices,
                      test_indices, 
                      dataset_name,
                      mode,
                      train_loader,
                      self_val_loader,
                      ppr_val_loader,
                      batch_val_loader,
                      model=model,
                      lr=lr,
                      reg=reg,
                      comment=comment,
                      run_no=stamp)
        num_train_batches = batch_params['num_batches'][0]
        del train_loader
        n = gc.collect()
        #sync the device 
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.max_memory_allocated()
        if inference:
            model_dir = osp.join('./saved_models', comment)
            assert osp.isdir(model_dir)
            model_path = osp.join(model_dir, f'model.pt')
            model_path = model_path.replace('.pt', f'_{num_train_batches}_{sprs}_{sprs_rate}.pt')
            model.load_state_dict(torch.load(model_path))
            model.eval()
            #print memory stat
            trainer.inference(self_val_loader,
                              ppr_val_loader,
                              batch_val_loader,
                              self_test_loader,
                              ppr_test_loader,
                              batch_test_loader,
                              model, 
                              dataset_name)

            trainer.full_graph_inference(model, graph, val_indices, test_indices, dataset_name)

        logging.info(f"{torch.cuda.memory_summary()}")
        runtime_train_lst = []
        runtime_self_val_lst = []
        runtime_part_val_lst = []
        runtime_ppr_val_lst = []
        for curves in trainer.database['training_curves']:
            runtime_train_lst += curves['per_train_time']
            runtime_self_val_lst += curves['per_self_val_time']
            runtime_part_val_lst += curves['per_part_val_time']
            runtime_ppr_val_lst += curves['per_ppr_val_time']
        results = {
            'runtime_train_perEpoch': sum(runtime_train_lst) / len(runtime_train_lst),
            'runtime_selfval_perEpoch': sum(runtime_self_val_lst) / len(runtime_self_val_lst),
            'runtime_partval_perEpoch': sum(runtime_part_val_lst) / len(runtime_part_val_lst),
            'runtime_pprval_perEpoch': sum(runtime_ppr_val_lst) / len(runtime_ppr_val_lst),
            'gpu_memory': gpu_memory,
            'max_memory': 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'curves': trainer.database['training_curves'],
            # ...
        }

        for key, item in trainer.database.items():
            if key != 'training_curves':
                results[f'{key}_record'] = item
                item = np.array(item)
                results[f'{key}_stats'] = (item.mean(), item.std(),) if len(item) else (0., 0.,)

        record_values = trainer.database['training_curves'][-1]
        #store final results dict in a csv file
        pd.DataFrame(record_values).to_csv(f'./results/{comment}_{batch_params["num_batches"][0]}_record.csv')
        logging.info(f'runtime_self_val: {sum(runtime_self_val_lst) / len(runtime_self_val_lst)}')
    except ZeroDivisionError:
        print("Error: Division by zero encountered. Terminating the program.")
        os._exit(1)
        raise SystemExit(1)
        exit()
    except:
        traceback.print_exc()
        os._exit(1)
        raise SystemExit(1)
        exit()
    else:
        logging.info("Finished successfully!")
        os._exit(1)
        raise SystemExit(1)
        exit()
