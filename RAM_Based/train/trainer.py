import logging
import os
import time
from collections import defaultdict
from math import ceil
import torch_geometric.transforms as T
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_sparse import SparseTensor
import GPUtil
from dataloaders.BaseLoader import BaseLoader
from .prefetch_generators import BackgroundGenerator
from .train_utils import run_batch
from data.data_utils import MyGraph
import networkit as nk
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.transforms import ToUndirected
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.typing import SparseTensor
from pprint import pprint
import gc
def add_to_edge_list(edge_list,u, i):
   edge_list[i].append(u)
   return


def which_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device, obj.name)
        except:
            pass
class Trainer:
    def __init__(self,
                 mode: str,
                 num_batches: int,
                 micro_batch: int = 1,
                 batch_size: int = 1,
                 epoch_max: int = 2,
                 epoch_min: int = 1,
                 patience: int = 1,
                 device: str = 'cuda', 
                 sprs: int = 0,
                 sprs_rate: float = 1.0,
                 sprs_method: str = 'SCAN', 
                 ):

        super().__init__()

        self.mode = mode
        self.device = device
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.micro_batch = micro_batch
        self.epoch_max = epoch_max
        self.epoch_min = epoch_min
        self.patience = patience
        self.sprs = sprs
        self.sprs_rate = sprs_rate
        self.sprs_method = sprs_method
        self.database = defaultdict(list)

    def get_loss_scaling(self, len_loader: int):
        micro_batch = int(min(self.micro_batch, len_loader))
        loss_scaling_lst = [micro_batch] * (len_loader // micro_batch) + [len_loader % micro_batch]
        return loss_scaling_lst, micro_batch

    def train(self,
              graph,
              val_nodes, 
              test_nodes,
              dataset_name,
              mode,
              train_loader,
              self_val_loader,
              ppr_val_loader,
              batch_val_loader,
              model,
              lr,
              reg,
              comment='',
              run_no=''):

        #         writer = SummaryWriter('./runs')
        patience_count = 0
        best_accs = {'train': 0., 'self': 0., 'part': 0., 'ppr': 0.}
        best_val_acc = -1.

        if not os.path.isdir('./saved_models'):
            os.mkdir('./saved_models')
        model_dir = os.path.join('./saved_models', comment)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_path = os.path.join(model_dir, f'model.pt')
        # add num_batches sprs and sprs_rate to the model_path
        model_path = model_path.replace('.pt', f'_{self.num_batches}_{self.sprs}_{self.sprs_rate}.pt')

        # start training
        training_curve = defaultdict(list)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.33, patience=30,
                                                               cooldown=10, min_lr=1e-4)

        next_loader = BackgroundGenerator(train_loader)
        if (self.sprs%2 == 1):
            min_edges = 10000000000
            max_edges = 0
            tmp_loader = BackgroundGenerator(train_loader)
            while True:
                data = tmp_loader.next() 
                logging.info(f"Sampled data: {data}")
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    break
                logging.info(f"edge_index: {data[0].edge_index}")
                prime_nodes = data[0].output_node_mask.sum()
                logging.info(f'prime_nodes: {prime_nodes}')
                num_edge = data[0].edge_index.nnz()
                if num_edge < min_edges and num_edge > 10000:
                    min_edges = num_edge
                if num_edge > max_edges:
                    max_edges = num_edge
            logging.info(f"min_edges: {min_edges}, max_edges: {max_edges}, ratio: {min_edges/max_edges}")
        for epoch in range(self.epoch_max):
            logging.info(f"Epoch {epoch}")
            data_dic = {'self': {'loss': 0., 'acc': 0., 'num': 0},
                        'part': {'loss': 0., 'acc': 0., 'num': 0},
                        'train': {'loss': 0., 'acc': 0., 'num': 0},
                        'ppr': {'loss': 0., 'acc': 0., 'num': 0}, }

            update_count = 0

            # train
            model.train()
            loss_scaling_lst, cur_micro_batch = self.get_loss_scaling(train_loader.loader_len)
            loader, next_loader = next_loader, None

            start_time = time.time()
            if (epoch == 0):
                sp_time = 0
            e_time = 0 
            n_samples = 0
            s_time = 0
            while True:
                time1 = time.time()
                data = loader.next()
                time2 = time.time()
                s_time += time2 - time1
                n_samples += 1
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    if batch_val_loader is not None:
                        next_loader = BackgroundGenerator(batch_val_loader)
                    elif ppr_val_loader is not None:
                        next_loader = BackgroundGenerator(ppr_val_loader)
                    else:
                        next_loader = BackgroundGenerator(self_val_loader)
                    break
                if (epoch == 1):
                    logging.info(f"edge_index: {data[0].edge_index}")
                if ((self.sprs%2 == 1) and epoch == 0):
                    sp_start_time = time.time()
                    row, col, edge_attr = data[0].edge_index.coo()
                    src = row.cpu().numpy()
                    dst = col.cpu().numpy()
                    edge_attr = edge_attr.cpu().numpy()
                    #print(f"num nodes in graph: {data[0].num_nodes}")
                    G = nk.graph.Graph(n=data[0].num_nodes , weighted=True, directed=True)
                    for i in range(len(src)):
                        G.addEdge(src[i], dst[i], edge_attr[i])
                    G.indexEdges()
                    if self.sprs_method == 'SCAN':
                        scansp = nk.sparsification.SCANSparsifier()
                    elif self.sprs_method == 'LSIM':
                        scansp = nk.sparsification.LocalSimilaritySparsifier()
                    target_ratio = min_edges / data[0].edge_index.nnz() 
                    logging.info(f"target_ratio: {target_ratio}")
                    scanG = G
                    if target_ratio != 1:
                        scanG = scansp.getSparsifiedGraphOfSize(G, target_ratio)
                    # create an empty list of edges
                    src = []
                    dst = []
                    edge_list = [[], [],[]]
                    # iterate over the edges of the graph
                    scanG.forEdges(lambda u, v, w, eid: add_to_edge_list(edge_list, u, 0))
                    scanG.forEdges(lambda u, v, w, eid: add_to_edge_list(edge_list, v, 1))
                    scanG.forEdges(lambda u, v, w, eid: add_to_edge_list(edge_list, w, 2))
                     
                    tensor_edge_list = torch.tensor(([edge_list[0], edge_list[1]])).to(self.device)
                    tensor_edge_attr = torch.tensor(edge_list[2]).to(self.device)
                    tensor_edge_list = to_undirected(tensor_edge_list, tensor_edge_attr, num_nodes=data[0].num_nodes, reduce="max")

                    tensor_edge_list = SparseTensor(row = tensor_edge_list[0][0], col = tensor_edge_list[0][1], value = tensor_edge_list[1], sparse_sizes=(data[0].num_nodes, data[0].num_nodes)).to(self.device)
                    sp_end_time = time.time()
                    sp_time += sp_end_time - sp_start_time
                    
                    data[0].edge_index = tensor_edge_list
                    print(f"final data: {data[0].edge_index}")
                loss, corrects, num_nodes, _, _, t_time = run_batch(mode, model, data[0], self.sprs, loss_scaling_lst[0])
                e_time += t_time
                data_dic['train']['loss'] += loss
                data_dic['train']['acc'] += corrects
                data_dic['train']['num'] += num_nodes
                update_count += 1

                if update_count >= cur_micro_batch:
                    loss_scaling_lst.pop(0)
                    opt.step()
                    opt.zero_grad()
                    update_count = 0
            if (epoch == 1):
                #only print two decimal points
                logging.info(f"Sampling OH: {s_time:.2f}")
                logging.info(f"Number of samples: {n_samples-1}")

            if (epoch == 0):
                logging.info(f"First epoch sparsification sp_time: {sp_time}, e_time: {e_time}")
            # remainder
            if update_count:
                opt.step()
                opt.zero_grad()

            #train_time = time.time() - start_time
            train_time = e_time
            logging.info('After train loader -- '
                         f'Allocated: {torch.cuda.memory_allocated()}, '
                         f'Max allocated: {torch.cuda.max_memory_allocated()}, '
                         f'Reserved: {torch.cuda.memory_reserved()}')
            model.eval()
            
            # part val first, for fairness of all methods
            start_time = time.time()
            if batch_val_loader is not None:
                loader, next_loader = next_loader, None

                while True:
                    data = loader.next()
                    if data is None:
                        loader = None
                        loss = None
                        corrects = None
                        if ppr_val_loader is not None:
                            next_loader = BackgroundGenerator(ppr_val_loader)
                        else:
                            next_loader = BackgroundGenerator(self_val_loader)
                        break
                    with torch.no_grad():
                        loss, corrects, num_nodes, _, _, e_time = run_batch(mode, model, data[0], self.sprs)
                        data_dic['part']['loss'] += loss
                        data_dic['part']['acc'] += corrects
                        data_dic['part']['num'] += num_nodes
            part_val_time = time.time() - start_time

            # ppr val
            start_time = time.time()
            if ppr_val_loader is not None:
                loader, next_loader = next_loader, None

                while True:
                    data = loader.next()
                    if data is None:
                        loader = None
                        loss = None
                        corrects = None
                        next_loader = BackgroundGenerator(self_val_loader)
                        break

                    with torch.no_grad():
                        loss, corrects, num_nodes, _, _, e_time = run_batch(mode, model, data[0])
                        data_dic['ppr']['loss'] += loss
                        data_dic['ppr']['acc'] += corrects
                        data_dic['ppr']['num'] += num_nodes
            ppr_val_time = time.time() - start_time


            # calcualte full graph inference accuracy and time 
            val_acc, test_acc, val_f1, test_f1, test_time = self.full_graph_inference(model, graph, val_nodes, test_nodes, dataset_name)
            logging.info(f"val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, val_f1: {val_f1:.5f}, test_f1: {test_f1:.5f}, test_time: {test_time:.5f}")
            # original val
            loader, next_loader = next_loader, None
            start_time = time.time()

            while True:
                data = loader.next()
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    if epoch < self.epoch_max - 1:
                        next_loader = BackgroundGenerator(train_loader)
                    else:
                        next_loader = None
                    break
                #else:
                #    if data[1]:  # stop signal
                #        if epoch < self.epoch_max - 1:
                #            next_loader = BackgroundGenerator(train_loader)
                #        else:
                #            next_loader = None

                with torch.no_grad():
                    loss, corrects, num_nodes, _, _, e_time = run_batch(mode, model, data[0])
                    data_dic['self']['loss'] += loss
                    data_dic['self']['acc'] += corrects
                    data_dic['self']['num'] += num_nodes

            self_val_time = time.time() - start_time
            # update training info
            for cat in ['train', 'self', 'part', 'ppr']:
                if data_dic[cat]['num'] > 0:
                    data_dic[cat]['loss'] = (data_dic[cat]['loss'] / data_dic[cat]['num']).item()
                    data_dic[cat]['acc'] = (data_dic[cat]['acc'] / data_dic[cat]['num']).item()
                best_accs[cat] = max(best_accs[cat], data_dic[cat]['acc'])

            # lr scheduler
            criterion_val_loss = data_dic['part']['loss'] if data_dic['part']['loss'] != 0 else data_dic['self']['loss']
            if scheduler is not None:
                scheduler.step(criterion_val_loss)

            # early stop
            criterion_val_acc = data_dic['part']['acc'] if data_dic['part']['acc'] != 0 else data_dic['self']['acc']
            if criterion_val_acc > best_val_acc:
                best_val_acc = criterion_val_acc
                patience_count = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_count += 1
                if epoch > self.epoch_min and patience_count > self.patience:
                    scheduler = None
                    opt = None
                    assert loader is None

                    if next_loader is not None:
                        next_loader.stop_signal = True
                        while next_loader.is_alive():
                            batch = next_loader.next()
                        next_loader = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    break

            logging.info(f"train_acc: {data_dic['train']['acc']:.5f}, "
                         f"self_val_acc: {data_dic['self']['acc']:.5f}, "
                         f"part_val_acc: {data_dic['part']['acc']:.5f}, "
                         f"ppr_val_acc: {data_dic['ppr']['acc']:.5f}, "
                         f"lr: {opt.param_groups[0]['lr']}, "
                         f"patience: {patience_count} / {self.patience}\n") 

            # maintain curves
            training_curve['per_train_time'].append(train_time)
            training_curve['per_self_val_time'].append(self_val_time)
            training_curve['per_part_val_time'].append(part_val_time)
            training_curve['per_ppr_val_time'].append(ppr_val_time)
            training_curve['train_loss'].append(data_dic['train']['loss'])
            training_curve['train_acc'].append(data_dic['train']['acc'])
            training_curve['self_val_loss'].append(data_dic['self']['loss'])
            training_curve['self_val_acc'].append(data_dic['self']['acc'])
            training_curve['ppr_val_loss'].append(data_dic['ppr']['loss'])
            training_curve['ppr_val_acc'].append(data_dic['ppr']['acc'])
            training_curve['part_val_loss'].append(data_dic['part']['loss'])
            training_curve['part_val_acc'].append(data_dic['part']['acc'])
            training_curve['full_val_acc'].append(val_acc)
            training_curve['full_test_acc'].append(test_acc)
            training_curve['full_val_f1'].append(val_f1)
            training_curve['full_test_f1'].append(test_f1)
            training_curve['full_test_time'].append(test_time)
            training_curve['post_sampling_sprs_time'].append(sp_time)
            training_curve['sampling_time'].append(s_time)
            training_curve['lr'].append(opt.param_groups[0]['lr'])
            gc.collect()
            torch.cuda.empty_cache()

        #             writer.add_scalar('train_loss', data_dic['train']['loss'], epoch)
        #             writer.add_scalar('train_acc', data_dic['train']['acc'], epoch)
        #             writer.add_scalar('self_val_loss', data_dic['self']['loss'], epoch)
        #             writer.add_scalar('self_val_acc', data_dic['self']['acc'], epoch)

        # ending
        self.database['best_train_accs'].append(best_accs['train'])
        self.database['training_curves'].append(training_curve)

        logging.info(f"best train_acc: {best_accs['train']}, "
                     f"best self val_acc: {best_accs['self']}, "
                     f"best part val_acc: {best_accs['part']}"
                     f"best ppr val_acc: {best_accs['ppr']}")
        gc.collect()
        torch.cuda.empty_cache()

        GPU = GPUtil.getGPUs()[0]
        logging.info(f"GPU name: {GPU.name}")
        logging.info(f"GPU memory total: {GPU.memoryTotal/1024:.1f} GB")
        logging.info(f"GPU memory used: {GPU.memoryUsed/1024:.1f} GB")
        logging.info(f"GPU memory free: {GPU.memoryFree/1024:.1f} GB")
        logging.info(f"GPU utilization: {GPU.load*100:.1f}%")
        logging.info(f"GPU memory utilization: {GPU.memoryUtil*100:.1f}%")
        #clean up device arrays
        assert next_loader is None and loader is None
        return

    #         writer.flush()

    def train_single_batch(self,
                           dataset,
                           model,
                           lr,
                           reg,
                           val_per_epoch=5,
                           comment='',
                           run_no=''):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  self_val_loader,
                  ppr_val_loader,
                  batch_val_loader,
                  self_test_loader,
                  ppr_test_loader,
                  batch_test_loader,
                  model,
                  dataset_name,
                  record_numbatch=False):

        cat_dict = {('self', 'val',): [self.database['self_val_accs'], self.database['self_val_f1s']],
                    ('part', 'val',): [self.database['part_val_accs'], self.database['part_val_f1s']],
                    ('ppr', 'val',): [self.database['ppr_val_accs'], self.database['ppr_val_f1s']],
                    ('self', 'test',): [self.database['self_test_accs'], self.database['self_test_f1s']],
                    ('part', 'test',): [self.database['part_test_accs'], self.database['part_test_f1s']],
                    ('ppr', 'test',): [self.database['ppr_test_accs'], self.database['ppr_test_f1s']], }

        loader_dict = {'val': {'self': self_val_loader, 'part': batch_val_loader, 'ppr': ppr_val_loader},
                       'test': {'self': self_test_loader, 'part': batch_test_loader, 'ppr': ppr_test_loader}}

        time_dict = {'self': self.database['self_inference_time'],
                     'part': self.database['part_inference_time'],
                     'ppr': self.database['ppr_inference_time']}

        for cat in ['val', 'test']:
            for sample in ['self', 'part', 'ppr']:
                acc, f1 = 0., 0.
                num_batch = 0
                torch.cuda.synchronize()
                start_time = time.time()
                if loader_dict[cat][sample] is not None:
                    loader = BackgroundGenerator(loader_dict[cat][sample])

                    pred_labels = []
                    true_labels = []

                    while True:
                        data = loader.next()
                        if data is None:
                            del loader
                            break
                        mode = 'infer'
                        _, _, _, pred_label_batch, true_label_batch, e_time = run_batch(mode, model, data[0])
                        pred_labels.append(pred_label_batch.detach())
                        true_labels.append(true_label_batch.detach())
                        num_batch += 1

                    pred_labels = torch.cat(pred_labels, dim=0).cpu().numpy()
                    true_labels = torch.cat(true_labels, dim=0).cpu().numpy()

                    acc = (pred_labels == true_labels).sum() / len(true_labels)
                    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

                cat_dict[(sample, cat,)][0].append(acc)
                cat_dict[(sample, cat,)][1].append(f1)

                if record_numbatch:
                    self.database[f'numbatch_{sample}_{cat}'].append(num_batch)

                logging.info("{}_{}_acc: {:.3f}, {}_{}_f1: {:.3f}, ".format(sample, cat, acc, sample, cat, f1))
                if cat == 'test':
                    torch.cuda.synchronize()
                    time_dict[sample].append(time.time() - start_time)

    @torch.no_grad()
    def full_graph_inference(self,
                             model,
                             graph,
                             val_nodes,
                             test_nodes,
                             dataset_name):

        start_time = time.time()
        if isinstance(val_nodes, torch.Tensor):
            val_nodes = val_nodes.numpy()
        if isinstance(test_nodes, torch.Tensor):
            test_nodes = test_nodes.numpy()

        start_time = time.time()
        mask = np.union1d(val_nodes, test_nodes)
        val_mask = np.in1d(mask, val_nodes)
        test_mask = np.in1d(mask, test_nodes)
        assert np.all(np.invert(val_mask) == test_mask)
        if (1): 
            pred = model.inference(graph, int((graph.num_nodes)/64), self.device).detach().cpu().numpy()
        else:
            if ('arxiv' in dataset_name or 'products' in dataset_name):
                graph.edge_index = graph.edge_index.to(self.device)
                graph.x = graph.x.to(self.device)
                #outputs = model.chunked_pass(MyGraph(x=graph.x, adj=adj, idx=torch.from_numpy(mask)),
                #                         self.num_batches // self.batch_size).detach().cpu().numpy() 
                graph.output_node_mask = np.zeros(graph.num_nodes)
                outputs = model(graph, 1).detach().cpu().numpy()
            else:
                #create a mask for val and test nodes 
                mask_igb = np.in1d(np.arange(graph.num_nodes), mask)
                #convert edge_index to sparse tensor
                graph.edge_index = graph.edge_index.to(self.device)
                graph.x = graph.x.to(self.device)
                #print shape of mask_igb and number of ones in it
                # output node mask is one for all nodes
                graph.output_node_mask = np.zeros(graph.num_nodes)
                outputs = model(graph, 1).detach().cpu().numpy()
                outputs = outputs
            pred = outputs

        for cat in ['val', 'test']:
            nodes = val_nodes if cat == 'val' else test_nodes
            _mask = val_mask if cat == 'val' else test_mask

            _pred = np.argmax(pred[nodes], axis=1).astype(int)
            node_true = graph.y.numpy()[nodes].astype(int).squeeze()

            acc = np.equal(_pred, node_true).sum() / len(node_true)
            f1 = f1_score(node_true, _pred, average='macro', zero_division=0)

            self.database[f'full_{cat}_accs'].append(acc)
            self.database[f'full_{cat}_f1s'].append(f1)
            if (cat == 'test'): 
                full_test_acc = acc
                full_test_f1 = f1
            else:
                full_val_acc = acc
                full_val_f1 = f1

            logging.info("full_{}_acc: {:.3f}, full_{}_f1: {:.3f}, ".format(cat, acc, cat, f1))

        self.database['full_inference_time'].append(time.time() - start_time)
        end_time = time.time() 
        return full_val_acc, full_test_acc, full_val_f1, full_test_f1, end_time - start_time
