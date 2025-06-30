import torch
import time
import logging
def run_batch(mode, model, graph, sprs=0, num_microbatches_minibatch=None):

    if 'saint' in mode:
        y = graph.y
    else:
        y = graph.y[graph.output_node_mask]
    num_prime_nodes = len(y)
    torch.cuda.synchronize() 
    start = time.time()
    outputs = model(graph)
    logging.info(f" shape of outputs: {outputs.shape}")
    if hasattr(graph, 'node_norm') and graph.node_norm is not None:
        loss = torch.nn.functional.nll_loss(outputs, y, reduction='none')
        if 'saint' in mode:
            loss = (loss * graph.node_norm).sum()
        else:
            loss = (loss * graph.node_norm[graph.output_node_mask]).sum()
    else:
        y = y.squeeze()
        loss = torch.nn.functional.nll_loss(outputs, y)
        
    return_loss = loss.clone().detach() * num_prime_nodes
    
    if model.training:
        loss = loss / num_microbatches_minibatch
        loss.backward()
    
    pred = torch.argmax(outputs, dim=1)
    corrects = pred.eq(y).sum().detach()
    torch.cuda.synchronize()
    end = time.time() 
    t_time = end - start
    return return_loss, corrects, num_prime_nodes, pred, y , t_time
