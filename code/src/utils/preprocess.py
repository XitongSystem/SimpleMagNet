#externel
import torch
import csv, os
import numpy as np
import pickle as pk
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import WebKB, WikipediaNetwork

#internel
from utils.hermitian import *

def geometric_dataset_sparse(q, K, root='../dataset/data/tmp/', subset='Cornell', dataset=WebKB, 
                        load_only = False, save_pk = True, laplacian = True, gcn_appr = False):
    if subset == '':
        dataset = dataset(root=root)
    else:
        dataset = dataset(root=root, name=subset)

    size = dataset[0].y.size(-1)
    #adj = torch.zeros(size, size).data.numpy().astype('uint8')
    #adj[dataset[0].edge_index[0], dataset[0].edge_index[1]] = 1

    f_node, e_node = dataset[0].edge_index[0], dataset[0].edge_index[1]

    label = dataset[0].y.data.numpy().astype('int')
    X = dataset[0].x.data.numpy().astype('float32')
    train_mask = dataset[0].train_mask.data.numpy().astype('bool_')
    val_mask = dataset[0].val_mask.data.numpy().astype('bool_')
    test_mask = dataset[0].test_mask.data.numpy().astype('bool_')

    if load_only:
        return X, label, train_mask, val_mask, test_mask
    
    try:
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, 
            max_eigen = 2.0, gcn_appr = gcn_appr, edge_weight = dataset[0].edge_weight)
    except AttributeError:
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, 
            max_eigen = 2.0, gcn_appr = gcn_appr, edge_weight = None)

    multi_order_laplacian = cheb_poly_sparse(L, K)
    
    save_name = root+'/data'+str(q)+'_'+str(K)
    if laplacian == False:
        save_name += '_P'
    if save_pk:
        data = {}
        data['L'] = multi_order_laplacian
        pk.dump(data, open(save_name+'_sparse.pk', 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return X, label, train_mask, val_mask, test_mask, multi_order_laplacian