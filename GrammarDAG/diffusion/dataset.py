import yaml
import json
import pickle
import pdb
import sys
from copy import deepcopy
import numpy as np
import grinpy as gp
import networkx as nx
import argparse
import torch
sys.path.append("..")

from private import *
from grammar_generator import MetaGrammar
from GrammarTree import GrammarTree, GrammarTreeNode, GrammarDAG
import DAG
from torch_geometric.utils.convert import from_networkx


def ChromaticNum(G):
    # get Chromatic number of a graph using grinpy
    return gp.chromatic_number(G)

def Density(G):
    return nx.density(G)

def GraphFeature(g, embedding):
    g_hash = nx.weisfeiler_lehman_graph_hash(g)
    feature_list = []
    for c in g_hash:
        feature_list.append(embedding(torch.tensor(int(c, 16))).squeeze())
    return torch.cat(feature_list, dim=0)

def GraphFeature_np(g):
    g_hash = nx.weisfeiler_lehman_graph_hash(g)
    feature_list = []
    for c in g_hash:
        feature_list.append(int(c, 16))
    return np.array(feature_list)

def get_data():
    path = "../dataset/grammar_2/grammar_DAG.pickle"
    # path = "../test_garmmar_DAG.pickle"
    dag = pickle.load(open(path, "rb"))
    embedding = torch.nn.Embedding(16, 3)
    G = nx.Graph()
    queue = [dag.root]
    node_list = []
    edge_list = []
    root_id = -1
    while len(queue) != 0:
        node = queue.pop(0)
        node_y = Density(node.hg.hg) if len(node.children) == 0 else 0
        # node_x = GraphFeature(node.hg.hg, embedding).detach().numpy()
        if len(node.children) != 0:
            node_x = np.zeros(32, dtype=np.int32)
        else:
            node_x = GraphFeature_np(node.hg.hg)
        # node_x = GraphFeature_np(node.hg.hg)
        node_list.append((node.node_id, {'y': node_y, 'x': node_x}))
        if node.parent[0] is not None: # skip root
            for p_n in node.parent:
                edge_list.append((p_n.node_id, node.node_id, {"label": node.rule_id}))
        else:
            assert root_id == -1 # only assign once
            root_id = node.node_id
        for child in node.children:
            if child not in queue:
                queue.append(child)
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    print(G.number_of_nodes())
    print(dag.num_nodes)
    node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            print(i, feat_dict, node_attrs)
            import pdb; pdb.set_trace()
    data = from_networkx(G)
    data.y = torch.unsqueeze(data.y, -1)
    # data.x = data.y # Changed
    data.root_id = root_id
    import pdb; pdb.set_trace()
    return data

class DAGDataset():
    def __init__(self, ratio = 0.2):
        self.data = get_data()
        self.data.num_features = 1
        self.data.num_nodes = self.data.x.size(0)
        self.ratio = ratio
        self.split_train_test()
    
    def split_train_test(self):
        leaf_mask = self.data.y != 0
        np.random.seed(2022)
        randn = torch.unsqueeze(torch.tensor(np.random.uniform(0, 1, self.data.num_nodes)), -1)
        train_mask = (randn * leaf_mask) >  self.ratio
        test_mask = (~train_mask) & leaf_mask
        self.data.train_mask = train_mask
        self.data.test_mask = test_mask


if __name__ == "__main__":
    get_data()
    