from copy import deepcopy
import numpy as np
import torch
import argparse
import graphviz
from networkx.algorithms.isomorphism import GraphMatcher
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.convert import from_networkx
import torch
from copy import deepcopy
from rdkit import Chem
import pickle
from functools import partial
from chemprop.data import MoleculeDatapoint, MoleculeDataset
import random
from private import *
from private.utils import _node_match_prod_rule_for_DAG, _node_match_prod_rule, _edge_match, _node_match_prod_rule_for_MetaDAG

class GrammarNode():
    def __init__(self, hypergraph, node_id):
        self.hg = hypergraph
        self.node_id = node_id
        self.parent_node_id = None
        
    def __eq__(self, other):
        return self.hg == other.hg
    
    def set_parent_node_id(self, parent_node_id):
        if self.parent_node_id is None:
            self.parent_node_id = [parent_node_id]
        else:
            self.parent_node_id.append(parent_node_id)

    def is_applicable(self, rule):
        gm = GraphMatcher(self.hg.hg, rule.rhs.hg, 
                partial(_node_match_prod_rule_for_DAG),
                partial(_edge_match,
                    ignore_order=True)) # if a subgraph of G1 is isomorphic to G2.
        is_starting = rule.is_start_rule
        if not is_starting:
            expanded_hgs = []
            all_mappings = [mapping for mapping in gm.subgraph_isomorphisms_iter()]
            for i, mapping in enumerate(all_mappings):
                expanded_n_hg = deepcopy(self.hg)
                # mapping: {self.hg.nodes/edges} -> {rule.rhs.nodes/edges}
                # for nodes (bonds), keep the external ones (ext_id), remove the else
                # remove all the edges, create a new NT edge (be careful with the name) and connect it to the external nodes
                ext_nodes = []
                for (hg_v, rhs_v) in mapping.items():
                    if hg_v.startswith('bond'): # a node(bond)
                        if 'ext_id' in rule.rhs.node_attr(rhs_v): # external node
                            ext_nodes.append(hg_v) # using the name string as the key
                        else:
                            expanded_n_hg.remove_node(hg_v, remove_connected_edges=False)
                    elif hg_v.startswith('e'): # an edge (atom)
                        expanded_n_hg.remove_edge(hg_v)
                    else:
                        raise('Error: invalid node name')
                assert len(rule.lhs.edges) == 1, 'Error: the number of edges of lhs should be 1'
                # import pdb; pdb.set_trace() # check ext_nodes are added correctly
                # create a new edge, the name is automatically generated inside the hypergraph class
                # its attribute is the same as the lhs of the rule
                expanded_n_hg.add_edge(ext_nodes, attr_dict=rule.lhs.edge_attr(list(rule.lhs.edges)[0]))
                # expanded_n_hg.draw_rule(file_path='expand_hg/expanded_hg_{}'.format(i))
                if expanded_n_hg not in expanded_hgs: # avoid duplicated hg, because of the symmetry
                    expanded_hgs.append(expanded_n_hg)
            # import pdb; pdb.set_trace() # check ext_nodes are added correctly
            return expanded_hgs, is_starting
        else: # if it is a starting rule, should use the whole graph for isomorphism
            if gm.subgraph_is_isomorphic():
                return [Hypergraph()], is_starting
            else:
                return [], is_starting

class GrammarDAG():
    def __init__(self, input_graphs_dict):
        self.input_graphs = {} # get the node id in the DAG of each input graph by querying its mol_key
        self.root_node_id = -1
        self.rules = []
        self.nodes = []
        self.node_id = []
        self.num_nodes = 0
        self.edges = []
        for i, (key, value) in enumerate(input_graphs_dict.items()):
            self.input_graphs[key] = i
            value.hypergraph.add_order_attr() # add the order attribute to the nodes
            self.nodes.append(GrammarNode(value.hypergraph, node_id=i))
            self.node_id.append(i)
            self.num_nodes += 1
    
    def construct_DAG(self, grammar):
        current_nodes = deepcopy(self.nodes[:2])
        # BFS
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            print("dealing with node: ", node.node_id)
            for i, rule in enumerate(grammar.prod_rule_list):
                print("rule id: ", i)
                rule.rhs.add_order_attr() # add the order attribute to the nodes
                rule.lhs.add_order_attr() # add the order attribute to the nodes
                expanded_hgs, is_starting = node.is_applicable(rule)
                if not is_starting:
                    for hg in expanded_hgs:
                        new_node = GrammarNode(hg, node_id=self.num_nodes)
                        # check if the new node is already in the DAG
                        if new_node not in self.nodes:
                            self.nodes.append(deepcopy(new_node))
                            self.node_id.append(self.num_nodes)
                            self.num_nodes += 1
                            self.edges.append((new_node.node_id, node.node_id))
                            self.nodes[node.node_id].set_parent_node_id(new_node.node_id)
                            current_nodes.append(new_node)
                        else:
                            find_matched_node = self.nodes.index(new_node)
                            matched_node_id = self.node_id[find_matched_node]
                            assert matched_node_id == self.nodes[find_matched_node].node_id, 'Error: node id not match'
                            self.edges.append((matched_node_id, node.node_id))
                            self.nodes[node.node_id].set_parent_node_id(matched_node_id)
                else:
                    if len(expanded_hgs) != 0:
                        self.edges.append((self.root_node_id, node.node_id))
                        self.nodes[node.node_id].set_parent_node_id(self.root_node_id)
                print(len(current_nodes))
        
        for node in self.nodes:
            if node.parent_node_id is None:
                self.edges.append((self.root_node_id, node.node_id))
                node.set_parent_node_id(self.root_node_id)
        import pdb; pdb.set_trace()
    
    def draw(self, filename, file_path=None, post_process_ext_id=True):
        G = graphviz.Digraph()
        G.node_attr["image"] = "none"
        # queue = [self.root]
        # while len(queue) != 0:
        for edge in self.edges:
            # node = queue.pop(0)
            for node_id in edge: # two nodes
                if node_id != -1:
                    node = self.nodes[node_id]
                    img_path = '{}/'.format(file_path) # change path
                    img_name = 'graph_{0}/viz'.format(node.node_id)
                    # post processing the ext_id
                    if post_process_ext_id:
                        for each_node in node.hg.nodes:
                            if 'ext_id' in node.hg.node_attr(each_node):
                                adjacent_edges = node.hg.adj_edges(each_node)
                                if len(adjacent_edges) > 1:
                                    assert len(adjacent_edges) == 2
                                    del node.hg.node_attr(each_node)['ext_id']
                        ext_id = 0
                        for each_node in node.hg.nodes:
                            if 'ext_id' in node.hg.node_attr(each_node):
                                node.hg.node_attr(each_node)['ext_id'] = ext_id
                                ext_id += 1
                    node.hg.draw_rule(file_path=img_path+img_name)
                    G.node("{}".format(node.node_id), label='{}'.format(node.node_id), image=img_name+'.png')
                else:
                    G.node("{}".format(node_id), label='{}'.format(node_id)) # root node
            G.edge('{}'.format(edge[0]), '{}'.format(edge[1])) #, label='{}'.format(node.rule_id))

        G.render(file_path+'/'+ filename, cleanup=True)
        with open(file_path+'/'+ filename + '.dot', 'w') as f:
            f.write(G.source)

class MetaDAGNode():
    def __init__(self, hg):
        self.hg = hg
    
    def __eq__(self, another):
        if len(self.hg.nodes) == 0 and len(another.hg.nodes) == 0:
            return True
        gm = GraphMatcher(self.hg.hg, 
                another.hg.hg,
                partial(_node_match_prod_rule_for_MetaDAG),
                partial(_edge_match, ignore_order=True))
        # import pdb; pdb.set_trace()
        return gm.is_isomorphic()

class MetaDAG():
    def __init__(self, dag_pkl_path, opt):
        with open(dag_pkl_path, 'rb') as outp:
            self.meta_DAG = pickle.load(outp)
            if opt['extended_DAG']:
                new_node = deepcopy(self.meta_DAG.nodes[6])
                new_node.node_id = max(self.meta_DAG.node_id) + 1
                center_attr = new_node.hg.edge_attr('e6')
                new_node.hg.remove_edge('e6')
                new_node.hg.add_node('bond_4', attr_dict=new_node.hg.node_attr('bond_3'))
                new_node.hg.add_edge(['bond_0', 'bond_3', 'bond_2', 'bond_1', 'bond_4'], attr_dict=center_attr)
                new_node.hg.add_edge(['bond_4'], attr_dict=new_node.hg.edge_attr('e1'))
                self.meta_DAG.nodes.append(new_node)
                self.meta_DAG.edges.append((6, len(self.meta_DAG.nodes)-1))
            print("DAG nodes: ", len(self.meta_DAG.nodes))
            print("DAG edges: ", len(self.meta_DAG.edges))
            # import pdb; pdb.set_trace()
            # if opt['DAG_edit']:
            #     nx_node_list = []
            #     for i, node in enumerate(self.meta_DAG.nodes):
            #         G = nx.Graph()
            #         G_edge_list = []
            #         for bond in list(node.hg.nodes):
            #             edges = node.hg.nodes_in_edge(bond)
            #             assert len(edges) == 2
            #             G_edge_list.append(set(edges))
            #         G.add_edges_from(G_edge_list)
            #         nx_node_list.append(G)
            #     from myTED import cal_TED
            #     import pdb; pdb.set_trace()
            #     gt_edge_list = []
            #     for i in range(len(nx_node_list)):
            #         for j in range(i + 1, len(nx_node_list)):
            #             # dist = cal_TED(nx_node_list[i], nx_node_list[j])
            #             # I dont know why but this will work
            #             G1 = nx.Graph()
            #             G2 = nx.Graph()
            #             G1.add_edges_from(list(nx_node_list[i].edges))
            #             G2.add_edges_from(list(nx_node_list[j].edges))
            #             dist_G = cal_TED(G1, G2)
            #             dist_nx = cal_TED(nx_node_list[i], nx_node_list[j])
            #             print(dist_G, dist_nx)
            #             # assert(dist_G == dist_nx)
            #             #{(58, 120), (57, 97), (72, 140), (32, 55), (20, 35), (57, 118), (50, 82), (70, 143), 
            #             # (35, 60), (42, 87), (50, 103), (68, 145), (36, 61), (72, 145), (36, 64), (46, 99), 
            #             # (32, 54), (26, 47), (67, 143), (26, 50), (41, 75), (57, 123), (68, 129), (68, 123),
            #             # (35, 62), (64, 110), (64, 107), (41, 81), (62, 107), (68, 126), (47, 100), (71, 134), 
            #             # (-1, 0), (71, 137), (72, 138), (36, 63), (56, 124), (64, 137), (70, 132), (61, 123), 
            #             # (48, 101), (61, 138), (50, 83), (65, 132), (35, 70), (35, 67), (64, 109), (42, 82), (61, 101)}
            #             if dist_nx == 1 or dist_G == 1:
            #                 # print(i, j)
            #                 gt_edge_list.append((i, j))
            #     import pdb; pdb.set_trace()


        self.data, self.dag_hg_nodes = self.get_torch_dag()
        self.feat_arch = opt['feat_arch']
        edge_index, _ = add_remaining_self_loops(self.data.edge_index, None) # check the edge_weights, is None
        self.meta_edge_index = edge_index
        self.reset_JT()
        self.train_edge_split_id = self.data.edge_index.size(1)
        self.meta_node_split_id = self.data.x_meta.size(0)
        self.model_arch = opt['feat_arch']
        # self.gnndata = torch.load('/home/mhg/pretrain-gnns/chem/gnn_data.pt')
    
    def GraphFeature_np(self, g):
        g_hash = nx.weisfeiler_lehman_graph_hash(g)
        feature_list = []
        for c in g_hash:
            feature_list.append((int(c, 16) - 7.5) / 7.5)
            # feature_list.append(int(c, 16))
        return np.array(feature_list, dtype=np.float32)
        # return np.array(feature_list, dtype=np.int)
    
    def reset_JT(self):
        self.data.y = None
        self.data.x = None
        self.data.conditions = []
        self.data.data_for_gnn_idx = []
        self.data_for_gnn = None
        self.num_mol_graphs = 0
        self.data.edge_index = self.meta_edge_index
    
    def get_torch_dag(self):
        G = nx.Graph()
        root = deepcopy(self.meta_DAG.nodes[0]).hg
        assert len(root.edges) == 2
        assert len(root.nodes) == 1
        root.remove_edge(list(root.edges)[1])
        root.remove_node(list(root.nodes)[0], remove_connected_edges=False)
        root.add_order_attr()
        node_list = [(0, {'x': self.GraphFeature_np(root.hg)})]
        dag_hg_nodes = [MetaDAGNode(root)]
        for i, node in enumerate(self.meta_DAG.nodes):
            node_x = self.GraphFeature_np(node.hg.hg)
            assert i == node.node_id
            node_list.append((node.node_id + 1, {'x': node_x}))
            node.hg.add_order_attr()
            dag_hg_nodes.append(MetaDAGNode(node.hg))
        edge_list = []
        count_root_edge = 0
        for edge in self.meta_DAG.edges:
            if -1 in edge:
                assert count_root_edge == 0, "Error: multiple root edges"
                assert 0 in edge
                edge_list.append((0, 1))
                count_root_edge += 1
            else:
                edge_list.append((edge[0] + 1, edge[1] + 1))
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                print(i, feat_dict, node_attrs)
                import pdb; pdb.set_trace()
        data = from_networkx(G)
        data.y = None
        data.x_meta = data.x
        data.x = None
        data.conditions = []
        data.data_for_gnn_idx = []
        return data, dag_hg_nodes

    def add_JT_node_with_conditions(self, input_graph, with_condition=True):
        assert(len(input_graph.prop_value) == 2)

        nx_JT = input_graph.get_JT_graph()
        newly_added_node_id = self.meta_node_split_id + (0 if self.data.x is None else self.data.x.size(0))
        try:
            connected_meta_node_id = self.dag_hg_nodes.index(MetaDAGNode(nx_JT))
        except:
            connected_meta_node_id = len(self.dag_hg_nodes) - 1
            # import pdb; pdb.set_trace()
            # pass
        if self.data.x is None:
            # self.data.x = torch.unsqueeze(torch.tensor(input_graph.mol_feature), dim=0)
            self.data.x = torch.unsqueeze(torch.tensor(input_graph.GNN_mol_feature), dim=0)
        else:
            # self.data.x = torch.cat((self.data.x, torch.unsqueeze(torch.tensor(input_graph.mol_feature), dim=0)), dim=0)
            self.data.x = torch.cat((self.data.x, torch.unsqueeze(torch.tensor(input_graph.GNN_mol_feature), dim=0)), dim=0)
        self.data.edge_index = torch.cat((self.data.edge_index, 
                                          torch.unsqueeze(torch.tensor([newly_added_node_id, connected_meta_node_id]), dim=-1), 
                                          torch.unsqueeze(torch.tensor([connected_meta_node_id, newly_added_node_id]), dim=-1),
                                          torch.unsqueeze(torch.tensor([newly_added_node_id, newly_added_node_id]), dim=-1), ), dim=1)
        if self.feat_arch == 'GNN':
            if self.data_for_gnn is None:
                self.data_for_gnn = input_graph.data_for_gnn
                self.data_for_gnn['batch'] = torch.zeros((self.data_for_gnn.x.size(0)), dtype=torch.long)
            else:
                self.data_for_gnn['batch'] = torch.cat((self.data_for_gnn['batch'], torch.ones((input_graph.data_for_gnn.x.size(0)), dtype=torch.long) * self.num_mol_graphs), dim=0)
                self.data_for_gnn['edge_index'] = torch.cat((self.data_for_gnn['edge_index'], self.data_for_gnn['x'].size(0) + input_graph.data_for_gnn['edge_index']), dim=1)
                self.data_for_gnn['x'] = torch.cat((self.data_for_gnn['x'], input_graph.data_for_gnn['x']), dim=0)
                self.data_for_gnn['edge_attr'] = torch.cat((self.data_for_gnn['edge_attr'], input_graph.data_for_gnn['edge_attr']), dim=0)
                
            data_for_gnn_id = self.data_for_gnn['batch'].max().item()
            
        else:
            data_point = MoleculeDatapoint(smiles=[input_graph.smiles], targets=[input_graph.prop_value], row=None, data_weight=1.,
                                            features_generator=None, features=None, phase_features=None, atom_features=None, atom_descriptors=None,
                                            bond_features=None, overwrite_default_atom_features=False, overwrite_default_bond_features=False, )
            if self.data_for_gnn is None:
                self.data_for_gnn = [data_point]
            else:
                self.data_for_gnn.append(data_point)
        
            data_for_gnn_id = len(self.data_for_gnn) - 1
        
        for pv in [input_graph.prop_value]:
            value, conditions = pv
            if with_condition:
                assert len(conditions) > 0
                
            if self.data.y is None:
                try:
                    assert(isinstance(self.data.conditions, list) and len(self.data.conditions) == 0)
                    assert(isinstance(self.data.data_for_gnn_idx, list) and len(self.data.data_for_gnn_idx) == 0)
                except:
                    import pdb; pdb.set_trace()
                    
                self.data.y = torch.unsqueeze(torch.tensor([value]), dim=-1).double()
                
                if with_condition:
                    self.data.conditions = torch.unsqueeze(torch.tensor(conditions), dim=0).double()
                    
                self.data.data_for_gnn_idx = torch.unsqueeze(torch.tensor([data_for_gnn_id]), dim=-1).int()
            else:
                try:
                    if with_condition:
                        assert(self.data.conditions.size(0) == self.data.y.size(0))
                    assert(self.data.data_for_gnn_idx.size(0) == self.data.y.size(0))
                except:
                    import pdb; pdb.set_trace()
                    
                self.data.y = torch.cat((self.data.y, torch.unsqueeze(torch.tensor([value]), dim=-1).double()), dim=0)
                
                if with_condition:
                    self.data.conditions = torch.cat((self.data.conditions, torch.unsqueeze(torch.tensor(conditions), dim=0).double()), dim=0)
                    
                self.data.data_for_gnn_idx = torch.cat((self.data.data_for_gnn_idx, torch.unsqueeze(torch.tensor([data_for_gnn_id]), dim=-1).int()), dim=0)

        self.num_mol_graphs += 1


class DAGDataset():
    def __init__(self, input_graphs_dict_init, testdata_idx_file, num_features, num_mols, ratio=0.2, classification=False, seed=42):
        self.num_features = num_features
        self.num_leaves = num_mols
        self.data = None
        self.ratio = ratio
        self.seed = seed
        self.classification = classification
        
        # split_train_test
        if testdata_idx_file is None:
            random.seed(self.seed)
            all_idx = list(range(num_mols))
            random.shuffle(all_idx)
            test_mask = all_idx[int((1 - self.ratio) * num_mols):]
        else:
            with open(testdata_idx_file, 'r') as fr:
                test_mask = [int(l.strip()) for l in fr.readlines()]
            
        train_mask = [i for i in range(num_mols) if i not in test_mask]

        self.train_mask = torch.tensor(train_mask, dtype=torch.long)
        self.test_mask = torch.tensor(test_mask, dtype=torch.long)

    def set_data(self, meta_DAG):
        self.data = meta_DAG.data
        self.data.num_features = self.num_features
        self.data.num_nodes = self.data.x.size(0) + self.data.x_meta.size(0)
        self.data_y_mean = self.data.y.mean(dim=0)
        self.data_y_std = self.data.y.std(dim=0)
        print("data_y_std", self.data_y_std)
        
        if not self.classification:
            self.data.y = (self.data.y - self.data_y_mean) / self.data_y_std
            
        self.data.data_for_gnn = meta_DAG.data_for_gnn
        self.train_edge_split_id = meta_DAG.train_edge_split_id
        self.meta_node_split_id = meta_DAG.meta_node_split_id
        self.num_leaves = self.data.num_nodes - self.meta_node_split_id
        
        assert self.meta_node_split_id == self.data.x_meta.size(0)
        
        self.data.train_mask = self.train_mask
        self.data.test_mask = self.test_mask

