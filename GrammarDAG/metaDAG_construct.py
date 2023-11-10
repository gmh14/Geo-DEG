import pickle
from time import time
# from mpi4py import MPI
import multiprocessing as mp
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import torch
import argparse
import graphviz
from networkx.algorithms.isomorphism import GraphMatcher, tree_isomorphism
from networkx import weisfeiler_lehman_graph_hash
from rdkit import Chem
from functools import partial
from itertools import combinations, product
from more_itertools import partitions
import random
import os

import sys
sys.path.append("../")

from private import *
from private.utils import _node_match_prod_rule_for_DAG, _node_match_prod_rule, _edge_match


class MetaGrammar():
    def __init__(self, max_lhs_deg, subgraph_structures=None, num_t_types=1, num_nt_types=1, constraint=None):
        self.max_lhs_deg = max_lhs_deg
        self.subgraph_structures = subgraph_structures or [
            "terminal", "append", "ring"]
        self.num_t_types = num_t_types
        self.num_nt_types = num_nt_types
        self.constraint = constraint
        if self.constraint is None:
            self.constraint = lambda _: True

        self.meta_grammar = ProductionRuleCorpus()
        self.build_meta_grammar()

    def _add_ext_node(self, hg, ext_nodes):
        """ mark nodes to be external (ordered ids are assigned)

        Parameters
        ----------
        hg : UndirectedHypergraph
        ext_nodes : list of str
            list of external nodes

        Returns
        -------
        hg : Hypergraph
            nodes in `ext_nodes` are marked to be external
        """
        ext_id = 0
        ext_id_exists = []
        for each_node in ext_nodes:
            ext_id_exists.append('ext_id' in hg.node_attr(each_node))
        if ext_id_exists and any(ext_id_exists) != all(ext_id_exists):
            raise ValueError
        if not all(ext_id_exists):
            for each_node in ext_nodes:
                hg.node_attr(each_node)['ext_id'] = ext_id
                ext_id += 1
        return hg

    def build_hs(self, nodes_list, bond_symbol_list, edge_nodes_list, ext_node_list, num_edge, NT):
        assert len(edge_nodes_list) == num_edge
        # We do not need bond_symbol_list for now, but may use it later

        hs_list = []
        num_types = self.num_nt_types if NT else self.num_t_types

        for symbol_combination in product(range(num_types), repeat=num_edge):
            hs = Hypergraph()
            new_node_list = []
            for i, (_, bond_symb) in enumerate(zip(nodes_list, bond_symbol_list)):
                node = 'bond_{}'.format(i)
                new_node_list.append(node)
                hs.add_node(node, attr_dict={'symbol': bond_symb})
            for i in range(num_edge):
                edge_nodes = [new_node_list[id] for id in edge_nodes_list[i]]
                edge_bonds = [bond_symbol_list[id]
                              for id in edge_nodes_list[i]]
                if NT:
                    terminal = False
                    symbol = NTSymbol(degree=len(edge_nodes),
                                      bond_symbol_list=edge_bonds,
                                      symbol="NT{}".format(symbol_combination[i]), id_=symbol_combination[i])
                else:
                    terminal = True
                    symbol = TSymbol(degree=len(edge_nodes),
                                     symbol="T{}".format(symbol_combination[i]), id_=symbol_combination[i])
                hs.add_edge(
                    edge_nodes,
                    attr_dict=dict(terminal=terminal, symbol=symbol, type_id=symbol_combination[i]))

            hs = self._add_ext_node(hs, [new_node_list[k]
                                    for k in ext_node_list])

            hs_list.append(hs)

        return hs_list

    def get_rules(self, lhs_deg, rhs_type, lhs_sym=None):  # lhs_sym can be modified later
        assert rhs_type in ["terminal", "append", "ring"]
        assert lhs_deg <= 3
        if lhs_deg == 0:
            lhs_list = [Hypergraph()]
            nodes = ["init"]  # TODO check the node names
            bond_symbols = [BondSymbol(bond_type=1, is_aromatic=False, stereo=0)]
            edge_nodes_list = [[0], [0]]
            edge_bond_list = [[0], [0]]
            rhs_list = self.build_hs(nodes_list=nodes, bond_symbol_list=bond_symbols,
                                     edge_nodes_list=edge_nodes_list,
                                     ext_node_list=[], num_edge=2, NT=True)
            # rhs_list = [rhs]
        else:
            # lhs
            bond_symbols = []
            ext_nodes = []
            edge_nodes_list = []
            edge_bond_list = []
            for i in range(lhs_deg):
                bond_symb = BondSymbol(bond_type=1, is_aromatic=False, stereo=0)
                # ext_node = "Ext_{}".format(i) # TODO check the node names
                ext_nodes.append(i)
                bond_symbols.append(bond_symb)
            lhs_list = self.build_hs(nodes_list=ext_nodes, bond_symbol_list=bond_symbols,
                                     edge_nodes_list=[list(range(lhs_deg))],
                                     ext_node_list=ext_nodes, num_edge=1, NT=True)
            # rhs
            rhs_list = []
            if rhs_type == 'terminal':
                rhs_list = self.build_hs(nodes_list=ext_nodes, bond_symbol_list=bond_symbols,
                                         edge_nodes_list=[
                                             list(range(lhs_deg))],
                                         ext_node_list=ext_nodes, num_edge=1, NT=False)
                # rhs_list.append(rhs)
            elif rhs_type == 'append':
                appended_bond_symb = BondSymbol(bond_type=1, is_aromatic=False, stereo=0)
                appended_node = "appnode"  # TODO check the node names
                # combinatorial way to connect ext_nodes to these two nodes
                # first put all in one basket
                nodes_list = [appended_node] + ext_nodes
                bond_symbol_list = [appended_bond_symb] + bond_symbols
                rhs_list = self.build_hs(nodes_list=nodes_list, bond_symbol_list=bond_symbol_list,
                                         edge_nodes_list=[
                                             [lhs_deg] + list(range(len(ext_nodes))), [0]],
                                         ext_node_list=[1 + k for k in range(len(ext_nodes))], num_edge=2, NT=True)
                # rhs_list.append(rhs)
                # otherwise
                for part in partitions(list(range(lhs_deg))):
                    edge_nodes_list = [[lhs_deg], [lhs_deg]]
                    nodes_list = ext_nodes + [appended_node]
                    bond_symbol_list = bond_symbols + [appended_bond_symb]
                    if len(part) == 2:
                        edge_nodes_list[0].extend(part[0])
                        edge_nodes_list[1].extend(part[1])
                        partial_rhs_list = self.build_hs(nodes_list=nodes_list, bond_symbol_list=bond_symbol_list,
                                                         edge_nodes_list=edge_nodes_list,
                                                         ext_node_list=list(range(len(ext_nodes))), num_edge=2, NT=True)
                        rhs_list.extend(partial_rhs_list)
            elif rhs_type == 'ring':
                appended_bond_symbs = [BondSymbol(bond_type=1, is_aromatic=False, stereo=0), BondSymbol(
                    bond_type=1, is_aromatic=False, stereo=0), BondSymbol(bond_type=1, is_aromatic=False, stereo=0)]
                # TODO check the node names
                appended_nodes = ["appnode", "appnode", "appnode"]
                nodes_list = appended_nodes + ext_nodes
                bond_symbol_list = appended_bond_symbs + bond_symbols
                edge_nodes_list = [[id1, id2]
                                   for (id1, id2) in combinations(range(3), 2)]
                for i in range(lhs_deg):
                    edge_nodes_list[i].append(3+i)
                rhs_list = self.build_hs(nodes_list=nodes_list, bond_symbol_list=bond_symbol_list,
                                         edge_nodes_list=edge_nodes_list,
                                         ext_node_list=[3 + k for k in range(len(ext_nodes))], num_edge=3, NT=True)
                # rhs_list.append(rhs)
            else:
                raise NotImplementedError
        return lhs_list, rhs_list

    def build_meta_grammar(self):
        for lhs_deg in range(self.max_lhs_deg+1):
            for rhs_type in self.subgraph_structures:
                lhs_list, rhs_list = self.get_rules(lhs_deg, rhs_type)
                for lhs in lhs_list:
                    for rhs in rhs_list:
                        rule = ProductionRule(lhs, rhs)
                        if self.constraint(rule):
                            self.meta_grammar.append(rule)

    def subgraph_checker(self, rule, subgraph_types_allowed):
        assert set(subgraph_types_allowed).issubset(
            set(self.subgraph_structures))
        grammar_allowed = MetaGrammar(
            self.max_lhs_deg, subgraph_types_allowed, self.num_t_typesself, self.num_nt_types, self.constraint)

        if rule in grammar_allowed.meta_grammar.prod_rule_list:
            return True
        return False

    def grammar_sample(self, num_rules, steps=5):
        grammar = ProductionRuleCorpus()

        for start_rule in self.meta_grammar.start_rule_list:
            grammar.append(start_rule)

        # assert len(grammar.prod_rule_list) == 1, len(grammar.prod_rule_list)
        for rule_id in range(num_rules):
            anchor_rule_idx = random.choice([i for i, rule in enumerate(
                self.meta_grammar.prod_rule_list) if not rule.is_ending])
            anchor_rule = self.meta_grammar.prod_rule_list[anchor_rule_idx]
            print("anchor_rule_idx:", anchor_rule_idx)
            lhs = anchor_rule.lhs
            rhs = anchor_rule.rhs
            curr_step = 0
            curr_iter = 0
            while True:
                if curr_step > steps or curr_iter > 10:
                    break
                curr_rule_idx = random.randint(
                    0, len(self.meta_grammar.prod_rule_list)-1)
                curr_rule = self.meta_grammar.prod_rule_list[curr_rule_idx]
                if curr_rule in self.meta_grammar.start_rule_list:
                    continue
                rhs_cand, _, avail = curr_rule.graph_rule_applied_to(rhs)
                if avail:
                    print("selected idx:", rule_id,
                          anchor_rule_idx, curr_rule_idx)
                    rhs = deepcopy(rhs_cand)
                    curr_step += 1
                curr_iter += 1
            # post processing the ext_id
            for each_node in rhs.nodes:
                if 'ext_id' in rhs.node_attr(each_node):
                    adjacent_edges = rhs.adj_edges(each_node)
                    if len(adjacent_edges) > 1:
                        assert len(adjacent_edges) == 2
                        del rhs.node_attr(each_node)['ext_id']
            ext_id = 0
            for each_node in rhs.nodes:
                if 'ext_id' in rhs.node_attr(each_node):
                    rhs.node_attr(each_node)['ext_id'] = ext_id
                    ext_id += 1
            rule = ProductionRule(deepcopy(lhs), deepcopy(rhs))
            grammar.append(rule)
        return grammar


class GrammarNode():
    def __init__(self, hypergraph, node_id):
        self.hg = hypergraph
        self.node_id = node_id
        self.parent_node_id = None

    def __eq__(self, other):
        return bool(tree_isomorphism(self.hg.hg, other.hg.hg))

    def set_parent_node_id(self, parent_node_id):
        if self.parent_node_id is None:
            self.parent_node_id = [parent_node_id]
        else:
            self.parent_node_id.append(parent_node_id)

    def is_applicable(self, rule):
        gm = GraphMatcher(self.hg.hg, rule.rhs.hg,
                          partial(_node_match_prod_rule_for_DAG),
                          partial(_edge_match,
                                  ignore_order=True))  # if a subgraph of G1 is isomorphic to G2.
        is_starting = rule.is_start_rule
        if not is_starting:
            expanded_hgs = []
            all_mappings = [
                mapping for mapping in gm.subgraph_isomorphisms_iter()]
            for i, mapping in enumerate(all_mappings):
                expanded_n_hg = deepcopy(self.hg)
                # mapping: {self.hg.nodes/edges} -> {rule.rhs.nodes/edges}
                # for nodes (bonds), keep the external ones (ext_id), remove the else
                # remove all the edges, create a new NT edge (be careful with the name) and connect it to the external nodes
                ext_nodes = []
                for (hg_v, rhs_v) in mapping.items():
                    if hg_v.startswith('bond'):  # a node(bond)
                        # external node
                        if 'ext_id' in rule.rhs.node_attr(rhs_v):
                            # using the name string as the key
                            ext_nodes.append(hg_v)
                        else:
                            expanded_n_hg.remove_node(
                                hg_v, remove_connected_edges=False)
                    elif hg_v.startswith('e'):  # an edge (atom)
                        expanded_n_hg.remove_edge(hg_v)
                    else:
                        raise ('Error: invalid node name')
                assert len(
                    rule.lhs.edges) == 1, 'Error: the number of edges of lhs should be 1'
                # import pdb; pdb.set_trace() # check ext_nodes are added correctly
                # create a new edge, the name is automatically generated inside the hypergraph class
                # its attribute is the same as the lhs of the rule
                expanded_n_hg.add_edge(
                    ext_nodes, attr_dict=rule.lhs.edge_attr(list(rule.lhs.edges)[0]))
                # expanded_n_hg.draw_rule(file_path='expand_hg/expanded_hg_{}'.format(i))
                if expanded_n_hg not in expanded_hgs:  # avoid duplicated hg, because of the symmetry
                    expanded_hgs.append(expanded_n_hg)
            # import pdb; pdb.set_trace() # check ext_nodes are added correctly
            return expanded_hgs, is_starting
        else:  # if it is a starting rule, should use the whole graph for isomorphism
            if gm.subgraph_is_isomorphic():
                return [Hypergraph()], is_starting
            else:
                return [], is_starting


class GrammarDAG():
    def __init__(self, input_graphs_dict):
        # get the node id in the DAG of each input graph by querying its mol_key
        self.input_graphs = {}
        self.root_node_id = -1
        self.rules = []
        self.nodes = []
        self.node_id = []
        self.num_nodes = 0
        self.edges = []
        for i, (key, value) in enumerate(input_graphs_dict.items()):
            self.input_graphs[key] = i
            value.hypergraph.add_order_attr()  # add the order attribute to the nodes
            self.nodes.append(GrammarNode(value.hypergraph, node_id=i))
            self.node_id.append(i)
            self.num_nodes += 1

    def construct_DAG(self, grammar, rank, max_nodes=20):
        current_nodes = []  # deepcopy(self.nodes[:2])
        # BFS
        # node_counter = 0
        added_graphs = set()  # hashed with WL test
        graph_lookup = {}
        current_depth = 1

        for rule in grammar.prod_rule_list:
            if rule.is_start_rule:
                new_node = GrammarNode(rule.graph_rule_applied_to(
                    Hypergraph())[0], node_id=self.num_nodes)
                hashed_hg = weisfeiler_lehman_graph_hash(
                    rule.graph_rule_applied_to(Hypergraph())[0].hg)
                added_graphs.add(hashed_hg)
                graph_lookup[hashed_hg] = graph_lookup.get(
                    hashed_hg, []) + [new_node]
                self.nodes.append(deepcopy(new_node))
                self.node_id.append(self.num_nodes)
                self.num_nodes += 1
                # node_counter += 1
                self.edges.append((self.root_node_id, new_node.node_id))
                self.nodes[new_node.node_id].set_parent_node_id(
                    self.root_node_id)
                current_nodes.append(deepcopy(new_node))

        # def get_adjacent_nodes(node, rank):
        #     adj = []
        #     for rule in set(grammar.prod_rule_list):
        #         # print("rule id: ", i)
        #         if rule.is_start_rule:
        #             continue
        #         hg, _, avail = rule.graph_rule_applied_to(node.hg)
        #         # print(avail, len(hg.edges))
        #         if avail and len(hg.edges) <= max_nodes:
        #             adj.append(GrammarNode(deepcopy(hg),node_id=self.num_nodes))
        #         # print("adj", len(adj))
        #     return adj

        # def get_neighbor(queue, comm, world, rank):
        #     # print(len(queue))
        #     split = np.array_split(queue, world, axis=0)
        #     # print(split)
        #     split = comm.scatter(split, root=0)
        #     # found_node = False
        #     nq = []
        #     for u in split:
        #         for v in get_adjacent_nodes(u, rank):
        #             if v not in self.nodes:
        #                 self.nodes.append(GrammarNode(deepcopy(v.hg), node_id = self.num_nodes))
        #                 self.node_id.append(self.num_nodes)
        #                 self.edges.append((u.node_id,self.num_nodes))
        #                 self.nodes[self.num_nodes].set_parent_node_id(u.node_id)
        #                 self.num_nodes += 1
        #                 nq.append(deepcopy(v))
        #     # print("nq", len(nq))
        #     data = comm.gather(nq, root=0)
        #     if rank == 0:
        #         # print(data)
        #         result = []
        #         for d in data:
        #             result.extend(d)
        #         return result

        count = 0
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)

            if len(node.hg.edges) > current_depth:
                current_depth += 1
                os.makedirs('dag_tree_iso', exist_ok=True)
                with open('dag_tree_iso/dag_hashed{}.pickle'.format(current_depth), 'wb') as f:
                    pickle.dump(self, f)
                # self.draw('test','dag{}'.format(current_depth))
                print("completed {} nodes".format(current_depth))
            # print("dealing with node: ", node.node_id)
            hgs = []
            for i, rule in enumerate(grammar.prod_rule_list):
                # print("rule id: ", i)
                # rule.rhs.add_order_attr() # add the order attribute to the nodes
                # rule.lhs.add_order_attr() # add the order attribute to the nodes
                if rule.is_start_rule:
                    continue

                # print(rule.get_all_compatible_edges(node.hg))
                # start = time()
                maps = rule.get_all_compatible_edges(node.hg)
                # print("comptible edges time: ", time() - start)
                if len(maps) == 2:
                    edges, iso_mappings = maps
                    # print(i, len(edges), [len(isomap) for isomap in iso_mappings])
                else:
                    continue

                # start = time()
                for i, edge in enumerate(edges):
                    for iso_mapping in iso_mappings[i]:
                        # print(edge,iso_mapping)
                        hg, _, avail = rule.graph_rule_applied_to(
                            node.hg, selected_edge=edge)
                        if avail and len(hg.edges) <= max_nodes:
                            hgs.append(deepcopy(hg))
                # print("apply rule time: ", time()-start)
                # for i, edge in enumerate(edges):
                # hg, _, avail = rule.graph_rule_applied_to(node.hg)
                # if avail and len(hg.edges) <= max_nodes:
                #     hgs.append(deepcopy(hg))

                # start = time()
            for hg in hgs:
                new_node = GrammarNode(hg, node_id=self.num_nodes)
                hg_hashed = weisfeiler_lehman_graph_hash(hg.hg)
                # check if the new node is already in the DAG
                # or new_node not in graph_lookup[hg_hashed]:
                if hg_hashed not in added_graphs:
                    self.nodes.append(deepcopy(new_node))
                    self.node_id.append(self.num_nodes)
                    self.num_nodes += 1
                    self.edges.append((node.node_id, new_node.node_id))
                    self.nodes[new_node.node_id].set_parent_node_id(
                        node.node_id)
                    current_nodes.append(new_node)
                    added_graphs.add(hg_hashed)
                    graph_lookup[hg_hashed] = graph_lookup.get(
                        hg_hashed, []) + [new_node]
                else:
                    try:
                        find_matched_node = graph_lookup[hg_hashed][graph_lookup[hg_hashed].index(
                            new_node)]
                        # self.node_id[find_matched_node]
                        matched_node_id = find_matched_node.node_id
                        # assert matched_node_id == self.nodes[find_matched_node].node_id, 'Error: node id not match'
                        if node.node_id not in self.nodes[matched_node_id].parent_node_id:
                            self.edges.append((node.node_id, matched_node_id))
                            self.nodes[matched_node_id].set_parent_node_id(
                                node.node_id)
                    except:
                        count += 1
                        print("{} hash errors".format(count))
                        self.nodes.append(deepcopy(new_node))
                        self.node_id.append(self.num_nodes)
                        self.num_nodes += 1
                        self.edges.append((node.node_id, new_node.node_id))
                        self.nodes[new_node.node_id].set_parent_node_id(
                            node.node_id)
                        current_nodes.append(new_node)
                        added_graphs.add(hg_hashed)
                        graph_lookup[hg_hashed] = graph_lookup.get(
                            hg_hashed, []) + [new_node]

                # print("adding node to graph time:", time()-start)

            # nq = get_neighbor(current_nodes,comm, world, rank=rank)
            # current_nodes = nq

            print(len(current_nodes), self.num_nodes)

        # comm.Abort()

    def draw(self, filename, file_path=None, post_process_ext_id=True):
        G = graphviz.Digraph()
        G.node_attr["image"] = "none"
        # queue = [self.root]
        # while len(queue) != 0:
        for edge in self.edges:
            # node = queue.pop(0)
            for node_id in edge:  # two nodes
                if node_id != -1:
                    node = self.nodes[node_id]
                    img_path = '{}/'.format(file_path)  # change path
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
                    G.node("{}".format(node.node_id), label='{}'.format(
                        node.node_id), image=img_name+'.png')
                else:
                    G.node("{}".format(node_id),
                           label='{}'.format(node_id))  # root node
            # , label='{}'.format(node.rule_id))
            G.edge('{}'.format(edge[0]), '{}'.format(edge[1]))
        G.render(file_path+'/' + filename, cleanup=True)
        with open(file_path+'/' + filename + '.dot', 'w') as f:
            f.write(G.source)


if __name__ == "__main__":
    grammar = MetaGrammar(max_lhs_deg=3, subgraph_structures=[
                          'append'], num_t_types=0, num_nt_types=1, constraint=None).meta_grammar
    dag = GrammarDAG({})
    # comm = MPI.COMM_WORLD
    # world = comm.size
    # rank = comm.Get_rank()
    dag.construct_DAG(grammar, rank=-1, max_nodes=10)
    node_sizes = {}
    for node in dag.nodes:
        node_sizes[len(node.hg.edges)] = node_sizes.get(
            len(node.hg.edges), 0) + 1
    print(node_sizes)
    # print(dag)
    # dag.draw('test','dag')
    with open('dag.pickle', 'wb') as f:
        pickle.dump(dag, f)
