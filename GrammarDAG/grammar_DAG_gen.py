from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import torch
import argparse
import time

import sys
sys.path.append("../")

from grammar_dag import GrammarDAG
from fuseprop import find_clusters, extract_subgraph, get_mol, get_smiles, find_fragments, find_fragments_with_scaffold
from private import *
from agent import sample


# Morgan Fingerprints
def fingerprints_matrix(mols, nBits, radius=3):
    ''' Given a list (mols) of rdkit-encoded molecules,
        return an indicator matrix of nBit Morgan fingerprints,
        i.e. each row is the morgan vector representation of a molecule
    '''
    fp_matrix = np.zeros((len(mols), nBits))
    for m, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nBits, radius=3)
        fpOn = set(fp.GetOnBits())
        fp_matrix[m] = np.array([x in fpOn for x in range(nBits)])
    return np.array(fp_matrix, dtype=int)


def partial_svd(matrix, k, keep_shape = False):
    ''' Perform rank-k partial singular value decomposition on a matrix of shape (m x n)
        and return a matrix that approximates the original
            - If keep_shape, then the output matrix is rank k and has same shape (m x n)
            - If not keep_shape, then the output matrix has smaller shape (m x k)
        # (See: https://stats.stackexchange.com/questions/107533/)
    '''
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    Uk = U[:,:k]
    Sk = np.diag(s[:k])
    out = Uk @ Sk
    if keep_shape:
        Vk = Vt.T[:,:k]
        out = out @ Vk.T
    return out


def data_processing(input_smiles, prop_values, GNN_model_path, motif, with_condition=False):
    input_mols = []
    input_graphs = []
    init_subgraphs = []
    subgraphs_idx = []
    input_graphs_dict = {}
    init_edge_flag = 0

    for n, smiles in enumerate(input_smiles):
        print("data processing {}/{}".format(n, len(input_smiles)))
        prop_value = prop_values[n]
        # Kekulized
        org_smiles = smiles
        try:
            smiles = get_smiles(get_mol(smiles))
        except:
            continue
        mol = get_mol(smiles)
        input_mols.append(mol)
        
        if n == -1:
            import pdb; pdb.set_trace()
        if motif == 'edge':
            clusters, atom_cls = find_clusters(mol)
            for i,cls in enumerate(clusters):
                clusters[i] = set(list(cls))
            for a in range(len(atom_cls)):
                atom_cls[a] = set(atom_cls[a])
        elif motif == 'motif':
            fragments = find_fragments(mol)
            clusters = [frag[1] for frag in fragments]
        elif motif == 'scaffold':
            fragments, scaffold_atom = find_fragments_with_scaffold(mol, smiles)
            clusters = [frag[1] for frag in fragments]
        else:
            raise ValueError("Invalid motif type")
        if len(clusters) > 10:
            print(len(clusters))
            import pdb; pdb.set_trace()
        
        # Construct graphs
        subgraphs = []
        subgraphs_idx_i = []
        for i, cluster in enumerate(clusters):
            _, subgraph_i_mapped, _ = extract_subgraph(smiles, cluster)
            subgraphs.append(SubGraph(subgraph_i_mapped, mapping_to_input_mol=subgraph_i_mapped, subfrags=list(cluster)))
            subgraphs_idx_i.append(list(cluster))
            init_edge_flag += 1
        
        if with_condition:
            assert(type(prop_value) == list)
            assert(len(prop_value) >= 1)
            assert(len(prop_value[0]) == 3)
            suffix = "-{}".format(prop_value[0][-1])
        else:
            suffix = "-{}".format(prop_value)

        if MolKey(mol, suffix).sml in input_graphs_dict.keys():
            print("Duplicate mol", Chem.MolToSmiles(graph.mol))
            import pdb; pdb.set_trace()
            continue
        init_subgraphs.append(subgraphs)
        subgraphs_idx.append(subgraphs_idx_i)
        graph = InputGraph(mol, smiles, org_smiles, prop_value, subgraphs, subgraphs_idx_i, GNN_model_path) #, mol_feature=crow_feature[n, :])
        input_graphs.append(graph)
        print(MolKey(mol, suffix).sml)
        input_graphs_dict[MolKey(graph.mol, suffix).sml] = graph

    assert len(input_graphs) == len(input_graphs_dict.values()), '{}, {}'.format(len(input_graphs), len(input_graphs_dict.values())) # make sure there is no duplicate
    # Construct subgraph_set 
    subgraph_set = SubGraphSet(init_subgraphs, subgraphs_idx, input_graphs)
    return subgraph_set, input_graphs_dict


def grammar_generation(agent, input_graphs_dict, subgraph_set, grammar, mcmc_iter, sample_number):
    # Selected hyperedge (subgraph)
    plist = [*subgraph_set.map_to_input]

    # Terminating condition
    if len(plist) == 0:
        # done_flag, new_input_graphs_dict, new_subgraph_set, new_grammar
        return True, input_graphs_dict, subgraph_set, grammar

    # Update every InputGraph: remove every subgraph that equals to p_star, for those subgraphs that contain atom idx in p_star, replace the atom with p_star
    org_input_graphs_dict = deepcopy(input_graphs_dict)
    org_subgraph_set = deepcopy(subgraph_set)
    org_grammar = deepcopy(grammar)

    input_graphs_dict = deepcopy(org_input_graphs_dict)
    subgraph_set = deepcopy(org_subgraph_set)
    grammar = deepcopy(org_grammar)

    for i, (key, input_g) in enumerate(input_graphs_dict.items()):
        print("---for graph {}---".format(i))
        action_list = []
        all_final_features = []
        # Skip the final iteration for training agent
        if len(input_g.subgraphs) > 1:
            for subgraph, subgraph_idx in zip(input_g.subgraphs, input_g.subgraphs_idx):
                subg_feature = input_g.get_subg_feature_for_agent(subgraph)
                num_occurance = subgraph_set.map_to_input[MolKey(subgraph)][1]
                num_in_input = len(subgraph_set.map_to_input[MolKey(subgraph)][0].keys())
                final_feature = []
                final_feature.extend(subg_feature.tolist())
                final_feature.append(1 - np.exp(-num_occurance))
                final_feature.append(num_in_input / len(list(input_graphs_dict.keys())))
                all_final_features.append(torch.unsqueeze(torch.from_numpy(np.array(final_feature)).float(), 0))
            while(True):
                action_list, take_action = sample(agent, torch.vstack(all_final_features), mcmc_iter, sample_number)
                if take_action:
                    break
        elif len(input_g.subgraphs) == 1:
            action_list = [1]
        else:
            continue
        print("Hyperedge sampling:", action_list)

        # Merge connected hyperedges
        p_star_list = input_g.merge_selected_subgraphs(action_list) 

        # update the JT nodes/edges
        input_g.update_JT(p_star_list) # TODO reset JT

        start = time.time()
        # Generate rules # TODO faster way to implement this?
        add_grammar_count = 0
        for p_star in p_star_list:
            subgraphs, subgraphs_idx = input_g.is_candidate_subgraph(p_star)
            for subg, subg_idx in zip(subgraphs, subgraphs_idx):
                if subg_idx not in input_g.subgraphs_idx:
                    # Skip the subg if it has been merged in previous iterations
                    continue
                grammar = generate_rule(input_g, subg, grammar)
                input_g.update_subgraph(subg_idx)
                add_grammar_count += 1

        if len(p_star_list) != add_grammar_count:
            import pdb; pdb.set_trace()
                    
    # Update subgraph_set
    subgraph_set.update([g for (k, g) in input_graphs_dict.items()])
    new_grammar = deepcopy(grammar)
    new_input_graphs_dict = deepcopy(input_graphs_dict)
    new_subgraph_set = deepcopy(subgraph_set)

    # print("post process time: ", time.time() - start)
    return False, new_input_graphs_dict, new_subgraph_set, new_grammar


def DAG_MCMC_sampling(agent, all_input_graphs_dict, all_subgraph_set, all_grammar, sample_number):
    iter_num = 0
    start = time.time()
    
    for i, (key, input_g) in enumerate(all_input_graphs_dict.items()):
        input_g.reset_JT()
        
    while(True):
        print("======MCMC iter{}======".format(iter_num))
        done_flag, new_input_graphs_dict, new_subgraph_set, new_grammar = grammar_generation(agent, all_input_graphs_dict, all_subgraph_set, all_grammar, iter_num, sample_number)
        print("Graph contraction status: ", done_flag)
        if done_flag:
            break
        all_input_graphs_dict = deepcopy(new_input_graphs_dict)
        all_subgraph_set = deepcopy(new_subgraph_set)
        all_grammar = deepcopy(new_grammar)
        iter_num += 1
        
    print(list(new_input_graphs_dict.values())[0].JT_nodes)
    print("total MCMC time: ", time.time() - start)
    return iter_num, new_grammar, new_input_graphs_dict

if __name__ == '__main__':
    sml = '[*]Oc1c(-c2ccccc2)cc([*])cc1-c1ccccc1'
    sml = '[*]Oc1c(c2ccccc2)cc([*])cc1c1ccccc1'
    # sml = 'c1ccccc1c2ccccc2'
    mol = Chem.MolFromSmiles(sml)
    fragments = find_fragments(mol)
    clusters = [frag[1] for frag in fragments]
    print(clusters)
    import pdb; pdb.set_trace()
