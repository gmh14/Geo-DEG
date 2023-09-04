from rdkit import Chem
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch.multiprocessing as mp
import logging
import torch
import math
import os
import time
import pprint
import pickle
import argparse
import fcntl
import sys
sys.path.append("../")
from private import *
from grammar_generation import *
from agent import Agent


def evaluate(grammar, opt, metrics=['diversity', 'syn']):
    # Metric evalution for the given gramamr
    div = InternalDiversity()
    eval_metrics = {}
    generated_samples = []
    generated_samples_canonical_sml = []
    iter_num_list = []
    idx = 0
    no_newly_generated_iter = 0
    print("Start grammar evaluation...")
    while(True):
        print("Generating sample {}/{}".format(idx, opt['num_generated_samples']))
        mol, iter_num = random_produce(grammar)
        if mol is None:
            no_newly_generated_iter += 1
            continue
        can_sml_mol = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        if can_sml_mol not in generated_samples_canonical_sml:
            generated_samples.append(mol)
            generated_samples_canonical_sml.append(can_sml_mol)
            iter_num_list.append(iter_num)
            idx += 1
            no_newly_generated_iter = 0
        else:
            no_newly_generated_iter += 1
        if idx >= opt['num_generated_samples'] or no_newly_generated_iter > 10:
            break

    for _metric in metrics:
        assert _metric in ['diversity', 'num_rules', 'num_samples']
        if _metric == 'diversity':
            diversity = div.get_diversity(generated_samples)
            eval_metrics[_metric] = diversity
        elif _metric == 'num_rules':
            eval_metrics[_metric] = grammar.num_prod_rule
        elif _metric == 'num_samples':
            eval_metrics[_metric] = idx
        # elif _metric == 'syn':
        #     eval_metrics[_metric] = retro_sender(generated_samples, opt)
        else:
            raise NotImplementedError
    return eval_metrics


# def retro_sender(generated_samples, opt):
#     # File communication to obtain retro-synthesis rate
#     with open(opt['receiver_file'], 'w') as fw:
#         fw.write('')
#     while(True):
#         with open(opt['sender_file'], 'r') as fr:
#             editable = lock(fr)
#             if editable:
#                 with open(opt['sender_file'], 'w') as fw:
#                     for sample in generated_samples:
#                         fw.write('{}\n'.format(Chem.MolToSmiles(sample)))
#                 break
#             fcntl.flock(fr, fcntl.LOCK_UN)
#     num_samples = len(generated_samples)
#     print("Waiting for retro_star evaluation...")
#     while(True):
#         with open(opt['receiver_file'], 'r') as fr:
#             editable = lock(fr)
#             if editable:
#                 syn_status = []
#                 lines = fr.readlines()
#                 if len(lines) == num_samples:
#                     for idx, line in enumerate(lines):
#                         splitted_line = line.strip().split()
#                         syn_status.append((idx, splitted_line[2]))
#                     break
#             fcntl.flock(fr, fcntl.LOCK_UN)
#         time.sleep(1)
#     assert len(generated_samples) == len(syn_status)
#     return np.mean([int(eval(s[1])) for s in syn_status])
