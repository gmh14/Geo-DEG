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
import matplotlib.pyplot as plt
import fcntl
import sys
import json
import csv
sys.path.append("../")

from private import *
from grammar_DAG_gen import DAG_MCMC_sampling, data_processing
from grammar_dag import GrammarDAG, GrammarNode, MetaDAG, DAGDataset
# from grammar_generation import data_processing
from agent import Agent
from diffusion.diffusion_model import DiffuseGNNGrammar, DiffuseMPNGrammar, DiffuseBLGNNGrammar, DiffuseConditionMPNGrammar, ConditionPostProcessing
from diffusion.utils import print_model_params
from chemprop.args import TrainArgs
from grammar_construct import evaluate
from sklearn.metrics import roc_auc_score
from fuseprop import get_mol, get_smiles
from sklearn.metrics import r2_score


def train(model, optimizer, data, classification=False):
    model.train()
    optimizer.zero_grad()
    # feat = data.x_init
    # out = model(data.x_meta, data.x)
    out = model(data.x_meta, data.data_for_gnn)

    if classification:
        lf = torch.nn.BCEWithLogitsLoss()
        loss = lf(out[data.train_mask], data.y[data.train_mask])
    else:
        # lf = torch.nn.MSELoss()
        lf = torch.nn.L1Loss()
        if out.dim() > 2:
            loss = lf(out[data.train_mask], torch.squeeze(data.y[data.train_mask]))
        else:
            loss = lf(out[data.train_mask], data.y[data.train_mask])
    # loss = lf(out, data.y)

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()
    return loss.item()

def test(model, data, classification=False):  # opt required for runtime polymorphism
    model.eval()
    # feat = data.x_init
    # out = model(feat)
    # out = model(data.x_meta, data.x)
    with torch.no_grad():
        out = model(data.x_meta, data.data_for_gnn)
    if classification:
        lf = torch.nn.BCEWithLogitsLoss()
        loss = lf(out[data.test_mask], data.y[data.test_mask])
    else:
        # lf = torch.nn.MSELoss()
        lf = torch.nn.L1Loss()
        if out.dim() > 2:
            loss = lf(out[data.test_mask], torch.squeeze(data.y[data.test_mask]))
        else:
            loss = lf(out[data.test_mask], data.y[data.test_mask])
    
    if classification:
        y_true = data.y[data.test_mask].cpu().numpy()
        y_scores = out[data.test_mask].cpu().numpy()
        roc_list = []
        #AUC is only defined when there is at least one positive data.
        roc_list.append(roc_auc_score(y_true, y_scores))
        return sum(roc_list)/len(roc_list), out.cpu().numpy()
        
    return loss, out.cpu().numpy()

def train_with_condition(model, optimizer, data):
    model[0].train()
    model[1].train()
    optimizer.zero_grad()

    num_total_samples = data.data_for_gnn_idx.size(0)
    num_mols = len(data.data_for_gnn)
    assert(num_mols == torch.max(data.data_for_gnn_idx) + 1)

    out = model[0](data.x_meta, data.data_for_gnn)
    assert(out.shape[0] == num_mols)

    conditions_mean = torch.mean(data.conditions, dim=0).float()
    conditions_std = torch.std(data.conditions, dim=0).float()
    training_idx = np.where(np.in1d(data.data_for_gnn_idx.detach().cpu().numpy(), data.train_mask.detach().cpu().numpy()))[0]

    num_train_samples = training_idx.shape[0]
    random_idx = torch.randperm(num_train_samples)
    training_idx = training_idx[random_idx]

    conditions_perm = data.conditions[training_idx, :].clone().float()
    conditions_perm_norm = (conditions_perm - conditions_mean) / conditions_std

    data_for_gnn_idx_perm = data.data_for_gnn_idx[training_idx, :]
    y_perm = data.y[training_idx, :]

    batch_size = 256
    num_batches = math.ceil(num_train_samples / batch_size)

    lf = torch.nn.L1Loss()
    all_loss_item = []
    all_loss_tensor_list = []
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_train_samples)
        out_batch = out[data_for_gnn_idx_perm[start_idx:end_idx, :].squeeze().cpu().numpy(), :]
        conditions_batch = conditions_perm_norm[start_idx:end_idx, :]
        y_batch = y_perm[start_idx:end_idx, :]

        pred_batch = model[1](out_batch, conditions_batch)

        loss = lf(pred_batch, y_batch)
        all_loss_tensor_list.append(loss)
        all_loss_item.append(loss.item() * (end_idx - start_idx))
        print('batch %d/%d, loss: %f' % (b, num_batches, loss.item()))

    torch.stack(all_loss_tensor_list, dim=0).sum().backward()
    optimizer.step()
    return np.sum(all_loss_item) / num_train_samples

def test_with_condition(model, data):  # opt required for runtime polymorphism
    model[0].eval()
    model[1].eval()
    
    pred_val_test = []

    with torch.no_grad():
        num_mols = len(data.data_for_gnn)
        assert(num_mols == torch.max(data.data_for_gnn_idx) + 1)

        out = model[0](data.x_meta, data.data_for_gnn)
        assert(out.shape[0] == num_mols)
        testing_idx = np.where(np.in1d(data.data_for_gnn_idx.detach().cpu().numpy(), data.test_mask.detach().cpu().numpy()))[0]
        conditions_mean = torch.mean(data.conditions, dim=0).float()
        conditions_std = torch.std(data.conditions, dim=0).float()
        num_test_samples = testing_idx.shape[0]

        conditions_test = data.conditions[testing_idx, :].clone().float()

        conditions_norm = (conditions_test - conditions_mean) / conditions_std
        data_for_gnn_idx_test = data.data_for_gnn_idx[testing_idx, :]
        y_test = data.y[testing_idx, :]

        batch_size = 256
        num_batches = math.ceil(num_test_samples / batch_size)
        all_loss = []

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_test_samples)
            out_batch = out[data_for_gnn_idx_test[start_idx:end_idx, :].squeeze().cpu().numpy(), :]
            conditions_batch = conditions_norm[start_idx:end_idx, :]
            y_batch = y_test[start_idx:end_idx, :]

            pred_batch = model[1](out_batch, conditions_batch)
            pred_val_test.append(pred_batch.cpu().numpy().squeeze())

            lf = torch.nn.L1Loss()
            loss = lf(pred_batch, y_batch)
            all_loss.append(loss.item() * (end_idx - start_idx))

    return np.sum(all_loss) / num_test_samples, np.concatenate(pred_val_test)

def diffusion(l_input_graphs_dict, model, dataset, meta_DAG, agent_epoch, MCMC_num, logger, device, opt):
    print(torch.get_num_threads())
    meta_DAG.reset_JT()
    for i in range(len(list(l_input_graphs_dict.values()))):
        print('======{}========'.format(i))
        hg = list(l_input_graphs_dict.values())[i]#.get_JT_graph()
        if opt['isAverage']:
            meta_DAG.add_JT_node(hg)
        else:
            meta_DAG.add_JT_node_with_conditions(hg)
    dataset.set_data(meta_DAG)
    data = dataset.data.to(device)
    if (isinstance(model, list)):
        model[0].init_diffusion(dataset)
        parameters = [p for m in model for p in m.parameters() if p.requires_grad]
        print_model_params(model[0])
        print_model_params(model[1])
        model_param_group = []
        assert(opt['feat_arch'] == 'MPN')
        model_param_group.append({"params": model[0].mpn.parameters(), "lr":  opt['lr']})
        model_param_group.append({"params": model[0].fnn.parameters(), "lr": opt['lr']})
        model_param_group.append({"params": model[0].odeblock.parameters(), "lr": opt['lr']})
        model_param_group.append({"params": model[0].meta_mapping.parameters(), "lr": opt['lr']})
        model_param_group.append({"params": model[1].ffn.parameters(), "lr": opt['lr']})
    else:
        model.init_diffusion(dataset)
        parameters = [p for p in model.parameters() if p.requires_grad]
        print_model_params(model)
        model_param_group = []
        if opt['feat_arch'] == 'GNN':
            model_param_group.append({"params": model.gnn.parameters(), "lr":  opt['lr']})
            model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": opt['lr']})
        else:
            model_param_group.append({"params": model.mpn.parameters(), "lr":  opt['lr']})
            model_param_group.append({"params": model.fnn.parameters(), "lr": opt['lr']})
        model_param_group.append({"params": model.odeblock.parameters(), "lr": opt['lr']})
        model_param_group.append({"params": model.meta_mapping.parameters(), "lr": opt['lr']})

    all_train_loss = []
    all_test_loss = []

    if opt['adam']:
        optimizer = torch.optim.Adam(model_param_group, lr=opt['lr'], weight_decay=opt['decay'])
    else:
        optimizer = torch.optim.SGD(model_param_group, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = 0
    test_loss = torch.tensor(0.0) if opt['classification'] else np.inf
    pred_vals = None
    train_loss_list = []
    test_loss_list = []
    logger.info("Start training for MC sample {} in agent epoch {}".format(MCMC_num, agent_epoch))
    for epoch in range(1, opt['epoch']+1):
        # if (opt['diffusion_type'] == "warmup_diffusion" and epoch > 100) or (agent_epoch > 1):
        # if (opt['diffusion_type'] == "warmup_diffusion" and MCMC_num > 1) or (agent_epoch > 1):
        # if (opt['diffusion_type'] == "warmup_diffusion" and epoch > 50): #) or (agent_epoch > 3):
        if (opt['diffusion_type'] == "warmup_diffusion" and agent_epoch > 2):
            if (isinstance(model, list)):
                model[0].with_diffusion = True
                # model.gnn.drop_ratio = 0.0
                model_param_group = []
                assert(opt['feat_arch'] == 'MPN')
                model_param_group.append({"params": model[0].mpn.parameters(), "lr":  opt['lr'] * opt['lr_scaling']})
                model_param_group.append({"params": model[0].fnn.parameters(), "lr": opt['lr'] * opt['lr_scaling']})
                model_param_group.append({"params": model[0].odeblock.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
                model_param_group.append({"params": model[0].meta_mapping.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
                model_param_group.append({"params": model[1].ffn.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
            else:
                model.with_diffusion = True
                # model.gnn.drop_ratio = 0.0
                model_param_group = []
                if opt['feat_arch'] == 'GNN':
                    model_param_group.append({"params": model.gnn.parameters(), "lr":  opt['lr'] * opt['lr_scaling']})
                    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": opt['lr'] * opt['lr_scaling']})
                else:
                    model_param_group.append({"params": model.mpn.parameters(), "lr":  opt['lr'] * opt['lr_scaling']})
                    model_param_group.append({"params": model.fnn.parameters(), "lr": opt['lr'] * opt['lr_scaling']})
                model_param_group.append({"params": model.odeblock.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
                model_param_group.append({"params": model.meta_mapping.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
            optimizer = torch.optim.Adam(model_param_group, lr=opt['lr'], weight_decay=opt['decay'])

        start_time = time.time()
        if opt['isAverage']:
            loss = train(model, optimizer, data, classification=opt['classification'])
            tmp_test_loss, tmp_pred_vals = test(model, data, classification=opt['classification'])
        else:
            loss = train_with_condition(model, optimizer, data)
            tmp_test_loss, tmp_pred_vals = test_with_condition(model, data)
        if (tmp_test_loss < test_loss and (not opt['classification'])) or (tmp_test_loss > test_loss and opt['classification']):
            best_epoch = epoch
            test_loss = tmp_test_loss
            pred_vals = tmp_pred_vals
        train_loss_list.append(loss)
        test_loss_list.append(tmp_test_loss)
        log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, Test: {:.4f}, Best time: {:.4f}'
        logger.info(log.format(epoch, time.time() - start_time, loss, tmp_test_loss, best_time))
    logger.info('For MC sample {} in agent epoch {}, best test {} {:03f} at epoch {:d} and best time {:03f}'.format(MCMC_num, agent_epoch, 'accuracy' if opt['classification'] else 'loss', test_loss, best_epoch, best_time))
    return test_loss, pred_vals

def main(mol_sml, prop_values, test_id_list, opt):

    torch.set_num_threads(20)
    print(torch.get_num_threads())
    # Create logger
    save_log_path = 'new-log-{}-{}-{}-{}-seed{}-{}'.format(opt['dataset'].split('_')[0], opt['property'], opt['feat_arch'], 'average' if opt['isAverage'] else None, opt['seed'], time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir('./') if f.endswith('.py')])
    logger = create_logger('global_logger', save_log_path + '/log-{}-{}.txt'.format(test_id_list[0], test_id_list[1]))
    logger.info('args:{}'.format(pprint.pformat(opt)))
    logger = logging.getLogger('global_logger')
    subgraph_set_init, input_graphs_dict_init = data_processing(mol_sml, prop_values, "../GCN/model_gin/supervised_contextpred.pth", motif=opt['motif'], with_condition=(not opt['isAverage']))
    # agent
    agent = Agent(feat_dim=300, hidden_size=128)
    agent_optim = optim.Adam(agent.parameters(), lr=opt['agent_lr'])
    # grammar diffusion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    meta_DAG = MetaDAG('../datasets/dag10_complete.pickle', opt)
    # meta_DAG = MetaDAG('../datasets/dag10.pickle', opt)
    if opt['diffusion_type'] == "no_diffusion" or opt['diffusion_type'] == "warmup_diffusion":
        with_diffusion = False
    elif opt['diffusion_type'] == "with_diffusion":
        with_diffusion = True
    else:
        raise TypeError("diffusion_type should be 'no_diffusion' or 'with_diffusion' or 'warmup_diffusion'")
    if opt['feat_arch'] == "GNN":
        # model = DiffuseGNNGrammar(opt, device, with_diffusion=with_diffusion, drop_ratio=opt['dropout_ratio'] if not opt['finetune'] else 0)
        model = DiffuseBLGNNGrammar(opt, device, with_diffusion=with_diffusion, drop_ratio=opt['dropout_ratio'] if not opt['finetune'] else 0)
    elif opt['feat_arch'] == "MPN":
        model_args = TrainArgs()
        with open('../datasets/freesolv_args.json', 'r') as fr:
            load_dict = json.load(fr)
        model_args.from_dict(load_dict, skip_unsettable=True)
        model_args.device = device
        model_args.dropout = opt['dropout_ratio']
        if opt['isAverage']:
            model = DiffuseMPNGrammar(opt, model_args, device, with_diffusion=with_diffusion)
        else:
            model = [DiffuseConditionMPNGrammar(opt, model_args, device, with_diffusion=with_diffusion), ConditionPostProcessing(opt)]
    else:
        raise TypeError("feat_arch should be 'GNN' or 'MPN'")

    if (isinstance(model, list)):
        model[0].to(device)
        model[1].to(device)
    else:
        model.to(device) # NEED TO move to GPU first before loading the model
    dataset = DAGDataset(input_graphs_dict_init, test_id_list=test_id_list, num_features=300, num_mols=len(input_graphs_dict_init.keys()), classification=opt['classification'], seed=opt['seed'], isAverage=opt['isAverage'])

    logger.info('starting\n')
    curr_max_R = 0
    curr_best_loss = torch.tensor(0) if opt['classification'] else np.inf
    curr_best_r2 = -np.inf

    for train_epoch in range(1, opt['grammar_epoches']+1):
        print(torch.get_num_threads())
        returns = []
        log_returns = []
        logger.info("<<<<< Epoch {}/{} >>>>>>".format(train_epoch, opt['grammar_epoches']))
        # MCMC sampling
        for num in range(opt['MCMC_size']):
            grammar_init = ProductionRuleCorpus()
            l_input_graphs_dict = deepcopy(input_graphs_dict_init)
            l_subgraph_set = deepcopy(subgraph_set_init)
            l_grammar = deepcopy(grammar_init)
            iter_num, l_grammar, l_input_graphs_dict = DAG_MCMC_sampling(agent, l_input_graphs_dict, l_subgraph_set, l_grammar, num)
            
            # DAG diffusion
            diffusion_loss, pred_vals = diffusion(l_input_graphs_dict, model, dataset, meta_DAG, train_epoch, num, logger, device, opt)

            if opt['isAverage']:
                test_idx = dataset.test_mask.tolist()
                curr_r2 = r2_score(dataset.data.y[test_idx].detach().cpu().numpy(), pred_vals[test_idx])
            else:
                testing_idx = np.where(np.in1d(dataset.data.data_for_gnn_idx.cpu().numpy(), dataset.test_mask.cpu().numpy()))[0]
                assert(pred_vals.shape[0] == len(testing_idx.tolist()))
                gt_unnorm = dataset.data.y[testing_idx].detach().cpu().numpy() * dataset.data_y_std.item() + dataset.data_y_mean.item()
                pred_vals_unnorm = pred_vals * dataset.data_y_std.item() + dataset.data_y_mean.item()
                curr_r2 = r2_score(gt_unnorm, pred_vals_unnorm)
            # if (diffusion_loss < curr_best_loss and (not opt['classification'])) or (diffusion_loss > curr_best_loss and (opt['classification'])):
            if curr_r2 > curr_best_r2:
                diffusion_save_path = os.path.join(save_log_path, 'diffusion_model')
                if not os.path.isdir(diffusion_save_path):
                    os.makedirs(diffusion_save_path)
                if (isinstance(model, list)):
                    torch.save(model[0].state_dict(), os.path.join(diffusion_save_path, 'DiffusionGNN{}_{}.pkl'.format(train_epoch, curr_r2)))
                    torch.save(model[1].state_dict(), os.path.join(diffusion_save_path, 'DiffusionGNNConditionPostProcessing{}_{}.pkl'.format(train_epoch, curr_r2)))
                else:
                    torch.save(model.state_dict(), os.path.join(diffusion_save_path, 'DiffusionGNN{}_{}.pkl'.format(train_epoch, curr_r2)))
                with open(os.path.join(diffusion_save_path, 'epoch_grammar_{}_{}.pkl'.format(train_epoch, curr_r2)), 'wb') as outp:
                    pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(diffusion_save_path, 'epoch_input_graphs_{}_{}.pkl'.format(train_epoch, curr_r2)), 'wb') as outp:
                    pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(diffusion_save_path, 'results_testidList-{}-{}-{}_{}.txt'.format(test_id_list[0], test_id_list[1], train_epoch, curr_r2)), 'w') as outp:
                    train_idx = dataset.train_mask.tolist()
                    test_idx = dataset.test_mask.tolist()
                    if opt['multi_task']:
                        outp.write('mean:{}, std:{}\n'.format(dataset.data_y_mean.tolist(), dataset.data_y_std.tolist()))
                    else:
                        outp.write('mean:{}, std:{}\n'.format(dataset.data_y_mean.item(), dataset.data_y_std.item()))

                    if opt['isAverage']:
                        outp.write('sml, pred_val, gt_val_norm, gt_val, is_train\n')
                        for i, graph in enumerate(l_input_graphs_dict.values()):
                                assert (i in train_idx or i in test_idx)
                                if opt['multi_task']:
                                    outp.write('{}, {}, {}, {}, {}\n'.format(graph.smiles, pred_vals[i][0], dataset.data.y[i].tolist(), graph.prop_value, int(i in train_idx)))
                                else:
                                    outp.write('{}, {}, {}, {}, {}\n'.format(graph.smiles, pred_vals[i][0], dataset.data.y[i][0], graph.prop_value, int(i in train_idx)))
                    else:
                        outp.write('sml, pred_val_norm, pred_val, gt_val_norm, gt_val, pressure, temperature\n')
                        assert(pred_vals.shape[0] == len(testing_idx.tolist()))
                        for i, id in enumerate(testing_idx.tolist()):
                            graph_id = dataset.data.data_for_gnn_idx[id].item()
                            conditions = dataset.data.conditions[id].tolist()
                            gt_norm = dataset.data.y[id].item()
                            gt = gt_norm * dataset.data_y_std.item() + dataset.data_y_mean.item()
                            pred_v_unnnorm = pred_vals[i] * dataset.data_y_std.item() + dataset.data_y_mean.item()
                            outp.write('{}, {}, {}, {}, {}, {}, {}\n'.format(list(l_input_graphs_dict.values())[graph_id].smiles, pred_vals[i], pred_v_unnnorm, gt_norm, gt, conditions[0], conditions[1]))

                curr_best_loss = diffusion_loss
                curr_best_r2 = curr_r2

            # Grammar evaluation
            # eval_metric = evaluate(l_grammar, opt, metrics=['diversity', 'syn'])
            # eval_metric = evaluate(l_grammar, opt, metrics=['diversity'])
            eval_metric = {}
            eval_metric['diffusion_loss'] = diffusion_loss.item()
            logger.info("eval_metrics: {}".format(eval_metric))
            # Record metrics
            # R = eval_metric['diversity'] + 2 * eval_metric['syn'] - 2 * eval_metric['diffusion_loss']
            # R = eval_metric['diversity'] - eval_metric['diffusion_loss']
            R = - eval_metric['diffusion_loss']
            R_ind = R
            returns.append(R)
            log_returns.append(eval_metric)
            logger.info("======Sample {} returns {}=======:".format(num, R_ind))
            # Save ckpt
            if R_ind > curr_max_R:
                torch.save(agent.state_dict(), os.path.join(save_log_path, 'epoch_agent_{}_{}.pkl'.format(train_epoch, R_ind)))
                with open('{}/epoch_grammar_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind), 'wb') as outp:
                    pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                with open('{}/epoch_input_graphs_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind), 'wb') as outp:
                    pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                curr_max_R = R_ind
        # Calculate loss
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) # / (returns.std() + eps)
        assert len(returns) == len(list(agent.saved_log_probs.keys()))
        policy_loss = torch.tensor([0.])
        for sample_number in agent.saved_log_probs.keys():
            max_iter_num = max(list(agent.saved_log_probs[sample_number].keys()))
            for iter_num_key in agent.saved_log_probs[sample_number].keys():
                log_probs = agent.saved_log_probs[sample_number][iter_num_key]
                for log_prob in log_probs:
                    policy_loss += (-log_prob * 0.99 ** (max_iter_num - iter_num_key) * returns[sample_number]).sum()

        # Back Propogation and update
        agent_optim.zero_grad()
        policy_loss.backward()
        agent_optim.step()
        agent.saved_log_probs.clear()

        # Log
        logger.info("Loss: {}".format(policy_loss.clone().item()))
        eval_metrics = {}
        for r in log_returns:
            for _key in r.keys():
                if _key not in eval_metrics:
                    eval_metrics[_key] = []
                eval_metrics[_key].append(r[_key])
        mean_evaluation_metrics = ["{}: {}".format(_key, np.mean(eval_metrics[_key])) for _key in eval_metrics]
        logger.info("Mean evaluation metrics: {}".format(', '.join(mean_evaluation_metrics)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="crow_smiles_and_Tg_celsius.txt", help="file name of the dataset")
    # parser.add_argument('--dataset', type=str, default="freesolv_DGhy_exp.txt", help="file name of the dataset")
    # parser.add_argument('--dataset', type=str, default="permeability_Bayesian.txt", help="file name of the dataset")
    parser.add_argument('--extended_DAG', action='store_true', default=False, help="use extended grammar DAG")
    parser.add_argument('--DAG_edit', action='store_true', default=False, help="use edited DAG")
    parser.add_argument('--feat_arch', choices=['GNN', 'MPN'], help="architecture of the feature extractor")
    parser.add_argument('--property', choices=['thermal', 'dynamic', 'density', 'kinematic', 'heat'], help="property to predict")
    parser.add_argument('--motif', default='edge', choices=['edge', 'motif', 'scaffold'], help="motif type")
    parser.add_argument('--multi_task', action='store_true', default=False, help="multi task training")
    parser.add_argument('--num_samples', type=int, default=-1, help="num_samples")
    parser.add_argument('--classification', default=False, action='store_true', help="classification")
    parser.add_argument('--isAverage', default=False, action='store_true', help="use average")

    # Grammar construct
    parser.add_argument('--grammar_epoches', type=int, default=10, help="maximal training epoches")
    parser.add_argument('--agent_lr', type=int, default=1e-2, help="learning rate")
    parser.add_argument('--sender_file', type=str, default="generated_samples.txt", help="file name of the generated samples")
    parser.add_argument('--receiver_file', type=str, default="output_syn.txt", help="file name of the output file of Retro*")
    parser.add_argument('--MCMC_size', type=int, default=5, help="sample number of each step of MCMC")
    parser.add_argument('--num_generated_samples', type=int, default=100, help="number of generated samples to evaluate grammar")
    # Diffusion
    parser.add_argument('--diffusion_type', default="warmup_diffusion", type=str, choices=['no_diffusion', 'with_diffusion', 'warmup_diffusion'])
    parser.add_argument('--finetune', default=False, action='store_true', help='Use pretrain weights.')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    # GNN args
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads for multi-head attention.')
    parser.add_argument('--dropout_ratio', type=float, default=0.3,
                        help='dropout ratio (default: 0.3)')
    # optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lr_scaling', type=float, default=1e-1, help='Learning rate scaling.')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay for optimization') # does not matter too much
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs per iteration.')
    parser.add_argument('--adam', default=False, action='store_true', help='Use Adam.')
    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")

    args = parser.parse_args()
    opt = vars(args)

    # with open("../datasets/{}".format(args.dataset), 'r') as fr:
    #     lines = fr.readlines()
    #     mol_sml = []
    #     prop_values = []
    #     for line in lines:
    #         if not (line.strip().split(',')[0].strip() in mol_sml):
    #             sml = line.strip().split(',')[0].strip()
    #             mol_sml.append(sml)
    #             if args.multi_task:
    #                 val = [float(line.strip().split(',')[i]) for i in range(1, 7)]
    #             else:
    #                 val = float(line.strip().split(',')[1])
    #             if args.classification:
    #                 val = int(val)
    #             prop_values.append(val)
    def str2float(x):
        if x == "":
            return np.nan
        else:
            return float(x)
        
    assert(args.dataset == "EvonikEster")
    mol_sml = []
    prop_values = []
    fold_idx_list = []
    keys = []
    for fold in range(5):
        with open("../datasets/{}/mol_fold_{}.csv".format(args.dataset, fold), 'r') as csvfile:
            fr = csv.reader(csvfile, delimiter=',')
            id_start = len(mol_sml)

            sml_dict = {}
            for l, line in enumerate(fr):
                if l == 0:
                    continue

                items = line
                try:
                    assert(len(items) == 9)
                except:
                    import pdb; pdb.set_trace() 
                
                sml = items[1].replace("/", "").replace("\\", "")
                sml = get_smiles(get_mol(sml))

                if sml not in sml_dict.keys():
                    sml_dict[sml] = []
                
                pressure = items[2]
                temperature = items[3]

                density = items[4]
                dynamic_viscosity = items[5]
                heat_capacity = items[6]
                kinematic_viscosity = items[7]
                thermal_conductivity = items[8]

                prop_val = None

                if args.property == 'thermal':
                    prop_val = [str2float(pressure), str2float(temperature), str2float(thermal_conductivity)]
                elif args.property == 'dynamic':
                    prop_val = [str2float(pressure), str2float(temperature), str2float(dynamic_viscosity)]
                elif args.property == 'density':
                    prop_val = [str2float(pressure), str2float(temperature), str2float(density)]
                elif args.property == 'kinematic':
                    prop_val = [str2float(pressure), str2float(temperature), str2float(kinematic_viscosity)]
                elif args.property == 'heat':
                    prop_val = [str2float(pressure), str2float(temperature), str2float(heat_capacity)]
                
                sml_dict[sml].append(prop_val)

            if opt['isAverage']:
                for (sml, pv) in sml_dict.items():
                    prop_val_all = [p[2] for p in sml_dict[sml] if not np.isnan(p[2])]
                    if len(prop_val_all) != 0:
                        mol_sml.append(sml)
                        prop_values.append(np.mean(prop_val_all))
            else:
                for (sml, pv) in sml_dict.items():
                    prop_val_all = [p for p in sml_dict[sml] if not np.isnan(p[2])]
                    if len(prop_val_all) != 0:
                        mol_sml.append(sml)
                        prop_values.append(prop_val_all)

            id_end = len(mol_sml)
            fold_idx_list.append((id_start, id_end))
    print(fold_idx_list)

    # Clear the communication files for Retro*
    with open(opt['sender_file'], 'w') as fw:
        fw.write('')
    with open(opt['receiver_file'], 'w') as fw:
        fw.write('')

    for fold in range(5):
        main(mol_sml, prop_values, fold_idx_list[fold], opt)
