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
from agent import Agent
from diffusion.diffusion_model import DiffuseConditionGNNGrammar, DiffuseConditionMPNGrammar, ConditionPostProcessing
from diffusion.utils import print_model_params
from chemprop.args import TrainArgs
from fuseprop import get_mol, get_smiles
from sklearn.metrics import r2_score


def train(model, optimizer, data, batch_size, num_mols):
    model[0].train()
        
    optimizer.zero_grad()

    assert(num_mols == torch.max(data.data_for_gnn_idx) + 1)

    out = model[0](data.x_meta, data.data_for_gnn)
    assert(out.shape[0] == num_mols)
    
    lf = torch.nn.L1Loss()
    
    if model[1] is not None:
        model[1].train()

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

        num_batches = math.ceil(num_train_samples / batch_size)

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
    
    else:
        if out.dim() > 2:
            loss = lf(out[data.train_mask], torch.squeeze(data.y[data.train_mask]))
        else:
            loss = lf(out[data.train_mask], data.y[data.train_mask])
            
        loss.backward()
        optimizer.step()
        return loss.item()
            

def test(model, data, batch_size, num_mols):  # opt required for runtime polymorphism
    model[0].eval()
    
    pred_val_test = []

    with torch.no_grad():
        assert(num_mols == torch.max(data.data_for_gnn_idx) + 1)

        out = model[0](data.x_meta, data.data_for_gnn)
        assert(out.shape[0] == num_mols)
        
        lf = torch.nn.L1Loss()
        
        if model[1] is not None:
            model[1].eval()
            testing_idx = np.where(np.in1d(data.data_for_gnn_idx.detach().cpu().numpy(), data.test_mask.detach().cpu().numpy()))[0]
            conditions_mean = torch.mean(data.conditions, dim=0).float()
            conditions_std = torch.std(data.conditions, dim=0).float()
            num_test_samples = testing_idx.shape[0]

            conditions_test = data.conditions[testing_idx, :].clone().float()

            conditions_norm = (conditions_test - conditions_mean) / conditions_std
            data_for_gnn_idx_test = data.data_for_gnn_idx[testing_idx, :]
            y_test = data.y[testing_idx, :]

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

                loss = lf(pred_batch, y_batch)
                all_loss.append(loss.item() * (end_idx - start_idx))

            return np.sum(all_loss) / num_test_samples, np.concatenate(pred_val_test)
        
        else:
            if out.dim() > 2:
                loss = lf(out[data.test_mask], torch.squeeze(data.y[data.test_mask]))
            else:
                loss = lf(out[data.test_mask], data.y[data.test_mask])
            return loss, out[data.test_mask].cpu().numpy()


def diffusion(l_input_graphs_dict, model, dataset, meta_DAG, agent_epoch, MCMC_num, logger, device, opt):
    print(torch.get_num_threads())
    
    meta_DAG.reset_JT()
    for i in range(len(list(l_input_graphs_dict.values()))):
        print('======{}========'.format(i))
        hg = list(l_input_graphs_dict.values())[i]#.get_JT_graph()
        meta_DAG.add_JT_node_with_conditions(hg, with_condition=opt['with_condition'])
            
    dataset.set_data(meta_DAG)
    data = dataset.data.to(device)
    
    model[0].init_diffusion(dataset)
    print_model_params(model[0])
    
    model_param_group = []
    if opt['feat_arch'] == 'GNN':
        model_param_group.append({"params": model[0].gnn.parameters(), "lr":  opt['lr']})
        model_param_group.append({"params": model[0].graph_pred_linear.parameters(), "lr": opt['lr']})
    else:
        model_param_group.append({"params": model[0].mpn.parameters(), "lr":  opt['lr']})
        model_param_group.append({"params": model[0].fnn.parameters(), "lr": opt['lr']})
    model_param_group.append({"params": model[0].odeblock.parameters(), "lr": opt['lr']})
    model_param_group.append({"params": model[0].meta_mapping.parameters(), "lr": opt['lr']})
    
    if model[1] is not None:
        model_param_group.append({"params": model[1].ffn.parameters(), "lr": opt['lr']})

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
        if (opt['diffusion_type'] == "warmup_diffusion" and agent_epoch > 2):
            model[0].with_diffusion = True
            model_param_group = []
            if opt['feat_arch'] == 'GNN':
                model_param_group.append({"params": model[0].gnn.parameters(), "lr":  opt['lr'] * opt['lr_scaling']})
                model_param_group.append({"params": model[0].graph_pred_linear.parameters(), "lr": opt['lr'] * opt['lr_scaling']})
            else:
                model_param_group.append({"params": model[0].mpn.parameters(), "lr":  opt['lr'] * opt['lr_scaling']})
                model_param_group.append({"params": model[0].fnn.parameters(), "lr": opt['lr'] * opt['lr_scaling']})
            model_param_group.append({"params": model[0].odeblock.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
            model_param_group.append({"params": model[0].meta_mapping.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
            
            if model[1] is not None:
                model_param_group.append({"params": model[1].ffn.parameters(), "lr": opt['lr'] / opt['lr_scaling']})
                
            optimizer = torch.optim.Adam(model_param_group, lr=opt['lr'], weight_decay=opt['decay'])

        start_time = time.time()
        
        if opt['feat_arch'] == 'GNN':
            num_mols = data.data_for_gnn.batch.max().item() + 1
        else:
            num_mols = len(data.data_for_gnn)
        
        loss = train(model, optimizer, data, opt['batch_size'], num_mols=num_mols)
        tmp_test_loss, tmp_pred_vals = test(model, data, opt['batch_size'], num_mols=num_mols)
        
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


def main(mol_sml, prop_values, opt):
    torch.set_num_threads(20)
    print(torch.get_num_threads())
    
    # Create logger
    save_log_path = 'log-{}-{}-{}-seed{}-{}'.format(os.path.basename(opt['dataset']).split('.')[0], opt['target'], opt['feat_arch'], opt['seed'], time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir('./') if f.endswith('.py')])
    
    logger = create_logger('global_logger', save_log_path + '/log.txt')
    logger.info('args:{}'.format(pprint.pformat(opt)))
    logger = logging.getLogger('global_logger')
    
    subgraph_set_init, input_graphs_dict_init = data_processing(mol_sml, prop_values, "../GCN/model_gin/supervised_contextpred.pth", motif=opt['motif'], with_condition=opt['with_condition'])
    
    # agent
    agent = Agent(feat_dim=opt['agent_feat_dim'], hidden_size=opt['agent_hidden_size'])
    agent_optim = optim.Adam(agent.parameters(), lr=opt['agent_lr'])
    
    # grammar diffusion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # meta DAG
    meta_DAG = MetaDAG(opt['metaDAG_path'], opt)
    
    if opt['diffusion_type'] == "no_diffusion" or opt['diffusion_type'] == "warmup_diffusion":
        with_diffusion = False
    elif opt['diffusion_type'] == "with_diffusion":
        with_diffusion = True
    else:
        raise TypeError("diffusion_type should be 'no_diffusion' or 'with_diffusion' or 'warmup_diffusion'")
    
    if opt['feat_arch'] == "GNN":
        model = [DiffuseConditionGNNGrammar(opt, device, with_diffusion=with_diffusion, drop_ratio=opt['dropout_ratio'] if not opt['finetune'] else 0), ConditionPostProcessing(opt) if opt['with_condition'] else None]
    elif opt['feat_arch'] == "MPN":
        model_args = TrainArgs()
        with open(opt['MPN_args_json'], 'r') as fr:
            load_dict = json.load(fr)
        model_args.from_dict(load_dict, skip_unsettable=True)
        model_args.device = device
        model_args.dropout = opt['dropout_ratio']
        model = [DiffuseConditionMPNGrammar(opt, model_args, device, with_diffusion=with_diffusion), ConditionPostProcessing(opt) if opt['with_condition'] else None]
    else:
        raise TypeError("feat_arch should be 'GNN' or 'MPN'")

    model[0].to(device)
    if model[1] is not None:
        model[1].to(device)
        
    dataset = DAGDataset(input_graphs_dict_init, testdata_idx_file=opt['testdata_idx_file'], num_features=300, num_mols=len(input_graphs_dict_init.keys()), classification=opt['classification'], seed=opt['seed'])

    logger.info('starting\n')
    curr_max_R = 0
    curr_best_loss = torch.tensor(0) if opt['classification'] else np.inf

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

            testing_idx = np.where(np.in1d(dataset.data.data_for_gnn_idx.cpu().numpy(), dataset.test_mask.cpu().numpy()))[0]
            try:
                assert(pred_vals.shape[0] == len(testing_idx.tolist()))
            except:
                import pdb; pdb.set_trace()
                
            gt_unnorm = dataset.data.y[testing_idx].detach().cpu().numpy() * dataset.data_y_std.item() + dataset.data_y_mean.item()
            pred_vals_unnorm = pred_vals * dataset.data_y_std.item() + dataset.data_y_mean.item()
            curr_r2 = r2_score(gt_unnorm, pred_vals_unnorm)
            
            if (diffusion_loss < curr_best_loss and (not opt['classification'])) or (diffusion_loss > curr_best_loss and (opt['classification'])):
                diffusion_save_path = os.path.join(save_log_path, 'diffusion_model')
                if not os.path.isdir(diffusion_save_path):
                    os.makedirs(diffusion_save_path)
                    
                torch.save(model[0].state_dict(), os.path.join(diffusion_save_path, 'DiffusionGNN{}_{:2f}.pkl'.format(train_epoch, curr_r2)))
                if model[1] is not None:
                    torch.save(model[1].state_dict(), os.path.join(diffusion_save_path, 'DiffusionGNNConditionPostProcessing{}_{:2f}.pkl'.format(train_epoch, curr_r2)))
                    
                with open(os.path.join(diffusion_save_path, 'epoch_grammar_{}_{:2f}.pkl'.format(train_epoch, curr_r2)), 'wb') as outp:
                    pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                    
                with open(os.path.join(diffusion_save_path, 'epoch_input_graphs_{}_{:2f}.pkl'.format(train_epoch, curr_r2)), 'wb') as outp:
                    pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                    
                with open(os.path.join(diffusion_save_path, 'results-epoch{}-r2{:2f}.txt'.format(train_epoch, curr_r2)), 'w') as outp:
                    if opt['multi_task']:
                        outp.write('mean:{}, std:{}\n'.format(dataset.data_y_mean.tolist(), dataset.data_y_std.tolist()))
                    else:
                        outp.write('mean:{}, std:{}\n'.format(dataset.data_y_mean.item(), dataset.data_y_std.item()))

                    outp_str = 'sml, pred_val_norm, pred_val, gt_val_norm, gt_val'
                    if opt['with_condition']:
                        outp_str += ', '
                        outp_str += ', '.join(opt['condition_names'])
                    outp_str += '\n'
                    outp.write(outp_str)
                        
                    assert(pred_vals.shape[0] == len(testing_idx.tolist()))
                    for i, id in enumerate(testing_idx.tolist()):
                        graph_id = dataset.data.data_for_gnn_idx[id].item()
                        gt_norm = dataset.data.y[id].item()
                        gt = gt_norm * dataset.data_y_std.item() + dataset.data_y_mean.item()
                        pred_v_unnnorm = pred_vals[i] * dataset.data_y_std.item() + dataset.data_y_mean.item()
                        if opt['with_condition']:
                            outp_str = '{}, {}, {}, {}, {}, '.format(list(l_input_graphs_dict.values())[graph_id].smiles, pred_vals[i], pred_v_unnnorm, gt_norm, gt)
                            conditions = [str(c) for c in dataset.data.conditions[id].tolist()]
                            outp_str += ", ".join(conditions)
                            outp_str += '\n'
                            outp.write(outp_str)
                        else:
                            outp.write('{}, {}, {}, {}, {}\n'.format(list(l_input_graphs_dict.values())[graph_id].smiles, pred_vals[i], pred_v_unnnorm, gt_norm, gt))
                            

                curr_best_loss = diffusion_loss

            # Grammar evaluation
            eval_metric = {}
            eval_metric['diffusion_loss'] = diffusion_loss.item()
            logger.info("eval_metrics: {}".format(eval_metric))
            
            # Record metrics
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
    # Data args
    parser.add_argument('--dataset', type=str, required=True, help="file name of the dataset")
    parser.add_argument('--target', type=str, required=True, help="target")
    parser.add_argument("--with_condition", default=False, action="store_true", help="use conditions or not")
    parser.add_argument("--condition_names", metavar='S', type=str, nargs='+', help="list of conditions")
    parser.add_argument("--testdata_idx_file", type=str, default=None, help="file name of the testdata_idx")
    parser.add_argument("--metaDAG_path", type=str, default='../datasets/dag10_complete.pickle', help="file name of the metaDAG pickle")
    parser.add_argument("--MPN_args_json", type=str, default='../datasets/freesolv_args.json' , help="file name of the MPN args json")
    
    parser.add_argument('--extended_DAG', action='store_true', default=False, help="use extended grammar DAG")
    parser.add_argument('--DAG_edit', action='store_true', default=False, help="use edited DAG")
    parser.add_argument('--feat_arch', choices=['GNN', 'MPN'], default='MPN', help="architecture of the feature extractor")
    parser.add_argument('--motif', default='motif', choices=['edge', 'motif', 'scaffold'], help="motif type")
    parser.add_argument('--multi_task', action='store_true', default=False, help="multi task training")
    parser.add_argument('--classification', default=False, action='store_true', help="classification")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size for training, only used for with_condition")
    # Agent
    parser.add_argument('--agent_hidden_size', type=int, default=128, help="hidden size of the agent")
    parser.add_argument('--agent_feat_dim', type=int, default=300, help="input feature dimension of the agent")
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
    parser.add_argument('--intermediate_feat_dim', type=int, default=16, help="intermediate feature dimension of the feature extractor")
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

    def str2float(x):
        if x == "":
            return np.nan
        else:
            return float(x)
    
    mol_sml = []
    prop_values = []
    
    with open(args.dataset, 'r') as csvfile:
        fr = csv.DictReader(csvfile, delimiter=',')
        
        for l, line in enumerate(fr):
            if l == 0:
                assert opt["target"] in line.keys(), "target {} not in the dataset".format(opt["target"])
                
                if opt["with_condition"]:
                    for c in opt["condition_names"]:
                        assert c in line.keys(), "condition {} not in the dataset".format(c)
            
            sml = line["smiles"]
            sml = get_smiles(get_mol(sml))
            mol_sml.append(sml)
            
            prop_v = [None, []]
            prop_v[0] = str2float(line[opt["target"]])
            
            if opt["with_condition"]:
                for c in opt["condition_names"]:
                    prop_v[1].append(str2float(line[c]))
            
            prop_values.append(tuple(prop_v))
            
    main(mol_sml, prop_values, opt)
