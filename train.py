import os
import argparse
import os
import random
import timeit
from itertools import cycle, islice, repeat
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import optim
from torch_geometric.datasets import word_net, RelLinkPredDataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.nn import GAE
from tqdm import tqdm

import wandb
from model.RGCN import RGCNEncoder, RGCNDecoder


# RGCN parts of code are based on the official implementation
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn_link_pred.py

def get_data_loader(opt):
    data = word_net.WordNet18RR(opt.dataset)
    loader = DataLoader(data, opt.batch_size, opt.shuffle)
    return loader


def train(model, optimizer, data, opt):
    model.train()

    start_time = timeit.default_timer()

    enc = model.encode(data.edge_index, data.edge_type)
    pos_pred = model.decode(enc, data.train_edge_index, data.train_edge_type)
    neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
    neg_pred = model.decode(enc, neg_edge_index, data.train_edge_type)
    out = torch.cat([pos_pred, neg_pred])
    gt = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = enc.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    epoch_time = timeit.default_timer() - start_time
    print(f"Epoch time {epoch_time}")

    log_metrics = {
        "train_epoch_loss": float(loss),
    }
    return log_metrics


@torch.no_grad()
def validation(model, data, edge_index, edge_type, opt, save_images=False):
    model.eval()
    start_time = timeit.default_timer()

    enc = model.encode(data.edge_index, data.edge_type)

    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(enc, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(enc, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    mrr = (1. / torch.tensor(ranks, dtype=torch.float)).mean()

    epoch_time = timeit.default_timer() - start_time

    log_metrics = {
        "mrr": mrr,
        "val_evaluation_time": epoch_time,
    }
    return log_metrics


def create_arg_parser(model_choices=None, optimizer_choices=None, scheduler_choices=None):
    # Default values for choices
    if scheduler_choices is None:
        scheduler_choices = {'cycliclr': optim.lr_scheduler.CyclicLR}
    if optimizer_choices is None:
        optimizer_choices = {'adamw': optim.AdamW}
    if model_choices is None:
        model_choices = {}

    parser = argparse.ArgumentParser()
    # Wandb logging options
    parser.add_argument('-entity', '--entity', type=str, default="weird-ai-yankovic",
                        help="Name of the team. Multiple projects can exist for the same team.")
    parser.add_argument('-project_name', '--project_name', type=str, default="knowledge-graph-completion",
                        help="Name of the project. Each experiment in the project will be logged separately"
                             " as a group")
    parser.add_argument('-group', '--group', type=str, default="default_experiment",
                        help="Name of the experiment group. Each model in the experiment group will be logged "
                             "separately under a different type.")
    parser.add_argument('-save_model_wandb', '--save_model_wandb', type=bool, default=True,
                        help="Save best model to wandb run.")
    parser.add_argument('-job_type', '--job_type', type=str, default="train",
                        help="Job type {train, eval}.")
    parser.add_argument('-tags', '--tags', nargs="*", type=str, default="train",
                        help="Add a list of tags that describe the run.")

    # Dataset options
    parser.add_argument('-d', '--dataset', type=str, default="data/wordnet18rrtemp",
                        help="Path to the dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('-shuffle', '--shuffle', type=bool, default=False, help="Shuffle dataset")
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help="Number of workers to be used")
    parser.add_argument('-seed_dataset', '--seed_dataset', type=int, default=-1, help="Set random seed for dataset")

    # Model options
    parser.add_argument('-m', '--model', type=str.lower, default="RGCN",
                        choices=model_choices.keys(),
                        help=f"Model to be used for training {model_choices.keys()}")
    # - RGCN
    parser.add_argument('-depth', '--depth', type=int, default=0, help="Model depth")
    parser.add_argument('-hidden_dim', '--hidden_dim', type=int, default=8, help="Number of hidden dims")
    parser.add_argument('-out_dim', '--out_dim', type=int, default=8, help="Number of out channels")
    parser.add_argument('-num_bases', '--num_bases', type=int, default=8, help="Number of bases")
    parser.add_argument('-num_blocks', '--num_blocks', type=int, default=5, help="Number of bases")
    parser.add_argument('-mlp_dim', '--mlp_dim', type=int, default=3,
                        help="Dimension of mlp at the end of the model. Should be the same as the number of classes")
    parser.add_argument('-dropout', '--dropout', type=float, default=0.2, help="Dropout used in models")

    # Training options
    parser.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    parser.add_argument('-e', '--n_epochs', type=int, default=500, help="Max number of epochs for the current model")
    parser.add_argument('-max_e', '--max_epochs', type=int, default=500, help="Maximum number of epochs for all models")
    parser.add_argument('-min_e', '--min_epochs', type=int, default=500, help="Minimum number of epochs for all models")
    parser.add_argument('-nm', '--n_models', type=int, default=1, help="Number of models to be trained")
    parser.add_argument('-pp', '--parallel_processes', type=int, default=1,
                        help="Number of parallel processes to spawn for models [0 for all available cores]")
    parser.add_argument('-seed_everything', '--seed_everything', type=int, default=-1,
                        help="Set random seed for everything")

    # Optimizer options
    parser.add_argument('-optim', '--optimizer', type=str.lower, default="adam",
                        choices=optimizer_choices.keys(),
                        help=f'Optimizer to be used {optimizer_choices.keys()}')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument('-momentum', '--momentum', type=float, default=0.9,
                        help="Momentum value for optimizers like SGD")

    # Scheduler options
    parser.add_argument('-sch', '--scheduler', type=str.lower, default='cycliclr',
                        choices=scheduler_choices.keys(),
                        help=f'Optimizer to be used {scheduler_choices.keys()}')
    parser.add_argument('-base_lr', '--base_lr', type=float, default=3e-4,
                        help="Base learning rate for scheduler")
    parser.add_argument('-max_lr', '--max_lr', type=float, default=0.001,
                        help="Max learning rate for scheduler")
    parser.add_argument('-step_size_up', '--step_size_up', type=int, default=0,
                        help="CycleLR scheduler: step size up. If 0, then it is automatically calculated.")
    parser.add_argument('-cyc_mom', '--cycle_momentum', type=bool, default=False,
                        help="CyclicLR scheduler: cycle momentum in scheduler")
    parser.add_argument('-sch_m', '--scheduler_mode', type=str, default="triangular2",
                        choices=['triangular', 'triangular2', 'exp_range'],
                        help=f"CyclicLR scheduler: mode {['triangular', 'triangular2', 'exp_range']}")
    return parser


def find_balanced_chunk_size(lst_size, n_processes):
    chunk = lst_size // (n_processes - 1)
    # Balanced chunks (ex. list of len 50 will be split into 4 chunks of lengths [13,13,13,11] instead of [16,16,16,2]
    while lst_size % chunk < chunk and lst_size // (chunk - 1) < n_processes:
        chunk -= 1
    return chunk


def get_chunked_lists(opt):
    model_ids = [f'model_{i}' for i in range(opt.n_models)]
    if opt.parallel_processes == 0:
        chunk = len(model_ids) // (mp.cpu_count() - 1)
    else:
        chunk = len(model_ids) // (opt.parallel_processes - 1)
        # Balanced chunks
        while len(model_ids) % chunk < chunk and len(model_ids) // (chunk - 1) < opt.parallel_processes:
            chunk -= 1
    epochs = [e for e in range(10, opt.max_epochs)]
    epoch_ranges = list(islice(cycle(epochs), opt.n_models))
    epoch_splits = [epoch_ranges[i:i + chunk] for i in range(0, len(epoch_ranges), chunk)]
    model_id_splits = [model_ids[i:i + chunk] for i in range(0, len(model_ids), chunk)]
    return epoch_splits, model_id_splits


# 2. Set the random seeds

def set_seed(seed_num):
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pass_right_constructor_arguments(target_class, opt):
    # TODO: Create an instance of a class by sending only the arguments that exist in the constructor
    pass


def create_experiments():
    parser = create_arg_parser()
    opt = parser.parse_args()

    model_ids = [f'model_{i}' for i in range(opt.n_models)]

    epoch_ranges = torch.linspace(opt.min_epochs, opt.max_epochs - 1, opt.n_models).long()
    # TODO: Add experiment description to args and log it in wandb

    functions_iter = repeat(run_experiment)
    args_iter = zip(model_ids)
    kwargs_iter = [{}]

    if opt.parallel_processes <= 1:
        # It is faster to run the experiments on the main process if only one process is used
        for f, args, kwargs in zip(functions_iter, args_iter, kwargs_iter):
            _proc_starter(f, args, kwargs)
    else:
        failed_process_args_kwargs = []
        with mp.Pool(opt.parallel_processes) as pool:
            for f, ret_args, ret_kwargs, process_failed in pool.starmap(_proc_starter,
                                                                        zip(functions_iter, args_iter, kwargs_iter)):
                if process_failed:
                    failed_process_args_kwargs.append((f, ret_args, ret_kwargs))
        print(f"Failed models: {len(failed_process_args_kwargs)}.")
        for f in failed_process_args_kwargs:
            print(f"Failed model: {f[1][0]}")
        n_retry_attempts = opt.n_models
        while failed_process_args_kwargs and n_retry_attempts > 0:
            val = failed_process_args_kwargs.pop(0)
            f, ret_args, ret_kwargs, process_failed = _proc_starter(val[0], val[1], val[2])
            if process_failed:
                failed_process_args_kwargs.append((f, ret_args, ret_kwargs))
            n_retry_attempts -= 1


def _proc_starter(f, args, kwargs):
    return f, *f(*args, **kwargs)


def key_pair(module) -> tuple:
    return module.__name__.lower(), module


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(),))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(),))
    return neg_edge_index


def run_experiment(model_id, *args, **kwargs):
    model_choices = dict(map(key_pair, []))
    optimizer_choices = dict(map(key_pair, [optim.AdamW, optim.SGD, optim.Adam]))
    scheduler_choices = dict(map(key_pair, [optim.lr_scheduler.CyclicLR]))

    parser = create_arg_parser(model_choices=model_choices, optimizer_choices=optimizer_choices,
                               scheduler_choices=scheduler_choices)
    opt = parser.parse_args()

    if opt.seed_everything >= 0:
        opt.seed_dataset = opt.seed_everything
        set_seed(opt.seed_everything)

    # Add specific options for experiments

    opt.device = 'cuda' if torch.cuda.is_available() and (opt.device == 'cuda') else 'cpu'
    print(opt.device)
    if opt.device == 'cuda':
        print(f'GPU {torch.cuda.get_device_name(0)}')

    dataset = RelLinkPredDataset(opt.dataset, 'FB15k-237')
    data = dataset[0]

    wb_run_train = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                              # save_code=True, # Pycharm complains about duplicate code fragments
                              job_type=opt.job_type,
                              tags=opt.tags,
                              name=f'{model_id}_train',
                              config=opt,
                              )

    # Define model
    model = GAE(
        encoder=RGCNEncoder(num_nodes=data.num_nodes, h_dim=opt.hidden_dim,
                            out_dim=opt.out_dim, num_rels=dataset.num_relations // 2,
                            num_bases=opt.num_bases, num_h_layers=opt.depth, num_blocks=opt.num_blocks),
        decoder=RGCNDecoder(num_rels=dataset.num_relations // 2, h_dim=opt.hidden_dim)
    )

    model = model.to(opt.device)

    optimizer = optimizer_choices[opt.optimizer](params=model.parameters(), lr=opt.learning_rate,
                                                 weight_decay=opt.weight_decay)

    best_model_mrr = -np.Inf
    best_model_path = None
    artifact = wandb.Artifact(
        name=f'{model_id}.pt',
        type='model')

    try:
        for epoch in range(1, opt.n_epochs + 1):
            print(f"{epoch=}")
            train_metrics = train(model=model, optimizer=optimizer, data=data, opt=opt)
            val_metrics = validation(model=model, optimizer=optimizer, data=data, edge_index=data.valid_edge_index,
                                     edge_type=data.valid_edge_type, opt=opt)

            wandb.log(train_metrics)
            wandb.log(val_metrics)

            if val_metrics['mrr'] > best_model_mrr:
                print(f"Saving model with new best {val_metrics['mrr']=}")
                best_model_mrr, best_epoch = val_metrics['mrr'], epoch
                Path(f'experiments/{opt.group}').mkdir(exist_ok=True, parents=True)
                new_best_path = os.path.join(f'experiments/{opt.group}',
                                             f'train-{opt.group}-{model_id}-max_epochs{opt.n_epochs}-epoch{epoch}'
                                             f'-metric{val_metrics["mrr"]:.4f}.pt')
                torch.save(model.state_dict(), new_best_path)
                if best_model_path:
                    os.remove(best_model_path)
                best_model_path = new_best_path

        if opt.save_model_wandb:
            artifact.add_file(best_model_path)
            wb_run_train.log_artifact(artifact)

        wb_run_train.finish()

    except FileNotFoundError as e:
        wb_run_train.finish()
        print(f"Exception happened for model {model_id}\n {e}")
        return [model_id, *args], {
            **kwargs}, True  # Run Failed is True

    # Test loading
    opt.job_type = "eval"
    wb_run_eval = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                             # save_code=True, # Pycharm complains about duplicate code fragments
                             job_type=opt.job_type,
                             tags=opt.tags,
                             name=f'{model_id}_eval',
                             config=opt,
                             )

    model = GAE(
        encoder=RGCNEncoder(num_nodes=data.num_nodes, h_dim=opt.hidden_dim,
                            out_dim=opt.out_dim, num_rels=dataset.num_relations // 2,
                            num_bases=opt.num_bases, num_h_layers=opt.depth, num_blocks=opt.num_blocks),
        decoder=RGCNDecoder(num_rels=dataset.num_relations // 2, h_dim=opt.hidden_dim)
    )

    model = model.to(opt.device)
    model.load_state_dict(torch.load(best_model_path))
    model.to(opt.device)
    try:
        eval_metrics = validation(model=model, data=data, edge_index=data.test_edge_index,
                                  edge_type=data.test_edge_type, opt=opt)
        wandb.log(eval_metrics)
        wb_run_eval.finish()
    except FileNotFoundError as e:
        wb_run_eval.finish()
        print(f"Exception happened for model {model_id}\n {e}")
        return [model_id, *args], {
            **kwargs}, True  # Run Failed is True
    return [model_id, *args], {**kwargs}, False  # Run Failed is False


def main():
    create_experiments()


if __name__ == "__main__":
    main()
