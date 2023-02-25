import argparse
import os
import random
import re
import timeit
from itertools import cycle, islice, repeat
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import word_net
from torch_geometric.nn import GCN, RGCNConv, FastRGCNConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.datasets.word_net
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.loader.dataloader import DataLoader

import wandb


def get_data_loader(opt):
    data = word_net.WordNet18RR(opt.dataset)
    loader = DataLoader(data, opt.batch_size, opt.shuffle)
    return loader


def train(model, optimizer, data_loader, opt, scheduler=None):
    model.train()
    total_samples = len(data_loader.dataset)

    global_target = torch.tensor([], device=opt.device)
    global_pred = torch.tensor([], device=opt.device)
    global_probs = torch.empty((0, 3), device=opt.device)

    running_loss = 0.0

    metrics = MetricCollection({'train_f1_micro': MulticlassF1Score(num_classes=opt.num_classes, average='micro'),
                                'train_f1_macro': MulticlassF1Score(num_classes=opt.num_classes, average='macro'),
                                'train_precision': MulticlassPrecision(num_classes=opt.num_classes),
                                'train_recall': MulticlassRecall(num_classes=opt.num_classes),
                                }
                               ).to(opt.device)
    auroc = MulticlassAUROC(num_classes=opt.num_classes, average='macro').to(opt.device)

    start_time = timeit.default_timer()
    for i, (data, target, path) in enumerate(data_loader):
        data = data.to(opt.device)
        target = target.to(opt.device)
        optimizer.zero_grad()
        predictions = model(data)
        probs = F.softmax(predictions, dim=1)
        _, pred = torch.max(probs, dim=1)

        loss = F.nll_loss(probs, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        metrics(pred, target)
        global_target = torch.concatenate((global_target, target))
        global_pred = torch.concatenate((global_pred, pred))
        global_probs = torch.vstack((global_probs, probs))

        running_loss += loss.item() * data.size(0)

        if i % 5 == 0 and scheduler:
            wandb.log({"train_lr": scheduler.get_last_lr()[0]},
                      # commit=False, # Commit=False just accumulates data
                      )

    epoch_loss = running_loss / total_samples

    epoch_time = timeit.default_timer() - start_time
    print(f"Epoch time {epoch_time}")

    auroc(global_probs, global_target.long())
    global_target = global_target.cpu().detach().numpy()
    global_pred = global_pred.cpu().detach().numpy()
    global_probs = global_probs.cpu().detach().numpy()
    log_metrics = {
        **metrics.compute(),
        "train_epoch_loss": epoch_loss,
        "train_epoch_time": epoch_time,
        "train_auroc_macro": auroc.compute(),
        # Values below are used for constructing the wandb plots and tables.
        # They should be deleted after they are used for creating plots and tables, and they should not be logged
        "train_global_probs": global_probs,
        "train_global_target": global_target,
    }
    return log_metrics


def validation(model, data_loader, opt, save_images=False):
    model.eval()

    total_samples = len(data_loader.dataset)

    global_target = torch.tensor([], device=opt.device)
    global_pred = torch.tensor([], device=opt.device)
    global_probs = torch.empty((0, 3), device=opt.device)
    incorrect_img_paths = []
    incorrect_img_labels = torch.tensor([], device=opt.device)
    incorrect_img_predictions = torch.tensor([], device=opt.device)
    incorrect_images = torch.tensor([], device=opt.device)

    running_loss = 0.0

    metrics = MetricCollection({'val_f1_micro': MulticlassF1Score(num_classes=opt.num_classes, average='micro'),
                                'val_f1_macro': MulticlassF1Score(num_classes=opt.num_classes, average='macro'),
                                'val_precision': MulticlassPrecision(num_classes=opt.num_classes),
                                'val_recall': MulticlassRecall(num_classes=opt.num_classes),
                                }
                               ).to(opt.device)
    auroc = MulticlassAUROC(num_classes=opt.num_classes, average='macro').to(opt.device)
    start_time = timeit.default_timer()
    with torch.no_grad():
        for data, target, path in data_loader:
            data = data.to(opt.device)
            target = target.to(opt.device)
            res = model(data)
            probs = F.softmax(res, dim=1)
            probs = probs.to(opt.device)
            loss = F.nll_loss(probs, target, reduction='sum')
            _, pred = torch.max(probs, dim=1)

            incorrect_img_paths += [path[i] for i in range(len(path)) if pred[i] != target[i]]
            incorrect_img_labels = torch.concatenate((incorrect_img_labels, target[pred != target]))
            incorrect_img_predictions = torch.concatenate((incorrect_img_predictions, pred[pred != target]))
            if save_images:
                incorrect_images = torch.concatenate((incorrect_images, data[pred != target]))

            metrics(pred, target)
            global_target = torch.concatenate((global_target, target))
            global_pred = torch.concatenate((global_pred, pred))
            global_probs = torch.vstack((global_probs, probs))

            running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / total_samples

    epoch_time = timeit.default_timer() - start_time

    auroc(global_probs, global_target.long())
    global_target = global_target.cpu().detach().numpy()
    global_pred = global_pred.cpu().detach().numpy()
    global_probs = global_probs.cpu().detach().numpy()
    incorrect_img_labels = incorrect_img_labels.cpu().detach().numpy()
    incorrect_img_predictions = incorrect_img_predictions.cpu().detach().numpy()
    incorrect_images = incorrect_images.cpu().detach().numpy()

    diff_mistakes = [int(re.search(r"(?<=diff)[0-9]", i).group()) for i in incorrect_img_paths]
    shapes_mistakes = [re.search(r"ellipse|triangle|square", i).group() for i in incorrect_img_paths]
    shape_diff_mistakes = [f"{s}_{d}" for s, d in zip(shapes_mistakes, diff_mistakes)]

    log_metrics = {
        **metrics.compute(),
        "val_epoch_loss": epoch_loss,
        "val_evaluation_time": epoch_time,
        "val_auroc_macro": auroc.compute(),
        # Values below are used for constructing the wandb plots and tables.
        # They should be deleted after they are used for creating plots and tables, and they should not be logged
        "val_global_probs": global_probs,
        "val_global_target": global_target,
        "val_diff_mistakes": diff_mistakes,
        "val_shapes_mistakes": shapes_mistakes,
        "val_shape_diff_mistakes": shape_diff_mistakes,
        "val_incorrect_img_paths": incorrect_img_paths,
        "val_incorrect_images": incorrect_images,
        "val_incorrect_img_predictions": incorrect_img_predictions,
        "val_incorrect_img_labels": incorrect_img_labels,
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
    parser.add_argument('-m', '--model', type=str.lower, default="",
                        choices=model_choices.keys(),
                        help=f"Model to be used for training {model_choices.keys()}")
    parser.add_argument('-depth', '--depth', type=int, default=2, help="Model depth")
    parser.add_argument('-in_channels', '--in_channels', type=int, default=1, help="Number of in channels")
    parser.add_argument('-out_channels', '--out_channels', type=int, default=8, help="Number of out channels")
    parser.add_argument('-mlp_dim', '--mlp_dim', type=int, default=3,
                        help="Dimension of mlp at the end of the model. Should be the same as the number of classes")
    parser.add_argument('-dropout', '--dropout', type=float, default=0.2, help="Dropout used in models")

    # Training options
    parser.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    parser.add_argument('-e', '--n_epochs', type=int, default=50, help="Max number of epochs for the current model")
    parser.add_argument('-max_e', '--max_epochs', type=int, default=20, help="Maximum number of epochs for all models")
    parser.add_argument('-min_e', '--min_epochs', type=int, default=5, help="Minimum number of epochs for all models")
    parser.add_argument('-nm', '--n_models', type=int, default=50, help="Number of models to be trained")
    parser.add_argument('-pp', '--parallel_processes', type=int, default=1,
                        help="Number of parallel processes to spawn for models [0 for all available cores]")
    parser.add_argument('-seed_everything', '--seed_everything', type=int, default=-1,
                        help="Set random seed for everything")

    # Optimizer options
    parser.add_argument('-optim', '--optimizer', type=str.lower, default="adamw",
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


def run_experiment(model_id, *args, **kwargs):
    # Model options
    model_choices = {}  # TODO: Add GNN model for knowledge graph completion

    optimizer_choices = {optim.AdamW.__name__.lower(): optim.AdamW,
                         optim.SGD.__name__.lower(): optim.SGD}  # TODO: Add more optimizer choices

    # Scheduler options
    scheduler_choices = {
        optim.lr_scheduler.CyclicLR.__name__.lower(): optim.lr_scheduler.CyclicLR, }  # TODO: Add more scheduler choices

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

    data_loader = get_data_loader(opt)

    # TODO: Determine optimal step_size_up for cyclicLR scheduler.
    # TODO: Change step_size_up formula for graph dataset
    if opt.step_size_up <= 0:
        opt.step_size_up = 2 * len(data_loader.dataset) // opt.batch_size

    wb_run_train = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                              # save_code=True, # Pycharm complains about duplicate code fragments
                              job_type=opt.job_type,
                              # TODO: Add tags as arguments for argparser
                              tags=['variable_max_score'],
                              name=f'{model_id}_train',
                              config=opt,
                              )

    # Define model
    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)  # TODO: Add appropriate model parameters

    model = model.to(opt.device)

    # TODO: Optimize hyper-params with WandB Sweeper
    optimizer = optimizer_choices[opt.optimizer](params=model.parameters(), lr=opt.learning_rate,
                                                 weight_decay=opt.weight_decay)

    scheduler = scheduler_choices[opt.scheduler](optimizer=optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr,
                                                 step_size_up=opt.step_size_up,
                                                 cycle_momentum=opt.cycle_momentum, mode=opt.scheduler_mode)

    # For watching gradients
    #  wandb.watch(net, log='all')

    best_model_f1_macro = -np.Inf
    best_model_path = None
    artifact = wandb.Artifact(
        name=f'{model_id}.pt',
        type='model')

    try:
        # TODO: Add training resuming. This can be done from the model saved in wandb or from the local model
        for epoch in range(1, opt.n_epochs + 1):
            print(f"{epoch=}")
            train_metrics = train(model=model, optimizer=optimizer, data_loader=data_loader, opt=opt,
                                  scheduler=scheduler)
            val_metrics = validation(model=model, data_loader=data_loader, opt=opt, save_images=opt.save_val_images)

            # TODO: Add early stopping - Maybe not needed for this experiment. In that case log tables before ending
            last = epoch >= opt.n_epochs
            if last:
                train_metrics.update(create_wandb_train_plots(train_metrics=train_metrics))
                val_metrics.update(create_wandb_val_plots(val_metrics=val_metrics, save_images=opt.save_val_images))

            del_wandb_train_untracked_metrics(train_metrics=train_metrics)
            del_wandb_val_untracked_metrics(val_metrics=val_metrics)

            wandb.log(train_metrics)
            wandb.log(val_metrics)

            if val_metrics['val_f1_macro'] > best_model_f1_macro:
                print(f"Saving model with new best {val_metrics['val_f1_macro']=}")
                best_model_f1_macro, best_epoch = val_metrics['val_f1_macro'], epoch
                Path(f'experiments/{opt.group}').mkdir(exist_ok=True)
                new_best_path = os.path.join(f'experiments/{opt.group}',
                                             f'train-{opt.group}-{model_id}-max_epochs{opt.n_epochs}-epoch{epoch}'
                                             f'-metric{val_metrics["val_f1_macro"]:.4f}.pt')
                torch.save(model.state_dict(), new_best_path)
                if best_model_path:
                    os.remove(best_model_path)
                best_model_path = new_best_path

            if last:
                print(
                    f"Finished training a {model_id=}"
                    f"with va_f1_macro {val_metrics['val_f1_macro']}")
                break

        if opt.save_model_wandb:
            artifact.add_file(best_model_path)
            wb_run_train.log_artifact(artifact)

        wb_run_train.finish()

    except FileNotFoundError as e:
        wb_run_train.finish()
        # wb_run_train.delete()  # Delete train run if an error has occurred
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

    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)
    # TODO: Load model from wandb for the current run
    model.load_state_dict(torch.load(best_model_path))
    model.to(opt.device)
    try:
        # TODO: Create a new helper function that returns the number of model parameters
        # TODO: Also save number of parameters for the model in the wandb config
        # TODO: Save a model architecture that was used. This should include the layer information,
        #  similarly to how torch returns the architecture.
        #  Maybe even as an image if it can be visualized with some library
        # pytorch_total_params = sum(p.numel() for p in model.parameters())
        # print(pytorch_total_params)
        eval_metrics = validation(model=model, data_loader=data_loader, opt=opt, save_images=opt.save_test_images)
        eval_metrics.update(create_wandb_val_plots(val_metrics=eval_metrics, save_images=opt.save_test_images))
        del_wandb_val_untracked_metrics(val_metrics=eval_metrics)
        wandb.log(eval_metrics)
        wb_run_eval.finish()
    except FileNotFoundError as e:
        wb_run_eval.finish()
        # wb_run_eval.delete()  # Delete eval run if an error has occurred
        # wb_run_train.delete()  # Delete train run also if an error has occurred
        print(f"Exception happened for model {model_id}\n {e}")
        return [model_id, *args], {
            **kwargs}, True  # Run Failed is True
    # TODO: Add our own code for removing models from the wandb folder during training.
    return [model_id, *args], {**kwargs}, False  # Run Failed is False


def create_wandb_train_plots(train_metrics):
    return {
        "train_confusion_matrix": wandb.plot.confusion_matrix(probs=train_metrics['train_global_probs'],
                                                              y_true=train_metrics['train_global_target'],
                                                              class_names=['ellipse', 'square', 'triangle'],
                                                              title="Train confusion matrix"),
        "train_roc": wandb.plot.roc_curve(y_true=train_metrics['train_global_target'],
                                          y_probas=train_metrics['train_global_probs'],
                                          labels=['ellipse', 'square', 'triangle'],
                                          # TODO: Determine why classes_to_plot doesn't work with roc
                                          # classes_to_plot=['ellipse', 'square', 'triangle'],
                                          title="Train ROC", ),
    }


def create_wandb_val_plots(val_metrics, save_images=False):
    val_mistakes_data = [[val_metrics["val_incorrect_img_paths"][i], val_metrics["val_diff_mistakes"][i],
                          val_metrics["val_shapes_mistakes"][i],
                          wandb.Image(data_or_path=val_metrics["val_incorrect_images"][i],
                                      caption=val_metrics["val_incorrect_img_paths"][i]) if save_images else None,
                          val_metrics["val_incorrect_img_predictions"][i],
                          val_metrics["val_incorrect_img_labels"][i]] for i in
                         range(len(val_metrics["val_incorrect_img_paths"]))]
    return {

        "val_confusion_matrix": wandb.plot.confusion_matrix(probs=val_metrics["val_global_probs"],
                                                            y_true=val_metrics["val_global_target"],
                                                            class_names=['ellipse', 'square', 'triangle'],
                                                            title="Validation confusion matrix"),
        "val_roc": wandb.plot.roc_curve(y_true=val_metrics["val_global_target"],
                                        y_probas=val_metrics["val_global_probs"],
                                        labels=['ellipse', 'square', 'triangle'],
                                        # classes_to_plot=['ellipse', 'square', 'triangle'],
                                        title="Validation ROC", ),
        "val_mistakes_by_diff_bar": wandb.plot.bar(
            table=wandb.Table(
                data=np.asarray([[d, val_metrics["val_diff_mistakes"].count(d)] for d in range(1, 5)]),
                columns=["difficulty", "mistakes"]),
            value="mistakes", label="difficulty", title="Mistakes by difficulty"),
        "val_mistakes_by_shape_bar": wandb.plot.bar(
            table=wandb.Table(data=np.asarray([[d, val_metrics["val_shapes_mistakes"].count(d)] for d in
                                               set(val_metrics["val_shapes_mistakes"])]),
                              columns=["shapes", "mistakes"]),
            value="mistakes", label="shapes", title="Mistakes by shape"),
        "val_mistakes_by_shape_diff_bar": wandb.plot.bar(
            table=wandb.Table(
                data=np.asarray([[d, val_metrics["val_shape_diff_mistakes"].count(d)] for d in
                                 set(val_metrics["val_shape_diff_mistakes"])]),
                columns=["shape_and_difficulty", "mistakes"]),
            value="mistakes", label="shape_and_difficulty", title="Mistakes by shape and difficulty"),
        "val_mistakes_table": wandb.Table(data=val_mistakes_data,
                                          columns=['path', 'difficulty', 'shape', 'image', 'prediction',
                                                   'label']),
    }


def del_wandb_train_untracked_metrics(train_metrics):
    del train_metrics["train_global_probs"]
    del train_metrics["train_global_target"]


def del_wandb_val_untracked_metrics(val_metrics):
    val_metrics.pop("val_global_probs", None)
    val_metrics.pop("val_global_target", None)
    val_metrics.pop("val_diff_mistakes", None)
    val_metrics.pop("val_shapes_mistakes", None)
    val_metrics.pop("val_shape_diff_mistakes", None)
    val_metrics.pop("val_incorrect_img_paths", None)
    val_metrics.pop("val_incorrect_images", None)
    val_metrics.pop("val_incorrect_img_predictions", None)
    val_metrics.pop("val_incorrect_img_labels", None)


def main():
    create_experiments()


if __name__ == "__main__":
    main()
