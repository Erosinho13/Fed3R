import glob
import os.path
import pickle
import sys
from collections import OrderedDict
import copy
import gc
import pandas as pd
import pickle as pkl
import random
import time
from datetime import timedelta
import tensorflow as tf
import torch
import wandb

from datasets import landmarks_utils, inat_utils
from datasets.inaturalist import InatClientDataloader
from datasets.landmarks import GLDv2ClientDataloader
from datasets.tf_utils import read_original_client_sizes_csv

from datasets.utils import get_federated_dataloader_from_args
from fed3r import one_loader_accuracy
from models import get_model_from_args
from pfl.metrics import summarize_client_metrics, compute_metrics_for_client
from pfl.utils import make_pfl_train_yml_parser, update_arch_params_from_arch_size, \
    setup_personalized_optimizer_from_args, get_device_from_arg, tf_hide_other_gpus


def log_wandb(train_metrics_summary, test_metrics_summary):
    print(f"Logging train metrics on wandb...")
    for step in range(len(train_metrics_summary[train_metrics_summary.columns[0]])):
        for column in train_metrics_summary.columns:
            wandb.log({
                f"train_{column}": train_metrics_summary[column][step]},
                step=step
            )
            wandb.log({
                f"test_{column}": test_metrics_summary[column][step]},
                step=step
            )
    print("Done.")


def main():
    config_path = sys.argv[1]
    config_name = config_path.split('/')[-1].split('.')[0]
    args = make_pfl_train_yml_parser(config_path)
    args.wandb_mode = getattr(args, 'wandb_mode', "online")
    args.delete_models = getattr(args, 'delete_models', False)
    args.oll = getattr(args, 'oll', False)
    args.load_original_test_set = getattr(args, 'load_original_test_set', False)

    update_arch_params_from_arch_size(args)
    print('Args', '-' * 50, '\n', args, '\n', '-' * 50)
    torch.manual_seed(args.seed + 25)
    if args.dataset == 'gldv2':  # For TFF dataloaders
        tf.random.set_seed(10)  # for a consistent train-test split
    else:
        tf.random.set_seed(args.seed + 10)
    device = get_device_from_arg(args.device)
    if device == 'cpu':
        print("Warning: forcing device cuda:0")
        device = 'cuda:0'
    tf_hide_other_gpus(args.device)
    print('Using device:', device)

    torch.set_float32_matmul_precision('high')

    global_start_time = time.time()

    # Setup model
    start_time = time.time()
    model = get_model_from_args(args, device).train()
    if args.pretrained_model_path is None:
        raise ValueError('--pretrained_model_path must be specified for the finetuning!')
    try:
        loaded = torch.load(args.pretrained_model_path, map_location=device)
    except FileNotFoundError as e:
        raise e
    model_state_dict = loaded['model_state_dict'] if 'model_state_dict' in loaded else loaded['server_model_state_dict']
    try:
        model.load_state_dict(model_state_dict, strict=False)  # load server params
    except RuntimeError:
        model.classifier = torch.nn.Linear(in_features=1280, out_features=1203)
        model.load_state_dict(model_state_dict)
    model.split_server_and_client_params(args.personalize_on_client)
    saved_client_params = loaded['client_params'] if 'client_params' in loaded else {}
    model.print_summary(args.train_batch_size)
    print(f'Setup model in', timedelta(seconds=round(time.time() - start_time)))
    client_params = list(model.client_parameters())
    server_params = list(model.server_parameters())
    print(
        f"""# Client params = {sum(v.view(-1).shape[0] for v in client_params)} ({len(client_params)} weights/biases)""")
    print(
        f"""# Server params = {sum(v.view(-1).shape[0] for v in server_params)} ({len(server_params)} weights/biases)""")

    classes_by_client = None
    original_classifier = None
    if args.oll:
        if args.dataset == 'gldv2':
            path_to_target_stats = './dataset_statistics/gldv2_classes_by_client.pkl'
        elif args.dataset == 'inaturalist':
            path_to_target_stats = './dataset_statistics/inaturalist_geo_100_classes_by_client.pkl'
        else:
            raise NotImplementedError
        with open(path_to_target_stats, 'rb') as f:
            classes_by_client = pickle.load(f)
        original_classifier = copy.deepcopy(model.classifier)

    # Setup dataloaders
    start_time = time.time()
    train_fed_loader, test_fed_loader = get_federated_dataloader_from_args(args)

    test_loader = None
    if args.load_original_test_set:
        if args.dataset == 'gldv2':
            tf_fed_dataset, test_dataset = landmarks_utils.load_data(cache_dir=args.data_dir)
        else:
            tf_fed_dataset, test_dataset = inat_utils.load_data(cache_dir=args.data_dir, inat_split='geo_100')
        path = (f"dataset_statistics/{args.dataset}_{f'geo_100_' if args.dataset == 'inaturalist' else ''}"
                f"original_client_sizes.csv")
        num_images_per_client = read_original_client_sizes_csv(path)
        if args.dataset == 'gldv2':
            test_loader = GLDv2ClientDataloader(test_dataset, 50, num_images_per_client,
                                                client_id=None, is_train=False)
        else:
            test_loader = InatClientDataloader(test_dataset, 50, num_images_per_client,
                                               client_id=None, is_train=False)

    print('Instantiated dataloaders in', timedelta(seconds=round(time.time() - start_time)))
    print(f'Number of clients: Train = {len(train_fed_loader)}, Test = {len(test_fed_loader)}.')

    # Setup loss and metrics function
    loss_fn, metrics_fn = train_fed_loader.get_loss_and_metrics_fn()

    num_clients = min(len(test_fed_loader), args.max_num_clients_for_personalization)
    rng_clients = random.Random(0)
    list_of_clients_to_finetune = rng_clients.sample(test_fed_loader.available_clients, k=num_clients) \
        if not args.load_original_test_set else test_fed_loader.available_clients[:num_clients]
    print(f'Finetuning for {num_clients} clients')

    per_client_train_sizes = [None] * len(list_of_clients_to_finetune)
    per_client_train_metrics = [None] * len(list_of_clients_to_finetune)
    per_client_test_sizes = [None] * len(list_of_clients_to_finetune)
    per_client_test_metrics = [None] * len(list_of_clients_to_finetune)

    # rng2 = random.Random(args.seed + 5)
    lst_count_clients = list(enumerate(list_of_clients_to_finetune))
    # rng2.shuffle(lst_count_clients)

    os.makedirs(args.savedir, exist_ok=True)
    if args.wandb:
        wandb.login()
        wandb_id_path = os.path.join(args.savedir, 'wandb_id.txt')
        if not os.path.exists(wandb_id_path):
            wandb_id = wandb.util.generate_id()
            with open(wandb_id_path, 'w') as f:
                f.write(wandb_id)
        else:
            with open(wandb_id_path, 'r') as f:
                wandb_id = f.read()

        wandb.init(
            name=sys.argv[1],
            entity="Fed3R",
            project=args.dataset,
            resume="allow",
            id=wandb_id,
            config=args.to_dict(),
            mode=args.wandb_mode
        )

    for cnt, (i, client_id) in enumerate(lst_count_clients):
        gc.collect()
        print(f'\n\n-------\nStarting client {i + 1}/{num_clients}: {client_id} \n--------')
        start_time = time.time()
        client_trainloader = train_fed_loader.get_client_dataloader(client_id)
        client_testloader = test_fed_loader.get_client_dataloader(client_id)
        client_params = saved_client_params[client_id] if len(saved_client_params) != 0 else None
        out = finetune_for_one_client(
            args, model, client_params, client_trainloader, client_testloader, loss_fn, metrics_fn, device, client_id,
            temperature=getattr(args, 'temperature', None), classes_by_client=classes_by_client,
            original_classifier=original_classifier, original_test_loader=test_loader, config_name=config_name
        )
        per_client_train_sizes[i] = out[0]
        per_client_train_metrics[i] = out[1]
        per_client_test_sizes[i] = out[2]
        per_client_test_metrics[i] = out[3]
        print(
            f'\nProcessed client {cnt + 1}/{num_clients} (unshuffled id: {i}) in',
            timedelta(seconds=round(time.time() - start_time)),
            'total time:', timedelta(seconds=round(time.time() - global_start_time)),
        )

    # Summarize metrics before and after personalization
    train_metrics_summary = summarize_personalized_metrics(per_client_train_sizes, per_client_train_metrics)
    test_metrics_summary = summarize_personalized_metrics(per_client_test_sizes, per_client_test_metrics)

    # Save and quit
    train_metrics_summary.to_csv(f'{args.logfilename}_train_finetune.csv')
    test_metrics_summary.to_csv(f'{args.logfilename}_test_finetune.csv')
    with open(f'{args.logfilename}_all_finetune.p', 'wb') as f:
        pkl.dump([per_client_train_sizes, per_client_train_metrics, per_client_test_sizes, per_client_test_metrics], f)
    print(f'Saved: {args.logfilename}_{{train,test}}_finetune.csv and _all_finetune.p')

    # Print
    print('Test metrics summary:')
    print(test_metrics_summary[f'accuracy|mean'])

    if args.wandb:
        log_wandb(train_metrics_summary, test_metrics_summary)

    if args.delete_models:
        prefix = 'model_for_client_'
        files_to_delete = glob.glob(os.path.join(args.savedir, prefix + '*'))
        for file in files_to_delete:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e.strerror}")



# save_dir is args.savedir
def save_client_model(save_dir, client_model, client_id):
    state_dict = client_model.client_state_dict()
    client_fn = f'{save_dir}/model_for_client_{client_id}.pt'
    torch.save(state_dict, client_fn)


def save_stats(save_dir, client_id, train_size, train_metrics, test_size, test_metrics):
    with open(f'{save_dir}/{client_id}_stats.pkl', 'wb') as f:
        pickle.dump({
            'train_size': train_size,
            'train_metrics': train_metrics,
            'test_size': test_size,
            'test_metrics': test_metrics
        }, f)


def load_stats(save_dir, client_id):
    with open(f'{save_dir}/{client_id}_stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    return stats['train_size'], stats['train_metrics'], stats['test_size'], stats['test_metrics']


def finetune_for_one_client(
        args, pretrained_model, client_params, trainloader, testloader, loss_fn, metrics_fn, device, client_id,
        temperature=None, classes_by_client=None, original_classifier=None, original_test_loader=None, config_name=None
):

    def _map_labels(y):
        new_labels = []
        for i in y:
            label = local_classes_mapping[int(i)]
            new_labels.append(label)
        return torch.tensor(new_labels)

    def _log(epoch, metrics_dict, dataloader, is_test):
        is_train = model.training
        model.eval()
        size, metrics = compute_metrics_for_client(model, dataloader, metrics_fn, temperature=temperature,
                                                   map_labels=_map_labels if local_classes is not None else None)
        metrics_dict[epoch] = metrics
        if is_train:
            model.train()
        if is_test:
            print(f'{metrics["loss"]:.2f}\t{metrics["accuracy"]:.4f}\t{optimizer.param_groups[0]["lr"]:.2g}')
        else:
            print(f'{epoch: 2d}\t{metrics["loss"]:.2f}\t{metrics["accuracy"]:.4f}', end='\t\t')
        return size

    if os.path.exists(f'{args.savedir}/{client_id}_stats.pt'):
        print(f"Client {client_id} already finetuned - skipping")
        return load_stats(args.savedir, client_id)

    local_classes = None
    local_classes_mapping = None
    if classes_by_client is not None:
        assert original_classifier is not None
        local_classes = classes_by_client[str(client_id)]
        local_classes_mapping = {int(local_classes[i]): i for i in range(len(local_classes))}

    # copy model (do not modify original one)
    model = copy.deepcopy(pretrained_model).to(device)
    if args.client_var_prox_to_init:
        prox_center = [v.detach() for v in pretrained_model.client_parameters()]  # pretrained model weights
    else:
        prox_center = None
    if client_params is not None and not args.stateless_clients:
        model.load_state_dict(client_params, strict=False)

    if local_classes is not None:
        model.classifier = torch.nn.Linear(in_features=1280, out_features=len(local_classes)).to(device)
        filtered_weights = torch.index_select(original_classifier.state_dict()['weight'].to(device), dim=0,
                                              index=torch.tensor(local_classes).to(device))
        filtered_bias = torch.index_select(original_classifier.state_dict()['bias'], dim=0,
                                           index=torch.tensor(local_classes).to(device))
        model.classifier.load_state_dict({'weight': filtered_weights, 'bias': filtered_bias})

    # Train only client params and not server params
    model.client_params_requires_grad_(True)
    model.server_params_requires_grad_(False)
    # Init other parameters
    max_num_updates = min(len(trainloader) * args.num_epochs_personalization, args.max_num_finetune_updates)
    optimizer, scheduler = setup_personalized_optimizer_from_args(args, model, max_num_updates)

    print('Epoch|Train Loss|Train Acc.|Test Loss|Test Acc.|LR')

    train_metrics = OrderedDict()
    test_metrics = OrderedDict()
    train_size = _log(0, train_metrics, trainloader, is_test=False)
    test_size = _log(0, test_metrics, testloader, is_test=True)

    num_updates = 0

    if original_test_loader is not None:
        print("Computing original classwise test accuracy...")
        one_loader_accuracy(model, original_test_loader, device, classwise_accuracy=True,
                            num_classes=2028 if args.dataset == 'gldv2' else 1203,
                            csv_filename=f"fp_vs_oll_logs/{config_name}_{client_id}.csv", epoch=0,
                            local_classes=local_classes)
        print("Done.")

    for epoch in range(1000):  # maximum number of epochs on local data

        if num_updates >= max_num_updates:  # done personalization
            break

        for x, y in trainloader:

            if local_classes is not None:
                y = _map_labels(y)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            yhat = model(x)
            if temperature is not None:
                yhat = yhat / temperature
            loss = loss_fn(yhat, y) + get_finetune_l2_penalty(args, model, prox_center)
            loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            num_updates += 1
            if num_updates >= max_num_updates:  # jump directly to logging
                continue

        if original_test_loader is None:
            _log(epoch + 1, train_metrics, trainloader, is_test=False)
            _log(epoch + 1, test_metrics, testloader, is_test=True)
        elif (epoch + 1) % 1 == 0:
            print("Computing original classwise test accuracy...")
            one_loader_accuracy(model, original_test_loader, device, classwise_accuracy=True,
                                num_classes=2028 if args.dataset == 'gldv2' else 1203,
                                csv_filename=f"fp_vs_oll_logs/{config_name}_{client_id}.csv", epoch=epoch + 1,
                                local_classes=local_classes)
            print("Done.")

    # save_client_model(args.savedir, model, client_id)
    save_stats(args.savedir, client_id,
               train_size, pd.DataFrame(train_metrics).T,
               test_size, pd.DataFrame(test_metrics).T)

    # access metric value using metrics_df.at[epoch, metric_name]
    return train_size, pd.DataFrame(train_metrics).T, test_size, pd.DataFrame(test_metrics).T


def get_finetune_l2_penalty(args, model, prox_center):
    l2reg = args.client_var_l2_reg_coef
    if l2reg <= 1e-10:
        return 0.0
    elif prox_center is None:  # plain l2 norm
        client_params = model.client_parameters()
        return l2reg * sum(torch.norm(v.reshape(-1)) ** 2 for v in client_params)
    else:  # l2 norm difference to prox center
        client_params = model.client_parameters()
        return l2reg * sum(
            torch.norm(v.reshape(-1) - v1.reshape(-1)) ** 2 for (v, v1) in zip(client_params, prox_center))


def summarize_personalized_metrics(sizes_lst, metrics_lst):
    # The following two lines have been added only for debugging
    sizes_lst = [x for x in sizes_lst if x is not None]
    metrics_lst = [x for x in metrics_lst if x is not None]
    # metrics_lst[i]: DataFrame with personalization logs of client i
    keys = metrics_lst[0].columns.to_list()
    # Summarize metrics
    all_metrics = []
    for i in range(len(metrics_lst[0])):  # iterate over the epochs
        metrics_lst_dict = [m.iloc[i].to_dict() for m in metrics_lst]
        metrics = OrderedDict([(key, [m[key] for m in metrics_lst_dict]) for key in keys])
        metrics = summarize_client_metrics(sizes_lst, metrics)  # OrderedDict
        all_metrics.append(metrics)
    # access with df.at['pretrained', f'{metric_name}|{statistic}']
    return pd.DataFrame({i: metrics for i, metrics in enumerate(all_metrics)}).T


if __name__ == '__main__':
    main()
