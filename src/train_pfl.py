import gc
import subprocess

import numpy as np
import os
import pandas as pd
import pickle as pkl
import sys
import time
from datetime import timedelta
import tensorflow as tf
import torch
import wandb
from types import SimpleNamespace

from datasets.utils import get_federated_dataloader_from_args
from fed3r import test_model
from models import get_model_from_args
from pfl.optim import FedAvg, FedSim, FedAlt, PFedMe
from pfl.utils import update_arch_params_from_arch_size, get_device_from_arg, tf_hide_other_gpus, \
                      get_fed_global_lr_scheduler, make_pfl_train_yml_parser


def main():
    config_path = sys.argv[1]
    args = make_pfl_train_yml_parser(config_path)
    args.wandb_mode = getattr(args, 'wandb_mode', "online")
    args.use_fedrdn = getattr(args, 'use_fedrdn', False)
    args.boost_domain = getattr(args, 'boost_domain', 0.0)
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory {args.data_dir} does not exists.")
    update_arch_params_from_arch_size(args)
    print('Args', '-' * 50, '\n', args, '\n', '-' * 50)
    torch.manual_seed(args.seed + 5)
    if args.dataset == 'gldv2' or args.dataset == 'inaturalist':
        tf.random.set_seed(10)  # for a consistent train-test split for this dataset.
    else:
        tf.random.set_seed(args.seed + 10)  # for TFF dataloaders
    device = get_device_from_arg(args.device)
    if device == 'cpu':
        print("Warning: forcing device cuda:0")
        device = 'cuda:0'
    tf_hide_other_gpus(args.device)
    print('Using device:', device)

    torch.set_float32_matmul_precision('high')

    global_start_time = time.time()

    # Setup server model
    start_time = time.time()
    if args.pfl_algo == 'fedavg' and args.personalize_on_client is not None:
        raise ValueError('FedAvg requires personalize_on_client = "none"')
    server_model = get_model_from_args(args, device).train()
    if args.pretrained_model_path is not None:
        print('Loading pretrained model from', args.pretrained_model_path)
        loaded = torch.load(args.pretrained_model_path, map_location=device)
        state_dict = loaded['model_state_dict'] if 'model_state_dict' in loaded else loaded['server_model_state_dict']
        try:
            server_model.load_state_dict(state_dict)
        except RuntimeError:
            server_model.classifier = torch.nn.Linear(in_features=1280,
                                                      out_features=state_dict['classifier.weight'].size()[0])
            server_model.load_state_dict(state_dict)
    server_model.print_summary(args.train_batch_size)
    server_model.split_server_and_client_params(args.personalize_on_client)
    print(f'Setup model in', timedelta(seconds=round(time.time() - start_time)))

    torch.compile(server_model)

    # Setup dataloaders
    start_time = time.time()
    train_fed_loader, test_fed_loader = get_federated_dataloader_from_args(args)

    print('Instantiated dataloaders in', timedelta(seconds=round(time.time() - start_time)))
    print(f'Number of clients: Train = {len(train_fed_loader)}, Test = {len(test_fed_loader)}.')

    # Setup PFL optimizer
    client_optimizer_args = SimpleNamespace(
        client_lr=args.client_lr,
        client_momentum=0,
        scheduler=args.client_scheduler,
        lr_decay_factor=args.client_lr_decay_factor,
        lr_decay_every=args.client_lr_decay_every,
        warmup_fraction=args.local_warmup_fraction,
        weight_decay=float(getattr(args, 'client_weight_decay', 0.0))
    )
    global_lr_args = SimpleNamespace(
        scheduler=args.global_scheduler,
        lr_decay_factor=args.global_lr_decay_factor,
        lr_decay_every=args.global_lr_decay_every,
        warmup_fraction=args.global_warmup_fraction
    )
    global_lr_fn = get_fed_global_lr_scheduler(args.num_communication_rounds, global_lr_args)
    available_clients = train_fed_loader.available_clients
    if args.boost_domain == 0.0:
        clients_to_cache = test_fed_loader.available_clients  # cache the local models for test clients to test
    else:
        clients_to_cache = None
    print('Number of training clients for federated training:', len(available_clients))
    pfl_args = dict(
        train_fed_loader=train_fed_loader,
        available_clients=available_clients,
        clients_to_cache=clients_to_cache,
        server_model=server_model,
        server_optimizer=args.server_optimizer,
        server_lr=args.server_lr,
        server_momentum=args.server_momentum,
        max_grad_norm=args.max_grad_norm,
        clip_grad_norm=args.clip_grad_norm,
        save_dir=args.savedir,
        seed=args.seed,
        save_client_params_to_disk=args.save_client_params_to_disk,
        stateless_clients=args.stateless_clients,
        client_var_l2_reg_coef=args.client_var_l2_reg_coef,
        client_var_prox_to_init=args.client_var_prox_to_init,
        max_num_pfl_updates=args.max_num_pfl_updates,
        args=args
    )
    if args.pfl_algo.lower() == 'pfedme':
        pfl_args['pfedme_reg_param'] = args.pfedme_l2_reg_coef
    pfl_optim = get_pfl_optimizer(args.pfl_algo, **pfl_args)
    del server_model  # give full ownership of `server_model` to pfl_optim

    # Setup logging
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    checkpoint_filename = f'{args.savedir}/checkpoint.pt'

    def _log_test(comm_round, pfl_optim, log_all_clients=False, post_finetune=False):
        nonlocal log_test
        gc.collect()
        log_start_time = time.time()
        metrics, all_metrics, client_sizes = pfl_optim.test_all_clients(test_fed_loader,
                                                                        args.max_num_clients_for_logging,
                                                                        return_all_metrics=True)
        metrics['round'] = comm_round
        log_test.append(metrics)
        # Save
        pd.DataFrame(log_test).to_csv(f'{args.logfilename}_test.csv')
        if log_all_clients:
            suffix = '_finetune' if post_finetune else ''
            with open(f'{args.logfilename}_test{suffix}_all.p', 'wb') as f:
                pkl.dump([all_metrics, client_sizes], f)
        # Print
        print('-' * 100)
        current_time = time.time() - log_start_time
        print('| checkpoint | round {:d} | time: {:5.2f}s | test loss {:5.2f} | '
              'test accuracy {:.4f}'.format(
            comm_round, current_time, metrics.get('loss|mean', -1), metrics.get('accuracy|mean', -1)
        ))
        print('-' * 100)
        if args.wandb:
            wandb.log({
                'test time': current_time,
                'loss|mean': metrics.get('loss|mean', -1),
                'accuracy|mean': metrics.get('accuracy|mean', -1)},
                step=comm_round)
        if comm_round >= 200 and 'accuracy|mean' in metrics and 0 <= metrics['accuracy|mean'] < 0.0005:
            print('Exiting since accuracy is very small.')
            sys.exit(-1)

    def _log_train(comm_round, avg_loss):
        nonlocal log_train, start_time
        model_norm = np.linalg.norm([torch.norm(v.view(-1)).item() for v in pfl_optim.server_model.parameters()])
        d = dict(round=comm_round, avg_loss=avg_loss, client_lr=client_optimizer_args.client_lr, model_norm=model_norm)
        log_train.append(d)
        pd.DataFrame(log_train).to_csv(f'{args.logfilename}_train.csv')
        current_time = timedelta(seconds=round(time.time() - start_time))
        global_time = timedelta(seconds=round(time.time() - global_start_time))
        print(f'round: {comm_round:d}, loss: {avg_loss:.4f}, norm: {model_norm:.4g},',
              f'client_lr: {client_optimizer_args.client_lr}',
              f'time: {current_time},',
              f'global time: {global_time}')
        if args.wandb:
            wandb.log({
                "loss": avg_loss,
                "norm": model_norm,
                "client_lr": client_optimizer_args.client_lr,
                "current time": current_time.seconds,
                "global time": global_time.seconds,
            }, step=comm_round)
        start_time = time.time()

    starting_round, avg_loss, log_train, log_test, wandb_id = try_restore_checkpoint_(
        checkpoint_filename, args.logfilename, pfl_optim, args.force_restart, device
    )

    if args.wandb:
        wandb.login()
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        wandb.init(
            name=sys.argv[1],
            entity="Fed3R",
            project=args.dataset,
            resume="allow",
            id=wandb_id,
            config=args.to_dict(),
            mode=args.wandb_mode
        )

    if not args.skip_first_log and avg_loss is None and args.boost_domain == 0.0:  # no checkpoint found
        _log_test(starting_round, pfl_optim)

    start_time = time.time()

    # Main training loop
    prev_ckpt_time = time.time()
    for comm_round in range(starting_round, args.num_communication_rounds):
        # Adjust LR
        cur_client_lr = global_lr_fn(comm_round) * args.client_lr
        client_optimizer_args.client_lr = cur_client_lr

        # Run local updates
        loss_per_round = pfl_optim.run_one_round(
            args.num_clients_per_round, args.num_local_epochs, args.client_optimizer, client_optimizer_args
        )
        if np.isnan(loss_per_round):
            print(f"""NaN encountered in round {comm_round}. Prev avg_loss = {avg_loss}. Exiting.""")
            sys.exit(-1)
        avg_loss = 0.9 * avg_loss + 0.1 * loss_per_round if avg_loss is not None else loss_per_round
        # logging
        if (comm_round + 1) % args.log_train_every_n_rounds == 0:
            _log_train(comm_round, avg_loss)
        if (comm_round + 1) % args.log_test_every_n_rounds == 0:
            save_checkpoint(comm_round, avg_loss, pfl_optim, checkpoint_filename, wandb_id=wandb_id)
            prev_ckpt_time = time.time()
            if comm_round + 1 != args.num_communication_rounds:
                if args.boost_domain == 0.0:
                    _log_test(comm_round, pfl_optim)
                else:
                    test_model(pfl_optim.combined_model, test_fed_loader, device, pfl_mode=False, log_wandb=args.wandb,
                               round_=comm_round)
        if time.time() - prev_ckpt_time >= 3600:  # save checkpoint every hour
            save_checkpoint(comm_round, avg_loss, pfl_optim, checkpoint_filename, wandb_id=wandb_id)
            prev_ckpt_time = time.time()

    if args.boost_domain == 0.0:
        _log_test(args.num_communication_rounds, pfl_optim, log_all_clients=True, post_finetune=False)
    else:
        test_model(pfl_optim.combined_model, test_fed_loader, device, pfl_mode=False, log_wandb=args.wandb,
                   round_=args.num_communication_rounds)
    save_checkpoint(args.num_communication_rounds, avg_loss, pfl_optim, checkpoint_filename, wandb_id=wandb_id)

    print('Saved:', f'{args.logfilename}_test.csv')
    print('Total running time:', timedelta(seconds=round(time.time() - global_start_time)))


def get_pfl_optimizer(pfl_algo, **kwargs):
    if pfl_algo.lower() == 'fedavg':
        return FedAvg(**kwargs)
    elif pfl_algo.lower() in ['fedsim', 'pfl_joint', 'pfl_simultaneous']:
        return FedSim(**kwargs)
    elif pfl_algo.lower() in ['fedalt', 'pfl_alternating', 'pfl_am']:
        return FedAlt(**kwargs)
    elif pfl_algo.lower() == 'pfedme':
        return PFedMe(**kwargs)
    else:
        raise ValueError(f'Unknown PFL algorithm: {pfl_algo}')


def save_checkpoint(comm_round, avg_training_loss, pfl_optim, checkpoint_filename, wandb_id=None):
    # Model save.
    to_save = dict(
        round=comm_round,
        avg_training_loss=avg_training_loss,
        server_model_state_dict=pfl_optim.server_model.state_dict(),
        server_optimizer_state_dict=pfl_optim.server_optimizer.state_dict(),
        client_params=pfl_optim.get_client_params_for_logging(),
        torch_rng_state=torch.random.seed(),
        pfl_optim_rng_state=pfl_optim.rng.getstate(),
        wandb_id=wandb_id
    )
    torch.save(to_save, checkpoint_filename)


def try_restore_checkpoint_(checkpoint_filename, log_filename, pfl_optim, force_restart, device):
    if os.path.isfile(checkpoint_filename) and not force_restart:
        # load checkpoint. Keep cpu weights on cpu but move GPU weights to our device
        print(f'Found checkpoint: {checkpoint_filename}. Restoring state.')
        start_time = time.time()
        map_location = {f'cuda:{i}': str(device) for i in range(8)}
        map_location['cpu'] = 'cpu'
        saved_data = torch.load(checkpoint_filename, map_location=map_location)
        # round and loss
        starting_round = saved_data['round'] + 1  # start from the next round
        avg_loss = saved_data['avg_training_loss']
        # server model/optimizer and client weights
        pfl_optim.server_model.load_state_dict(saved_data['server_model_state_dict'])
        pfl_optim.server_optimizer.load_state_dict(saved_data['server_optimizer_state_dict'])
        pfl_optim.load_client_params_from_checkpoint(saved_data['client_params'])
        # seeds
        torch.random.manual_seed(saved_data['torch_rng_state'])
        pfl_optim.rng.setstate(saved_data['pfl_optim_rng_state'])
        # logs
        log_test = _load_log_from_csv(f'{log_filename}_test.csv')
        max_round = saved_data['round']  # max([d['round'] for d in log_test])
        log_train = _load_log_from_csv(f'{log_filename}_train.csv')
        log_train = [d for d in log_train if d['round'] <= max_round]  # discard logs for rounds which will be rerun
        wandb_id = saved_data['wandb_id']
        del saved_data
        gc.collect()
        print(f'Loaded checkpoint in', timedelta(seconds=round(time.time() - start_time)))
    else:
        # Start from scratch
        print(f'No checkpoint exists (or --args.force_restart) was specified. Starting from scratch.')
        starting_round = 0
        avg_loss = None
        log_train = []
        log_test = []
        wandb_id = None
    return starting_round, avg_loss, log_train, log_test, wandb_id


def _load_log_from_csv(fn):
    logs = []
    df = pd.read_csv(fn, index_col=0)
    for i in range(df.shape[0]):
        row = df.iloc[i].to_dict()
        logs.append(row)
    return logs


if __name__ == '__main__':
    main()
