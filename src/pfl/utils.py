import os
from datetime import datetime

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# Signal handling
import signal


def handler(signum, frame):
    print('\n', '-' * 50)
    print('DANGER. DANGER. DANGER.')
    print(f"Caught signal {signum} at {datetime.now()}")
    print('This is your premeption notice!')
    print('-' * 50, '\n\n', flush=True)


signal.signal(signal.SIGUSR1, handler)

CPU_DEVICE = torch.device('cpu')


def get_device_from_arg(device_id):
    if (device_id is not None and
            torch.cuda.is_available() and
            0 <= device_id < torch.cuda.device_count()):
        return torch.device(f'cuda:{device_id}')
    else:
        return CPU_DEVICE


def tf_hide_other_gpus(device_id):
    import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices('GPU')
    try:  # Disable unnecessary GPUs
        # tf.config.set_visible_devices([physical_devices[device_id]], 'GPU')
        tf.config.set_visible_devices([], 'GPU')
    except:  # Invalid device or cannot modify virtual devices once initialized.
        pass


class DotDict(dict):
    """A dictionary that supports dot notation."""

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def to_dict(self):
        """Return the dictionary of attributes and their values."""
        return dict(self)


def make_pfl_train_yml_parser(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_name = config_path.split('/')[-1].split('.')[0]
    ckpt_path = './saved_models/'
    if 'ckpt_path' in config.keys():
        ckpt_path = config['ckpt_path']
    if 'savedir' not in config.keys():
        config['savedir'] = os.path.join(ckpt_path, config_name)
    if 'savefilename' not in config.keys():
        config['savefilename'] = os.path.join(ckpt_path, f"{config_name}.pt")
    if 'modelfilename' not in config.keys():
        config['modelfilename'] = os.path.join(ckpt_path, f"{config_name}.pt")
    if 'logfilename' not in config.keys():
        config['logfilename'] = f"./logs/{config_name}"
    args = DotDict(config)
    args.ckpt_path = ckpt_path
    return args


def update_arch_params_from_arch_size(args):
    if args.dataset != 'stackoverflow':
        return
    if args.arch_size == 'tiny':
        args.num_attn_heads = 2
        args.num_transformer_layers = 2
        args.input_dim = 128
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 512
    elif args.arch_size == 'mini':
        args.num_attn_heads = 4
        args.num_transformer_layers = 4
        args.input_dim = 256
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 1024
    elif args.arch_size == 'half':
        args.num_attn_heads = 6
        args.num_transformer_layers = 6
        args.input_dim = 384
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 1536
    elif args.arch_size == 'medium':
        args.num_attn_heads = 8
        args.num_transformer_layers = 8
        args.input_dim = 512
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 2048
    elif args.arch_size == 'base':
        args.num_attn_heads = 12
        args.num_transformer_layers = 12
        args.input_dim = 768
        args.attn_hidden_dim = 64
        args.fc_hidden_dim = 1536
    else:
        raise ValueError(f'Unknown arch size: {args.arch_size}')
    if 0 < args.model_dropout < 1:
        args.dropout_tr = args.model_dropout
        args.dropout_io = args.model_dropout
        print('Using dropout =', args.model_dropout)
    else:
        args.dropout_tr = 0
        args.dropout_io = 0


def setup_centralized_optimizer_from_args(args, model, num_clients_to_process):
    lr = args.lr
    if args.central_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.central_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {args.central_optimizer}')
    # args: scheduler, lr_decay_factor, lr_decay_every, warmup_fraction
    if args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif args.scheduler == 'linear':
        num_warmup_steps = args.warmup_fraction * num_clients_to_process

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0,
                       (num_clients_to_process - current_step) /
                       max(1.0, num_clients_to_process - num_warmup_steps)
                       )
    elif args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, args.lr_decay_factor)) ** (current_step / num_clients_to_process)
    elif args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // args.lr_decay_every
            return args.lr_decay_factor ** factor

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def adjust_optimizer_centralized_(args, optimizer, epoch, num_clients_processed):
    if (args.use_warmup and
            optimizer.param_groups[0]['lr'] == args.warmup_lr and
            num_clients_processed == args.num_warmup_updates
    ):  # warmup completed
        print(f'Warmup completed at epoch: {epoch}, updates: {num_clients_processed}. Using full LR')
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    elif epoch > 1 and epoch % args.lr_decay_every == 0:
        # decay LR
        for g in optimizer.param_groups:
            g['lr'] /= args.lr_decay_factor


def get_fed_global_lr_scheduler(num_communication_rounds, optimizer_args):
    """Get a scheduler for the maximum client learning rate
        optimizer_args: scheduler, warmup_fraction, lr_decay_factor, lr_decay_every
    Returns:
        Callable: current_round -> lr_mulitplier
    """
    # optimizer_args: scheduler, lr_decay_factor, lr_decay_every, warmup_fraction
    if optimizer_args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif optimizer_args.scheduler == 'linear':
        num_warmup_steps = optimizer_args.warmup_fraction * num_communication_rounds

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0,
                       (num_communication_rounds - current_step) /
                       max(1.0, num_communication_rounds - num_warmup_steps)
                       )
    elif optimizer_args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, optimizer_args.lr_decay_factor)) ** (current_step / num_communication_rounds)
    elif optimizer_args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // optimizer_args.lr_decay_every
            return optimizer_args.lr_decay_factor ** factor

    return lr_lambda


def setup_personalized_optimizer_from_args(args, model, num_training_steps):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    # Setup scheduler
    scheduler = None
    if args.scheduler == 'const':
        lr_lambda = lambda current_step: 1.0  # mult. factor = 1.0
    elif args.scheduler == 'linear':
        num_warmup_steps = args.warmup_fraction * num_training_steps

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1.0, num_warmup_steps)
            return max(0.0,
                       (num_training_steps - current_step) /
                       max(1.0, num_training_steps - num_warmup_steps)
                       )
    elif args.scheduler == 'expo':
        def lr_lambda(current_step):
            return min(1.0, max(0.0, args.lr_decay_factor)) ** (current_step / num_training_steps)
    elif args.scheduler == 'const_and_cut':
        def lr_lambda(current_step):
            factor = current_step // args.lr_decay_every
            return args.lr_decay_factor ** factor

    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)

    if scheduler is None:
        scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
