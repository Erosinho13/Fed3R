import os
import pickle
import random
import sys
import wandb
import csv

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import yaml
from tqdm import tqdm

from datasets import landmarks_utils, inat_utils
from datasets.inaturalist import InatClientDataloader, PersInatClientDataloader, PersInatFederatedDataloader
from datasets.landmarks import GLDv2ClientDataloader, PersGLDv2ClientDataloader, PersGLDv2FederatedDataloader
from datasets.tf_utils import read_original_client_sizes_csv
from models.mobilenetv2 import mobilenetv2, save_classifier
from pfl.utils import DotDict


def calc_ridge_params(A, b):
    W = torch.linalg.solve(A, b)
    bias = W[-1, :]
    W = W[:-1, :]
    return W, bias


def normalize_W(W, bias, warnings=True):
    norm = torch.norm(W, dim=0, keepdim=True)
    if torch.any(norm == 0.0):
        if warnings:
            print("WARNING: 0 encountered in norm, substituting with 1e-6")
        norm[norm == 0.0] = 1e-6
    W = W / norm
    bias = bias / norm[0, :]
    return W, bias


def fed3r_step(A, b, dataloader, model, num_classes, device):

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = data.to(device), target.to(device)
        try:
            features, _ = model(data, return_feats=True)
        except:
            features = model(data)
        feats_with_bias = torch.cat((features, torch.ones((features.shape[0], 1), dtype=torch.float32).to(device)),
                                    dim=1)
        feats_with_bias = feats_with_bias.detach()
        feats_with_bias.requires_grad = False
        one_hot_encoded = torch.nn.functional.one_hot(target, num_classes=num_classes)
        one_hot_encoded = one_hot_encoded.float()
        one_hot_encoded.requires_grad = False
        Ak = feats_with_bias.T @ feats_with_bias
        bk = feats_with_bias.T @ one_hot_encoded
        A += Ak
        b += bk

    return A, b


class ClasswiseAccuracy:
    def __init__(self, num_classes, csv_filename="classwise_accuracies.csv"):
        self.num_classes = num_classes
        self.correct_per_class = torch.zeros(num_classes)
        self.total_per_class = torch.zeros(num_classes)
        self.csv_filename = csv_filename

        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.csv_filename):
            columns = ["Round"] + [f"Class_{i}_Acc" for i in range(num_classes)]
            pd.DataFrame(columns=columns).to_csv(self.csv_filename, index=False)

    def update(self, predictions, labels):
        """Update accuracy counts with a new batch of predictions and labels."""
        for cls in range(self.num_classes):
            mask = labels == cls
            self.correct_per_class[cls] += (predictions[mask] == cls).sum().item()
            self.total_per_class[cls] += mask.sum().item()

    def compute(self):
        """Compute class-wise accuracy and return as a list."""
        return (self.correct_per_class / self.total_per_class.clamp(min=1)).tolist()

    def log_to_csv(self, epoch_number):
        """Log current class-wise accuracies to a CSV file."""
        accuracies = self.compute()
        data = {"Epoch": epoch_number, **{f"Class_{i}_Acc": acc for i, acc in enumerate(accuracies)}}
        df = pd.DataFrame([data])
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)

    def reset(self):
        """Reset counters for the next round."""
        self.correct_per_class.zero_()
        self.total_per_class.zero_()


def one_loader_accuracy(model, dataloader, device, return_size=False, local_classes=None, unknown_class=False,
                        filter_predictions=False, num_classes=None, classwise_accuracy=False,
                        csv_filename=None, epoch=-1):

    total = 0
    correct = 0
    size = 0

    local_classes_mapping = None
    if local_classes is not None:
        local_classes_mapping = {int(local_classes[i]): i for i in range(len(local_classes))}

    cols_to_zero = None
    if filter_predictions:
        cols_to_zero = torch.tensor([col for col in range(num_classes) if col not in local_classes])
    distinct_labels = set()

    cls_acc = None
    if classwise_accuracy:
        assert csv_filename is not None
        assert epoch >= 0
        cls_acc = ClasswiseAccuracy(num_classes, csv_filename=csv_filename)

    if classwise_accuracy:
        dataloader = tqdm(dataloader)
        if local_classes is not None:
            data = {f"Class_{i}": j for i, j in enumerate(local_classes)}
            df = pd.DataFrame([data])
            df.to_csv(f"{csv_filename.split('.')[0]}_local_classes.csv", mode='w', header=False, index=False)
            local_classes = torch.tensor(local_classes).cuda()

    for images, labels in dataloader:

        images, labels = images.cuda(), labels.cuda()

        if local_classes is not None and not filter_predictions and not classwise_accuracy:
            new_labels = []
            for i in labels:
                distinct_labels.add(int(i))
                if int(i) in local_classes_mapping:
                    label = local_classes_mapping[int(i)]
                elif unknown_class:
                    label = len(local_classes)
                else:
                    raise IndexError
                new_labels.append(label)
            labels = torch.tensor(new_labels)

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if filter_predictions:
            outputs[:, cols_to_zero] = 0
        _, predictions = torch.max(outputs.data, 1)
        if classwise_accuracy and local_classes is not None:
            predictions = local_classes[predictions]
        total += labels.size(0)
        if classwise_accuracy:
            cls_acc.update(predictions, labels)

        correct += (predictions == labels).sum().item()
        if return_size:
            size += len(labels)

    if classwise_accuracy:
        cls_acc.log_to_csv(epoch_number=epoch)
        cls_acc.reset()
    acc = 100 * correct / total

    if return_size:
        return acc, size

    return acc


def test_model(model, dataloader, device, pfl_mode=False, log_wandb=False, round_=-1):

    with torch.no_grad():

        model.eval()

        if pfl_mode:

            accs = {}
            sizes = {}
            for client_id, client_dataloader in tqdm(dataloader.items()):
                acc, size = one_loader_accuracy(model, client_dataloader, device, return_size=True)
                sizes[client_id] = size
                accs[client_id] = acc

            sum_acc = sum(accs[client_id] * sizes[client_id] for client_id in accs)
            sum_samples = sum(sizes.values())
            acc_values = np.array(list(accs.values()))

            argmax = max(accs, key=accs.get)
            argmin = min(accs, key=accs.get)
            weighted_avg_acc = sum_acc / sum_samples
            avg_acc = acc_values.mean()
            std_acc = acc_values.std()
            max_acc = accs[argmax]
            min_acc = accs[argmin]
            q1_acc = np.percentile(acc_values, 25)
            q2_acc = np.percentile(acc_values, 50)  # this is the median
            q3_acc = np.percentile(acc_values, 75)

            print(f"Weighted avg acc: {round(weighted_avg_acc, 2)}%")
            print(f"Avg acc: {round(avg_acc, 2)}%")
            print(f"Std acc: {round(std_acc, 2)}%")
            print(f"Max acc: {round(max_acc, 2)}%, client {argmax}")
            print(f"Min acc: {round(min_acc, 2)}%, client {argmin}")
            print(f"Q1 acc: {round(q1_acc, 2)}%")
            print(f"Q2 acc: {round(q2_acc, 2)}%")
            print(f"Q3 acc: {round(q3_acc, 2)}%")

            if log_wandb:
                wandb.log({
                    'weighted avg acc': weighted_avg_acc,
                    'avg acc': avg_acc,
                    'std acc': std_acc,
                    'max acc': max_acc,
                    'min acc': min_acc,
                    'client max acc': int(argmax),
                    'client min acc': int(argmin),
                    'q1 acc': q1_acc,
                    'q2 acc': q2_acc,
                    'q3 acc': q3_acc
                }, step=round_)

        else:
            acc = one_loader_accuracy(model, dataloader, device)
            print(f"Accuracy on the test set: {round(acc, 2)}%")
            if log_wandb:
                wandb.log({
                    'acc': acc,
                }, step=round_)


def get_train_test_loaders(ds_root, dataset_name, pfl_mode, shuffle=True, only_train=False, inat_split='user_120k'):

    mean_clients, std_clients = {}, {}
    if dataset_name == 'gldv2' or 'inaturalist':

        if dataset_name == 'gldv2':
            tf_fed_dataset, test_dataset = landmarks_utils.load_data(cache_dir=ds_root)
        else:
            tf_fed_dataset, test_dataset = inat_utils.load_data(cache_dir=ds_root, inat_split=inat_split)

        if pfl_mode:
            ds_and_split_name = f"{dataset_name}{f'_{inat_split}' if dataset_name == 'inaturalist' else ''}"
            num_images_per_client = pd.read_csv(f"dataset_statistics/{ds_and_split_name}_client_sizes_train.csv",
                                                index_col=0, dtype='string').squeeze().to_dict()
            num_images_per_client = {k: int(v) for (k, v) in num_images_per_client.items()}
            clients_list = (pd.read_csv(f'dataset_statistics/{ds_and_split_name}_client_ids_train.csv', dtype=str)
                            .to_numpy().reshape(-1).tolist())
            num_clients = len(clients_list)
            PersClientDataloader = PersGLDv2ClientDataloader if dataset_name == 'gldv2' else PersInatClientDataloader
            PersFederatedDataloader = PersGLDv2FederatedDataloader if dataset_name == 'gldv2' else PersInatFederatedDataloader

            def train_loaders(client_id, idx=None):
                return PersClientDataloader(
                    tf_fed_dataset.create_tf_dataset_for_client(client_id),
                    50, num_images_per_client[client_id], int(client_id),
                    1000, True, False, False, force_test_augmentation=True, split_name=inat_split,
                    idx=idx, mean_clients=mean_clients, std_clients=std_clients
                )

            if only_train:
                return num_images_per_client, num_clients, clients_list, train_loaders

            full_test_loader = PersFederatedDataloader(
                ds_root, f'dataset_statistics/{ds_and_split_name}_client_ids_test.csv', 'test',
                50, 1000,
                validation_mode=False, validation_holdout=True, split_name=inat_split
            )

            print("Loading test dataloaders...")
            test_loader = {}
            for client_id in tqdm(clients_list):
                test_loader[client_id] = list(full_test_loader.get_client_dataloader(client_id))
            print("Done.")

        else:

            clients_list = [e for e in tf_fed_dataset.client_ids if e != 'missing']
            num_clients = len(clients_list)
            path = (f"dataset_statistics/{dataset_name}_{f'{inat_split}_' if dataset_name == 'inaturalist' else ''}"
                    f"original_client_sizes.csv")
            if os.path.exists(path):
                num_images_per_client = read_original_client_sizes_csv(path)
            else:
                num_images_per_client = {}
                for k in tqdm(clients_list):
                    num_images_per_client[k] = len(list(tf_fed_dataset._data_per_client[k].as_numpy_iterator()))
                with open(path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["client", "num_images"])
                    for client_id, train_size in num_images_per_client.items():
                        writer.writerow([client_id, train_size])
                print("CSV written! Remember to add the test key")
                quit()

            if dataset_name == 'gldv2':
                client_dataloader_class = GLDv2ClientDataloader
            else:
                client_dataloader_class = InatClientDataloader

            def train_loaders(client_id):
                return client_dataloader_class(tf_fed_dataset, 50, num_images_per_client,
                                               client_id=str(client_id), is_train=False)

            if only_train:
                return num_images_per_client, num_clients, clients_list, train_loaders

            test_loader = client_dataloader_class(test_dataset, 50, num_images_per_client,
                                                  client_id=None, is_train=False)

    else:
        raise NotImplementedError

    if shuffle:
        random.shuffle(clients_list)

    return num_images_per_client, num_clients, clients_list, train_loaders, test_loader


def save_stats(A, b, root, config_name):
    A = A.to('cpu')
    b = b.to('cpu')
    with open(os.path.join(root, f'{config_name}_stats.pkl'), 'wb') as f:
        pickle.dump({'A': A, 'b': b}, f)


def main():

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = DotDict(config)
    args.wandb_mode = getattr(args, 'wandb_mode', "online")
    args.pretrained_exp = getattr(args, 'pretrained_exp', None)
    if args.pretrained_exp is None:
        pretrained_path = '../model_ckpts/pt_mobilenetv2_100-4065ukrp/pretrained_imagenet.pth'
    else:
        pretrained_path = os.path.join('saved_models', args.pretrained_exp, 'checkpoint.pt')
    if not os.path.exists(args.ds_root):
        raise FileNotFoundError(f"Dataset directory {args.ds_root} does not exists.")
    test_every = getattr(args, 'test_every', 1)

    if args.wandb:
        wandb.login()
        wandb_id = wandb.util.generate_id()
        wandb.init(
            name=sys.argv[1],
            entity="Fed3R",
            project=args.dataset_name,
            resume="allow",
            id=wandb_id,
            config=args.to_dict(),
            mode=args.wandb_mode
        )

    alg = getattr(args, 'algorithm', 'fed3r')
    print(f"Running {alg} experiment")

    seed = args.seed
    tf.random.set_seed(10)  # for a consistent train-test split for the gldv2 dataset, no matter the seed
    random.seed(seed)  # the only seed to fix, as we are interested in different shuffles of clients_list, see later

    if args.model_name == 'mobilenetv2':
        latent_dim = 1280
    elif args.model_name == 'resnet18':
        latent_dim = 512
    else:
        raise NotImplementedError

    if args.dataset_name == 'gldv2':
        num_classes = 2028
    elif args.dataset_name == 'inaturalist':
        num_classes = 1203
    else:
        raise NotImplementedError

    device = args.device
    dataset_name = args.dataset_name
    pfl_mode = args.pfl_mode

    if args.model_name == 'mobilenetv2':
        model = mobilenetv2(num_classes, return_features=False, pretrained_path=pretrained_path,
                            force_num_classes=dataset_name == 'inaturalist' and args.pretrained_exp is not None)
        model_feats = mobilenetv2(num_classes, return_features=True, pretrained_path=pretrained_path,
                                  force_num_classes=dataset_name == 'inaturalist' and args.pretrained_exp is not None)
        model_feats.to(device)
        model_feats.eval()
    else:
        raise NotImplementedError
    model.to(device)
    model.eval()

    num_images_per_client, num_clients, clients_list, train_loaders, test_loader = \
        get_train_test_loaders(args.ds_root, dataset_name, pfl_mode,
                               inat_split=getattr(args, 'split', 'user_120k'))

    A = args.lambda_ * torch.eye(latent_dim + 1, requires_grad=False).to(device)
    b = torch.zeros((latent_dim + 1, num_classes), requires_grad=False).to(device)

    for i, client_id in enumerate(clients_list):

        print(f"Client {client_id} ({i + 1}/{num_clients})...")
        train_loader = train_loaders(client_id, idx=i) if args.pfl_mode else train_loaders(client_id)

        if alg == 'fed3r':
            A, b = fed3r_step(A, b, train_loader, model_feats, num_classes, device)
        else:
            raise NotImplementedError

        if (((i + 1) % args.clients_per_round == 0 and ((i + 1) // args.clients_per_round) % test_every == 0)
                or (i + 1) == num_clients):
            print("Testing...")
            W, bias = normalize_W(*calc_ridge_params(A, b))
            if args.model_name == 'mobilenetv2':
                model.classifier.weight = torch.nn.Parameter(W.T.to(device))
                model.classifier.bias = torch.nn.Parameter(bias.to(device))
            elif args.model_name == 'resnet18':
                model.fc.weight = torch.nn.Parameter(W.T.to(device))
                model.fc.bias = torch.nn.Parameter(bias.to(device))
            else:
                raise NotImplementedError
            round_ = (i + 1) // args.clients_per_round if (i + 1) != num_clients \
                else (i + 1) // args.clients_per_round + 1
            test_model(model, test_loader, device, pfl_mode=pfl_mode,
                       log_wandb=args.wandb, round_=round_)

    if args.save:
        root = 'saved_models/fed3r_classifiers'
        os.makedirs(root, exist_ok=True)
        save_classifier(model, root, config_path.split('/')[-1].split('.')[0])
        save_stats(A, b, root, config_path.split('/')[-1].split('.')[0])


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
