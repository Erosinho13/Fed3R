import csv

from .landmarks import PersGLDv2FederatedDataloader, GLDv2ClientDataloader
from .inaturalist import PersInatFederatedDataloader, InatClientDataloader
from .tf_utils import read_original_client_sizes_csv


def get_federated_dataloader_from_args(args):
    train_batch_size = args.train_batch_size
    eval_batch_size = args.train_batch_size if args.eval_batch_size is None else args.eval_batch_size
    train_mode = 'train'
    test_mode = 'train' if args.validation_mode else 'test'

    mean_clients, std_clients = {}, {}

    if args.dataset.lower() == 'gldv2':
        client_list_fn = 'dataset_statistics/gldv2_client_ids_{}.csv'
        train_loader = PersGLDv2FederatedDataloader(
            args.data_dir, client_list_fn.format('train'), train_mode,
            train_batch_size, args.max_num_elements_per_client,
            validation_mode=args.validation_mode, validation_holdout=False,
            mean_clients=mean_clients, std_clients=std_clients
        )
        path = f"dataset_statistics/gldv2_original_client_sizes.csv"
        num_images_per_client = read_original_client_sizes_csv(path)
        test_loader = GLDv2ClientDataloader(train_loader.fl_test_dataset, 50, num_images_per_client,
                                            client_id=None, is_train=False, mean_clients=mean_clients,
                                            std_clients=std_clients)

    elif args.dataset.lower() == 'inaturalist':
        split = getattr(args, 'split', 'geo_100')
        client_list_fn = 'dataset_statistics/inaturalist_{}_client_ids_{}.csv'
        train_loader = PersInatFederatedDataloader(
            args.data_dir, client_list_fn.format(split, 'train'), train_mode,
            train_batch_size, args.max_num_elements_per_client,
            validation_mode=args.validation_mode, validation_holdout=False, split_name=split,
            mean_clients=mean_clients, std_clients=std_clients
        )
        path = f"dataset_statistics/inaturalist_geo_100_original_client_sizes.csv"
        num_images_per_client = read_original_client_sizes_csv(path)
        test_loader = InatClientDataloader(train_loader.fl_test_dataset, 50, num_images_per_client,
                                           client_id=None, is_train=False,
                                           mean_clients=mean_clients, std_clients=std_clients)

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return train_loader, test_loader
