import csv
import os
import time
from collections import OrderedDict
from typing import List, Dict

import math
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tqdm import tqdm

from datasets.dataloader import ClientDataloader, FederatedDataloader


def center_crop_and_resize(image, image_size, crop_padding):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + crop_padding)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)

    resized_image = tf.image.resize(cropped_image, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC)

    return resized_image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(3. / 4, 4. / 3.),
                                area_range=(0.08, 1.0), max_attempts=10):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image


def _resize(image, image_size):
    return tf.image.resize([image], [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC)[0]


def random_crop_and_resize(image, image_size, crop_padding):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    cropped_image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10)

    original_shape = tf.shape(image)
    bad = _at_least_x_are_equal(original_shape, tf.shape(cropped_image), 3)

    image = tf.cond(
        bad,
        lambda: center_crop_and_resize(image, image_size, crop_padding),
        lambda: tf.image.resize(cropped_image, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC))

    return image


def read_original_client_sizes_csv(csv_file_path):
    clients = []
    num_images = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            clients.append(row[0])
            num_images.append(int(row[1]))
    return {k: v for k, v in zip(clients, num_images)}


class TFClientDataloader:

    def __init__(self, tf_dataset, batch_size, num_images_per_client, client_id=None, is_train=True,
                 mean_clients=None, std_clients=None):

        self.tf_dataset = tf_dataset.create_tf_dataset_for_client(client_id) if client_id is not None else tf_dataset
        self.dataset_size = num_images_per_client[client_id] if client_id is not None else num_images_per_client['test']
        self.batch_size = batch_size
        self.client_id = client_id
        self.is_train = is_train
        self.tf_dataset_iterator = None
        self.idx = 0
        self.mean_clients = mean_clients
        self.std_clients = std_clients

        if type(self.mean_clients) == dict:
            if len(self.mean_clients) == 0:
                self.mean_clients = None
        if type(self.std_clients) == dict:
            if len(self.std_clients) == 0:
                self.std_clients = None

        self.mean_clients_np = np.array(list(self.mean_clients.values())) if self.mean_clients is not None else None
        self.std_clients_np = np.array(list(self.std_clients.values())) if self.std_clients is not None else None

        if self.mean_clients_np is not None:
            if len(self.mean_clients_np) == 0:
                self.mean_clients_np = None
        if self.std_clients_np is not None:
            if len(self.std_clients_np) == 0:
                self.std_clients_np = None

        self.reinitialize()

    def reinitialize(self):
        iterator = self.tf_dataset.shuffle(len(self), seed=torch.randint(1 << 20, (1,)).item())
        if self.is_train:
            map_fn = self.train_map_fn
        else:
            map_fn = self.test_map_fn
        iterator = iterator.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        iterator = iterator.filter(self.filter_fn)
        self.tf_dataset_iterator = iter(iterator.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE))
        self.idx = 0

    def __len__(self):
        return int(math.ceil(self.dataset_size / self.batch_size))

    def __iter__(self):  # reintialize each time the iterator is called
        self.reinitialize()
        return self

    def __next__(self):
        x, y = next(self.tf_dataset_iterator)  # (tf.Tensor, tf.Tensor)
        x = torch.from_numpy(x.numpy())  # (B, H, W, C)
        y = torch.from_numpy(y.numpy())  # (B, 1)
        if self.is_train and self.mean_clients_np is not None and self.std_clients_np is not None:
            raise NotImplementedError  # We should never go here
        elif not self.is_train and self.mean_clients_np is not None and self.std_clients_np is not None:
            mean = torch.from_numpy(self.mean_clients_np[self.idx % 3]).float()
            std = torch.from_numpy(self.std_clients_np[self.idx % 3]).float()
            x = (x - mean[None, None, None]) / std[None, None, None]  # Normalize
        else:
            x = (x - self.mean[None, None, None]) / self.std[None, None, None]  # Normalize
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        self.idx += 1
        return x, y.view(-1)

    @property
    def train_map_fn(self):
        raise NotImplementedError

    @property
    def test_map_fn(self):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError

    @staticmethod
    def filter_fn(img, target):
        raise NotImplementedError


class PersTFClientDataloader(ClientDataloader):
    """An iterator which wraps the tf.data iteratator to behave like a PyTorch data loader.
    """

    def __init__(
            self, tf_dataset, batch_size, dataset_size, client_id, max_elements_per_client, is_train,
            validation_mode=False, validation_holdout=False, force_test_augmentation=False,
            **kwargs
    ):
        super().__init__()
        self.idx = kwargs['kwargs']['idx'] if 'idx' in kwargs['kwargs'].keys() else None
        self.tf_dataset = tf_dataset
        self.batch_size = batch_size
        self.original_dataset_size = min(dataset_size, max_elements_per_client)  # Number of datapoints in client
        self.client_id = client_id  # int
        self.max_elements_per_client = max_elements_per_client
        self.is_train = is_train
        self.force_test_augmentation = force_test_augmentation
        if not self.is_train:  # test
            self.skip = self.original_dataset_size  # skip the train part
            self.dataset_size = self.original_dataset_size
        elif validation_mode:
            if validation_holdout:
                self.skip = 0
                self.dataset_size = max(1, int(0.2 * self.original_dataset_size))  # 20% holdout
            else:
                self.skip = max(1, int(0.2 * self.original_dataset_size))  # skip the validation part
                self.dataset_size = self.original_dataset_size - self.skip
        else:
            self.skip = 0
            self.dataset_size = self.original_dataset_size
        self.tf_dataset_iterator = None
        self.reinitialize()  # initialize iterator

    def reinitialize(self):
        # iterator = self.tf_dataset.shuffle(self.original_dataset_size, seed=self.ds_seed)  # for the train-test split
        iterator = self.tf_dataset.skip(self.skip).take(self.dataset_size)
        if self.is_train:
            iterator = iterator.shuffle(self.dataset_size, seed=torch.randint(1 << 20, (1,)).item())
            map_fn = self.train_map_fn if not self.force_test_augmentation else self.test_map_fn
        else:
            map_fn = self.test_map_fn
        self.tf_dataset_iterator = iter(iterator
                                        .map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                                        .filter(self.filter_fn)
                                        .batch(self.batch_size)
                                        .prefetch(tf.data.experimental.AUTOTUNE)
                                        )

    def __len__(self):
        return int(math.ceil(self.dataset_size / self.batch_size))

    def __iter__(self):  # reintialize each time the iterator is called
        self.reinitialize()
        return self

    def __next__(self):

        x, y = next(self.tf_dataset_iterator)  # (tf.Tensor, tf.Tensor)
        x = torch.from_numpy(x.numpy())  # (B, H, W, C)
        y = torch.from_numpy(y.numpy())  # (B, 1)

        if self.is_train and self.mean_clients_np is not None and self.std_clients_np is not None:
            indices = np.random.choice(self.mean_clients_np.shape[0], x.shape[0], replace=True)
            mean = torch.from_numpy(self.mean_clients_np[indices]).float()
            std = torch.from_numpy(self.std_clients_np[indices]).float()
            x = (x - mean[:, None, None]) / std[:, None, None]  # Normalize
        elif not self.is_train and self.mean_clients_np is not None and self.std_clients_np is not None:
            raise NotImplementedError  # we should never go here
        else:
            mean = self.mean
            std = self.std
            x = (x - mean[None, None, None]) / std[None, None, None]  # Normalize
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x, y.view(-1)

    @property
    def train_map_fn(self):
        raise NotImplementedError

    @property
    def test_map_fn(self):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError

    @property
    def ds_seed(self):
        raise NotImplementedError

    @staticmethod
    def filter_fn(img, target):
        raise NotImplementedError


class PersTFFederatedDataloader(FederatedDataloader):
    def __init__(self, data_dir, client_list, split, batch_size, max_num_elements_per_client=1000, shuffle=True,
                 validation_mode=False, validation_holdout=False, **kwargs):
        """Federated dataloader. Takes a client id and returns the dataloader for that client.

        Args:
            data_dir ([str]): Directory containing the cached data
            client_list ([str or list or None]): List of clients or filename from which to load clients
            split ([str]): 'train' or 'test'
            batch_size ([int]): batch size on client
            max_num_elements_per_client ([int]): maximum allowed data size
            shuffle (bool, optional): Does client dataloader shuffle the data? Defaults to True.
                    Ignored for this datset.
        """
        super().__init__(data_dir, client_list, split, batch_size, max_num_elements_per_client)
        self.is_train = (split == 'train')
        self.data_dir = data_dir
        if split not in ['train', 'test']:
            raise ValueError(f'Unknown split: {split}')
        if type(client_list) is str:  # It is a filename, read it
            client_list = pd.read_csv(client_list, dtype=str).to_numpy().reshape(-1).tolist()
        elif client_list is None:  # use all clients
            pass
        elif type(client_list) is not list or len(client_list) <= 1:
            raise ValueError(f'{self.dataset_name} dataset requires the list of clients to be specified.')
        if client_list is not None:
            self.available_clients_set = set(client_list)
            self.available_clients = client_list
        self.batch_size = batch_size
        self.max_num_elements_per_client = max_num_elements_per_client
        self.validation_mode = validation_mode
        self.validation_holdout = validation_holdout
        self.ds_and_split_name = \
            f"{self.dataset_name}{f'_{self.split_name}' if self.dataset_name == 'inaturalist' else ''}"

        sizes_filename = f'dataset_statistics/{self.ds_and_split_name}_client_sizes_{split}.csv'
        self.client_sizes = pd.read_csv(sizes_filename, index_col=0, dtype='string').squeeze().to_dict()
        self.client_sizes = {k: int(v) for (k, v) in self.client_sizes.items()}  # convert client size to int

        print('Loading data')
        start_time = time.time()
        self.tf_fed_dataset, self.fl_test_dataset = self.load_tf_fed_dataset()
        if client_list is None:  # use all clients
            self.available_clients = self.tf_fed_dataset.client_ids
            self.available_clients_set = set(self.tf_fed_dataset.client_ids)
        self.mean_clients, self.std_clients = None, None
        if 'mean_clients' in kwargs['kwargs'].keys() and 'std_clients' in kwargs['kwargs'].keys():
            self.mean_clients = kwargs['kwargs']['mean_clients']
            self.std_clients = kwargs['kwargs']['std_clients']

        print(f'Loaded data in {round(time.time() - start_time, 2)} seconds')

    def get_client_dataloader(self, client_id, idx):
        if client_id in self.available_clients_set:
            return self.PersClientDataloader(
                self.tf_fed_dataset.create_tf_dataset_for_client(client_id),
                self.batch_size, self.client_sizes[client_id], int(client_id),
                self.max_num_elements_per_client, self.is_train,
                self.validation_mode, self.validation_holdout,
                mean_clients=self.mean_clients, std_clients=self.std_clients,
                client_name=client_id, idx=idx)
        else:
            raise ValueError(f'Unknown client: {client_id}')

    def __len__(self):
        return len(self.available_clients)

    def get_loss_and_metrics_fn(self):
        return self.loss_of_batch_fn, self.metrics_of_batch_fn

    @property
    def dataset_name(self):
        return self.ds_and_split_name

    def load_tf_fed_dataset(self):
        raise NotImplementedError

    @property
    def PersClientDataloader(self):
        raise NotImplementedError

    @property
    def loss_of_batch_fn(self):
        return loss_of_batch_fn

    @property
    def metrics_of_batch_fn(self):
        return metrics_of_batch_fn

    @property
    def num_classes(self):
        raise NotImplementedError


@torch.no_grad()
def metrics_of_batch_fn(y_pred, y_true):
    # y_true: (batch_size,); y_pred: (batch_size, num_classes)
    loss_fn = torch.nn.functional.cross_entropy
    argmax = torch.argmax(y_pred, axis=1)
    metrics = OrderedDict([
        ('loss', loss_fn(y_pred, y_true).item()),
        ('accuracy', (argmax == y_true).sum().item() * 1.0 / y_true.shape[0])
    ])
    return y_true.shape[0], metrics


def loss_of_batch_fn(y_pred, y_true):
    return torch.nn.functional.cross_entropy(y_pred, y_true)


class CustomClientData:
    def __init__(self, client_ids: List[str], data_per_client: Dict[str, tf.data.Dataset]):
        self._client_ids = client_ids
        self._data_per_client = data_per_client

    @property
    def client_ids(self):
        return self._client_ids

    def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
        return self._data_per_client[client_id]

    @staticmethod
    def from_directory(data_dir: str, _parse_and_transform):
        client_ids = sorted(os.listdir(data_dir))
        data_per_client = {}
        for client_id in tqdm(client_ids):
            client_path = os.path.join(data_dir, client_id)
            if os.path.isdir(client_path):
                # If it's a directory, load all files inside it
                data_files = [os.path.join(client_path, f) for f in os.listdir(client_path)]
            else:
                # If it's a single file, treat it as the only data file for this client
                data_files = [client_path]

            client_dataset = tf.data.TFRecordDataset(data_files).map(_parse_and_transform)
            data_per_client[client_id] = client_dataset
        return CustomClientData(client_ids, data_per_client)