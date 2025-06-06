import csv
import numpy as np

import torch
import tensorflow as tf
from tqdm import tqdm

from datasets import inat_utils
from datasets.tf_utils import random_crop_and_resize, TFClientDataloader, read_original_client_sizes_csv, \
    center_crop_and_resize, PersTFClientDataloader, PersTFFederatedDataloader

# ImageNet defaults
# MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
# STD = torch.FloatTensor([0.229, 0.224, 0.225])
MEAN = torch.FloatTensor([0.5, 0.5, 0.5])
STD = torch.FloatTensor([0.5, 0.5, 0.5])
CROP_PADDING = 32


def train_map_fn(ex):
    image_size = 299
    image = ex['image/decoded']
    image = random_crop_and_resize(image, image_size, CROP_PADDING) / 255.0
    image = tf.image.random_flip_left_right(image)
    y = ex['class']
    return image, y


def test_map_fn(ex):
    image_size = 299
    image = ex['image/decoded']
    x = center_crop_and_resize(image, image_size, CROP_PADDING) / 255.0
    y = ex['class']
    return x, y


class InatClientDataloader(TFClientDataloader):

    def __init__(self, tf_dataset, batch_size, num_images_per_client, client_id=None, is_train=True,
                 mean_clients=None, std_clients=None):
        super().__init__(tf_dataset, batch_size, num_images_per_client, client_id=client_id, is_train=is_train,
                         mean_clients=mean_clients, std_clients=std_clients)

    @property
    def train_map_fn(self):
        return train_map_fn

    @property
    def test_map_fn(self):
        return test_map_fn

    @property
    def mean(self):
        return MEAN

    @property
    def std(self):
        return STD

    @staticmethod
    def filter_fn(img, target):
        return tf.equal(tf.shape(img)[-1], 3)  # filter out grayscale images (needed only for a few clients)


class PersInatClientDataloader(PersTFClientDataloader):

    def __init__(self, tf_dataset, batch_size, dataset_size, client_id, max_elements_per_client, is_train,
                 validation_mode=False, validation_holdout=False, force_test_augmentation=False, split_name='user_120k',
                 **kwargs):

        self.mean_clients = kwargs['mean_clients'] if 'mean_clients' in kwargs.keys() else None
        self.std_clients = kwargs['std_clients'] if 'std_clients' in kwargs.keys() else None
        self.mean_clients_np = np.array(list(self.mean_clients.values())) if self.mean_clients is not None else None
        self.std_clients_np = np.array(list(self.std_clients.values())) if self.std_clients is not None else None
        if self.mean_clients_np is not None:
            if len(self.mean_clients_np) == 0:
                self.mean_clients_np = None
        if self.std_clients_np is not None:
            if len(self.std_clients_np) == 0:
                self.std_clients_np = None
        self.client_name = kwargs['client_name'] if 'client_name' in kwargs.keys() else None

        super().__init__(
            tf_dataset, batch_size, dataset_size, client_id, max_elements_per_client, is_train,
            validation_mode=validation_mode, validation_holdout=validation_holdout,
            force_test_augmentation=force_test_augmentation, kwargs=kwargs
        )
        self.split_name = split_name

    @property
    def train_map_fn(self):
        return train_map_fn

    @property
    def test_map_fn(self):
        return test_map_fn

    @property
    def mean(self):
        return MEAN

    @property
    def std(self):
        return STD

    @property
    def ds_seed(self):
        return sum(ord(char) for char in str(self.client_id))

    @staticmethod
    def filter_fn(img, target):
        return tf.equal(tf.shape(img)[-1], 3)  # filter out grayscale images (needed only for a few clients)


class PersInatFederatedDataloader(PersTFFederatedDataloader):
    def __init__(self, data_dir, client_list, split, batch_size,
                 max_num_elements_per_client=1000, shuffle=True,
                 validation_mode=False, validation_holdout=False, split_name='user_120k', **kwargs):
        self.split_name = split_name
        super().__init__(data_dir, client_list, split, batch_size, max_num_elements_per_client, shuffle=shuffle,
                         validation_mode=validation_mode, validation_holdout=validation_holdout, kwargs=kwargs)

    @property
    def dataset_name(self):
        return f'inaturalist_{self.split_name}'

    def load_tf_fed_dataset(self):
        return inat_utils.load_data(cache_dir=self.data_dir, inat_split=self.split_name)

    @property
    def PersClientDataloader(self):
        return PersInatClientDataloader

    @property
    def num_classes(self):
        return 1203


def get_client_sizes(tf_fed_dataset, split='user_120k'):

    map_fn = test_map_fn

    with open(f'dataset_statistics/inaturalist_{split}_original_client_sizes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('client', 'num_images'))

    client_ids = tf_fed_dataset.client_ids

    for cid in tqdm(client_ids):
        ds = tf_fed_dataset.create_tf_dataset_for_client(cid)
        ds = ds.map(map_fn)
        ds = ds.filter(lambda img, target: tf.equal(tf.shape(img)[-1], 3))
        ds = iter(ds.batch(50).prefetch(tf.data.experimental.AUTOTUNE))
        try:
            len_ds = 0
            for img, target in ds:
                len_ds += len(target)
            with open(f'dataset_statistics/inaturalist_{split}_original_client_sizes.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow((cid, len_ds))
        except:
            print(f"Problematic client: {cid}")

