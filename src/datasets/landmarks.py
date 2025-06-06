import numpy as np
import torch
import csv
import tensorflow as tf

import datasets.landmarks_utils as landmarks_utils
from datasets.tf_utils import center_crop_and_resize, random_crop_and_resize, read_original_client_sizes_csv, \
    TFClientDataloader, PersTFClientDataloader, PersTFFederatedDataloader

# ImageNet defaults
# MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
# STD = torch.FloatTensor([0.229, 0.224, 0.225])
MEAN = torch.FloatTensor([0.5, 0.5, 0.5])
STD = torch.FloatTensor([0.5, 0.5, 0.5])
CROP_PADDING = 32



def train_map_fn(ex):
    image_size = 224
    image = ex['image/decoded']
    image = random_crop_and_resize(image, image_size, CROP_PADDING) / 255.0
    image = tf.image.random_flip_left_right(image)
    y = ex['class']
    return image, y  # image: (H, W, 3), y: tf.int64


def test_map_fn(ex):
    image_size = 224
    image = ex['image/decoded']
    x = center_crop_and_resize(image, image_size, CROP_PADDING) / 255.0
    y = ex['class']
    return x, y


class GLDv2ClientDataloader(TFClientDataloader):

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
        return True


class PersGLDv2ClientDataloader(PersTFClientDataloader):

    def __init__(self, tf_dataset, batch_size, dataset_size, client_id, max_elements_per_client, is_train,
                 validation_mode=False, validation_holdout=False, force_test_augmentation=False, **kwargs):

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
        return self.client_id

    @staticmethod
    def filter_fn(img, target):
        return True


class PersGLDv2FederatedDataloader(PersTFFederatedDataloader):
    def __init__(self, data_dir, client_list, split, batch_size,
                 max_num_elements_per_client=1000, shuffle=True,
                 validation_mode=False, validation_holdout=False, **kwargs):
        super().__init__(data_dir, client_list, split, batch_size, max_num_elements_per_client, shuffle=shuffle,
                         validation_mode=validation_mode, validation_holdout=validation_holdout, kwargs=kwargs)

    @property
    def dataset_name(self):
        return 'gldv2'

    def load_tf_fed_dataset(self):
        return landmarks_utils.load_data(cache_dir=self.data_dir)

    @property
    def PersClientDataloader(self):
        return PersGLDv2ClientDataloader

    @property
    def num_classes(self):
        return 2028


def generate_gldv2_original_client_sizes_csv(test_dataset, tf_fed_dataset):
    num_test_images = sum(1 for _ in test_dataset)
    num_client_images = {'test': num_test_images}
    for cl_id in tf_fed_dataset.client_ids:
        if cl_id == 'missing':
            continue
        num_client_images[cl_id] = sum(1 for _ in tf_fed_dataset.create_tf_dataset_for_client(cl_id))
        print(f"client {cl_id}: {num_client_images[cl_id]}")
    print(f"Total training images: {sum(num_client_images.values())}")
    print(f"Total clients: {len(num_client_images.keys())}")
    headers = ['client', 'num_images']
    with open('dataset_statistics/gldv2_original_client_sizes.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(list(num_client_images.items()))
