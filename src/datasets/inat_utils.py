import collections
import os
import logging
import logging.handlers
import multiprocessing.pool
import sys
import traceback
import tensorflow as tf

from typing import Tuple

from datasets.tf_utils import CustomClientData

LOGGER = 'inaturalist'
KEY_IMAGE_BYTES = 'image/encoded_jpeg'
KEY_IMAGE_DECODED = 'image/decoded'
KEY_CLASS = 'class'
TRAIN_SUB_DIR = 'train'
TEST_FILE_NAME = 'test.tfRecord'


def _parse_and_transform(example_proto) -> collections.OrderedDict:
    feature_description = {
        KEY_IMAGE_BYTES: tf.io.FixedLenFeature([], tf.string, default_value=''),
        KEY_CLASS: tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    decoded_image = tf.io.decode_jpeg(parsed_example[KEY_IMAGE_BYTES])
    class_label = tf.reshape(parsed_example[KEY_CLASS], [1])
    return collections.OrderedDict([
        (KEY_IMAGE_DECODED, decoded_image),
        (KEY_CLASS, class_label)
    ])


def _listener_process(queue: multiprocessing.Queue, log_file: str):
    """
    Sets up a separate process for handling logging messages.
    This setup is required because without it, the logging messages will be
    duplicated when multiple processes are created for downloading GLD dataset.
    Args:
        queue: The queue to receive logging messages.
        log_file: The file which the messages will be written to.
    """
    root = logging.getLogger()
    h = logging.FileHandler(log_file)
    fmt = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    h.setFormatter(fmt)
    root.addHandler(h)
    while True:
        try:
            record = queue.get()
            # We send None as signal to stop
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:  # pylint: disable=broad-except
            print('Something went wrong:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def _load_data_from_cache(cache_dir: str) -> Tuple[CustomClientData, tf.data.Dataset]:
    """Load train and test data from the TFRecord files.
  Args:
    cache_dir: The directory containing the TFRecord files.
  Returns:
    A tuple of `ClientData`, `tf.data.Dataset`.
  """
    logger = logging.getLogger(LOGGER)
    train_dir = os.path.join(cache_dir, TRAIN_SUB_DIR)
    logger.info('Start to load train data from cache directory: %s', train_dir)
    train = CustomClientData.from_directory(train_dir, _parse_and_transform)
    logger.info('Finish loading train data from cache directory: %s', train_dir)
    test_file = os.path.join(cache_dir, TEST_FILE_NAME)
    logger.info('Start to load test data from file: %s', test_file)
    test = _load_tfrecord(test_file)
    logger.info('Finish loading test data from file: %s', test_file)
    return train, test


def load_data(num_worker: int = 1,
              cache_dir: str = 'cache',
              inat_split: str = 'user_120k'):

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    q = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(
        target=_listener_process,
        args=(q, os.path.join(cache_dir, 'load_data.log')))
    listener.start()
    logger = logging.getLogger(LOGGER)
    qh = logging.handlers.QueueHandler(q)
    logger.addHandler(qh)
    logger.info('Start to load data.')
    existing_data_cache = os.path.join(cache_dir, 'inaturalist', inat_split)
    logger.info('Try loading dataset from cache')
    return _load_data_from_cache(existing_data_cache)


def _load_tfrecord(fname: str) -> tf.data.Dataset:
    """
    Reads a `tf.data.Dataset` from a TFRecord file.
    Parse each element into a `tf.train.Example`.
    Args:
        fname: The file name of the TFRecord file.
    Returns:
        `tf.data.Dataset`.
    """
    logger = logging.getLogger(LOGGER)
    logger.info('Start loading dataset for file %s', fname)
    raw_dataset = tf.data.TFRecordDataset([fname])

    def _parse(example_proto):
        feature_description = {
            KEY_IMAGE_BYTES: tf.io.FixedLenFeature([], tf.string, default_value=''),
            KEY_CLASS: tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        }
        return collections.OrderedDict(
            tf.io.parse_single_example(example_proto, feature_description))

    ds = raw_dataset.map(_parse)

    def _transform(item):
        return collections.OrderedDict([
            (KEY_IMAGE_DECODED, tf.io.decode_jpeg(item[KEY_IMAGE_BYTES])),
            (KEY_CLASS, tf.reshape(item[KEY_CLASS], [1]))
        ])

    ds = ds.map(_transform)
    logger.info('Finished loading dataset for file %s', fname)
    return ds
