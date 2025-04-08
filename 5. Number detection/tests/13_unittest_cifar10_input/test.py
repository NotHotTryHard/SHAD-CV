import math
import os
import random
import time

import numpy as np

import common as com
from interface import Model
from solution import train_cifar10_model

test_path = os.path.dirname(__file__)


def train_valid_split(x, y, valid_size):
    valid_size = math.ceil(valid_size * x.shape[0])
    return (x[valid_size:], y[valid_size:]), (x[:valid_size], y[:valid_size])


def prepare_data(images, labels):
    cifar10_means = np.asarray([120.846252, 120.84548652, 120.84175166])
    cifar10_stds = np.asarray([64.13531818, 64.13287616, 64.13969862])
    cifar10_means = cifar10_means[np.newaxis, :, np.newaxis, np.newaxis]
    cifar10_stds = cifar10_stds[np.newaxis, :, np.newaxis, np.newaxis]

    normalized_images = (images - cifar10_means) / cifar10_stds
    onehot_labels = np.eye(10)[labels]

    return normalized_images, onehot_labels


def test_main():
    random.seed(42)
    np.random.seed(42)
    os.environ.setdefault("USE_FAST_CONVOLVE", "True")

    # Load CIFAR10 data
    x, y = prepare_data(
        com.load_test_data("cifar10_train_images", test_path),
        com.load_test_data("cifar10_train_labels", test_path),
    )
    (x_train, y_train), (x_valid, y_valid) = train_valid_split(x, y, 0.2)

    print()
    train_start = time.time()
    model = train_cifar10_model(x_train, y_train, x_valid, y_valid)
    train_end = time.time()
    assert isinstance(model, Model)

    x_test, y_test = prepare_data(
        com.load_test_data("cifar10_test_images", test_path),
        com.load_test_data("cifar10_test_labels", test_path),
    )
    test_start = time.time()
    test_loss, test_accuracy = model.evaluate(x_test, y_test, 1)
    test_end = time.time()

    print(f"Final test loss: {test_loss:.8g}")
    print(f"Final test accuracy: {test_accuracy:.4%}")
    assert test_accuracy >= 0.7

    train_time = train_end - train_start
    test_time = test_end - test_start
    print(f"Total training time: {train_time:.8g}")
    print(f"Total testing time: {test_time:.8g}")
    assert train_time + test_time <= 60 * 60
