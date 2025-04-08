import os

import pytest

import common as com
from solution import BatchNorm

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_batchnorm_interface():
    com.check_interface(BatchNorm, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("batchnorm_forward", test_path),
)
def test_batchnorm_forward(test_data):
    com.forward_layer(BatchNorm, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("batchnorm_backward", test_path),
)
def test_batchnorm_backward(test_data):
    com.backward_layer(BatchNorm, test_data)
