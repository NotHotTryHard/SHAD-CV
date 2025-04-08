import os

import pytest

import common as com
from solution import ReLU

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_relu_interface():
    com.check_interface(ReLU, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("relu_forward", test_path),
)
def test_relu_forward(test_data):
    com.forward_layer(ReLU, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("relu_backward", test_path),
)
def test_relu_backward(test_data):
    com.backward_layer(ReLU, test_data)
