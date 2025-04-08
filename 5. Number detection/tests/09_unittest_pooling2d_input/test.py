import os

import pytest

import common as com
from solution import Pooling2D

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_pooling_interface():
    com.check_interface(Pooling2D, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("pooling2d_forward", test_path),
)
def test_pooling2d_forward(test_data):
    com.forward_layer(Pooling2D, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("pooling2d_backward", test_path),
)
def test_pooling2d_backward(test_data):
    com.backward_layer(Pooling2D, test_data)
