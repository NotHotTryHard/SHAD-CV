import os

import pytest

import common as com
from solution import Flatten

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_flatten_interface():
    com.check_interface(Flatten, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("flatten_forward", test_path),
)
def test_flatten_forward(test_data):
    com.forward_layer(Flatten, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("flatten_backward", test_path),
)
def test_flatten_backward(test_data):
    com.backward_layer(Flatten, test_data)
