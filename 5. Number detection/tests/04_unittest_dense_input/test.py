import os

import pytest

import common as com
from solution import Dense

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_dense_interface():
    com.check_interface(Dense, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("dense_forward", test_path),
)
def test_dense_forward(test_data):
    com.forward_layer(Dense, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("dense_backward", test_path),
)
def test_dense_backward(test_data):
    com.backward_layer(Dense, test_data)
