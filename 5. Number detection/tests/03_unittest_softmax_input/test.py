import os

import pytest

import common as com
from solution import Softmax

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_softmax_interface():
    com.check_interface(Softmax, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("softmax_forward", test_path),
)
def test_softmax_forward(test_data):
    com.forward_layer(Softmax, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("softmax_backward", test_path),
)
def test_softmax_backward(test_data):
    com.backward_layer(Softmax, test_data)
