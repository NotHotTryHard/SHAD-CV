import os

import pytest

import common as com
from solution import Dropout

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_dropout_interface():
    com.check_interface(Dropout, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("dropout_forward", test_path),
)
def test_dropout_forward(test_data):
    com.forward_layer(Dropout, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("dropout_backward", test_path),
)
def test_dropout_backward(test_data):
    com.backward_layer(Dropout, test_data)
