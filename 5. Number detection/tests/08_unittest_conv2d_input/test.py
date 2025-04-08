import os

import pytest

import common as com
from solution import Conv2D

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_conv_interface():
    com.check_interface(Conv2D, com.interface.Layer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("conv2d_forward", test_path),
)
def test_conv2d_forward(test_data):
    com.forward_layer(Conv2D, test_data)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("conv2d_backward", test_path),
)
def test_conv2d_backward(test_data):
    com.backward_layer(Conv2D, test_data)
