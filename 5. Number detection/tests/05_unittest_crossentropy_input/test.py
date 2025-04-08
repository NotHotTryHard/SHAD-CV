import os

import pytest

import common as com
from solution import CategoricalCrossentropy

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_crossentropy_interface():
    com.check_interface(CategoricalCrossentropy, com.interface.Loss)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("crossentropy_values", test_path),
)
def test_crossentropy_values(test_data):
    com.loss(CategoricalCrossentropy, test_data, "value")


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("crossentropy_gradients", test_path),
)
def test_crossentropy_gradients(test_data):
    com.loss(CategoricalCrossentropy, test_data, "gradient")
