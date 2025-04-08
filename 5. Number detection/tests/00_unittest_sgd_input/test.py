import os

import pytest

import common as com
from solution import SGD

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_sgd_interface():
    com.check_interface(SGD, com.interface.Optimizer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("sgd_updates", test_path),
)
def test_sgd_updates(test_data):
    com.simulate_optimizer(SGD, test_data)
