import os

import pytest

import common as com
from solution import SGDMomentum

test_path = os.path.dirname(__file__)


@pytest.mark.tryfirst
def test_momentum_interface():
    com.check_interface(SGDMomentum, com.interface.Optimizer)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("momentum_updates", test_path),
)
def test_momentum_updates(test_data):
    com.simulate_optimizer(SGDMomentum, test_data)
