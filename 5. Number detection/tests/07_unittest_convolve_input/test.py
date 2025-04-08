import os

import pytest

import common as com
from solution import convolve_numpy

test_path = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "test_data",
    com.load_test_data("convolve_function", test_path),
)
def test_convolve_function(test_data):
    com.function(convolve_numpy, test_data)
