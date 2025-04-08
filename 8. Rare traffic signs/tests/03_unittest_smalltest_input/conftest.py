# -*- coding: utf-8 -*-


def pytest_addoption(parser):
    parser.addoption("--data_dir", action="store")
    parser.addoption("--output_dir", action="store")
