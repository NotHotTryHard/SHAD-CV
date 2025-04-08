import collections
import os

from rare_traffic_sign_solution import CustomBatchSampler, DatasetRTSD


def test_custom_batch_sampler_1():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    sample = next(iter(CustomBatchSampler(trainset, 4, 1)))
    classes = [trainset.samples[elem][1] for elem in sample]
    classes = dict(collections.Counter(classes))
    assert len(classes) == 1
    for _, v in classes.items():
        assert v == 4


def test_custom_batch_sampler_2():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    sample = next(iter(CustomBatchSampler(trainset, 2, 4)))
    classes = [trainset.samples[elem][1] for elem in sample]
    classes = dict(collections.Counter(classes))
    assert len(classes) == 4
    for _, v in classes.items():
        assert v == 2


def test_custom_batch_sampler_3():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    sample = next(iter(CustomBatchSampler(trainset, 3, 2)))
    classes = [trainset.samples[elem][1] for elem in sample]
    classes = dict(collections.Counter(classes))
    assert len(classes) == 2
    for _, v in classes.items():
        assert v == 3
