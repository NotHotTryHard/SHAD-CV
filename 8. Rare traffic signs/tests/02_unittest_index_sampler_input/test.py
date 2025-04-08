import collections
import os

from rare_traffic_sign_solution import DatasetRTSD, IndexSampler


def test_index_sampler_1():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    sample = [elem for elem in IndexSampler(trainset, 1)]
    classes = [trainset.samples[elem][1] for elem in sample]
    classes = dict(collections.Counter(classes))
    assert len(classes) == 4
    for _, v in classes.items():
        assert v == 1


def test_index_sampler_2():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    sample = [elem for elem in IndexSampler(trainset, 2)]
    classes = [trainset.samples[elem][1] for elem in sample]
    classes = dict(collections.Counter(classes))
    assert len(classes) == 4
    for _, v in classes.items():
        assert v == 2
