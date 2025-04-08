import os

from rare_traffic_sign_solution import DatasetRTSD
from rare_traffic_sign_solution import TestData as _TestData


def test_get_classes():
    here = os.path.dirname(os.path.realpath(__file__))
    classes, class_to_idx = DatasetRTSD.get_classes(f"{here}/classes_example.json")
    assert classes == ["1.1", "1.2", "1.12.1", "2.3.2"]
    assert class_to_idx == {"1.1": 0, "1.2": 1, "1.12.1": 2, "2.3.2": 3}


def test_rtsd_dataset_1():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    gt_samples = [
        (f"{here}/data_example_1/1.1/012213.png", 0),
        (f"{here}/data_example_1/1.1/012212.png", 0),
        (f"{here}/data_example_1/1.2/002997.png", 1),
        (f"{here}/data_example_1/1.2/002998.png", 1),
    ]
    assert set(trainset.samples) == set(gt_samples)
    gt_classes_to_samples = {
        0: [
            trainset.samples.index(gt_samples[0]),
            trainset.samples.index(gt_samples[1]),
        ],
        1: [
            trainset.samples.index(gt_samples[2]),
            trainset.samples.index(gt_samples[3]),
        ],
        2: [],
        3: [],
    }
    assert set(trainset.classes_to_samples) == set(gt_classes_to_samples)
    assert len(trainset) == 4
    for i in range(len(trainset)):
        assert trainset.__getitem__(i)[1] == trainset.samples[i][0]
        assert trainset.__getitem__(i)[2] == trainset.samples[i][1]


def test_rtsd_dataset_2():
    here = os.path.dirname(os.path.realpath(__file__))
    trainset = DatasetRTSD(
        root_folders=[f"{here}/data_example_1", f"{here}/data_example_2"],
        path_to_classes_json=f"{here}/classes_example.json",
    )
    gt_samples = [
        (f"{here}/data_example_1/1.1/012213.png", 0),
        (f"{here}/data_example_1/1.1/012212.png", 0),
        (f"{here}/data_example_1/1.2/002997.png", 1),
        (f"{here}/data_example_1/1.2/002998.png", 1),
        (f"{here}/data_example_2/1.12.1/001642.png", 2),
        (f"{here}/data_example_2/1.12.1/001643.png", 2),
        (f"{here}/data_example_2/2.3.2/000528.png", 3),
        (f"{here}/data_example_2/2.3.2/000531.png", 3),
    ]
    assert set(trainset.samples) == set(gt_samples)
    gt_classes_to_samples = {
        0: [
            trainset.samples.index(gt_samples[0]),
            trainset.samples.index(gt_samples[1]),
        ],
        1: [
            trainset.samples.index(gt_samples[2]),
            trainset.samples.index(gt_samples[3]),
        ],
        2: [
            trainset.samples.index(gt_samples[4]),
            trainset.samples.index(gt_samples[5]),
        ],
        3: [
            trainset.samples.index(gt_samples[6]),
            trainset.samples.index(gt_samples[7]),
        ],
    }
    assert set(trainset.classes_to_samples) == set(gt_classes_to_samples)
    assert len(trainset) == 8
    for i in range(len(trainset)):
        assert trainset.__getitem__(i)[1] == trainset.samples[i][0]
        assert trainset.__getitem__(i)[2] == trainset.samples[i][1]


def test_test_dataset_1():
    here = os.path.dirname(os.path.realpath(__file__))
    testset = _TestData(
        f"{here}/data_example_3",
        path_to_classes_json=f"{here}/classes_example.json",
        annotations_file=f"{here}/data_example_3_annotations.csv",
    )
    gt_samples = ["0.png", "2.png", "4.png", "3.png", "1.png"]
    assert set(testset.samples) == set(gt_samples)
    gt_targets = {"0.png": 0, "1.png": 1, "2.png": 1, "3.png": 2, "4.png": 0}
    assert set(testset.targets) == set(gt_targets)
    assert len(testset) == 5
    for i in range(len(testset)):
        assert testset.__getitem__(i)[1] == testset.samples[i]
        assert testset.__getitem__(i)[2] == testset.targets[testset.samples[i]]


def test_test_dataset_2():
    here = os.path.dirname(os.path.realpath(__file__))
    testset = _TestData(
        f"{here}/data_example_3",
        path_to_classes_json=f"{here}/classes_example.json",
    )
    gt_samples = ["0.png", "2.png", "4.png", "3.png", "1.png"]
    assert set(testset.samples) == set(gt_samples)
    assert testset.targets is None
    assert len(testset) == 5
    for i in range(len(testset)):
        assert testset.__getitem__(i)[1] == testset.samples[i]
        assert testset.__getitem__(i)[2] == -1
