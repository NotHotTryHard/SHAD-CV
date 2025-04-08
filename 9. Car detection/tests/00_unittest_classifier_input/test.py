from detection_and_metrics import fit_cls_model, fit_cls_model_good
from numpy import load, int64
from os.path import dirname, join
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import argmax, from_numpy
import torch


def test_classifier_training():
    data = load(join(dirname(__file__), 'train_data.npz'))
    X = data['X'].reshape(-1, 1, 40, 100)   #pytorch dimensions are (N, C, H, W)
    y = data['y'].astype(int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    X_train, X_test = from_numpy(X_train), from_numpy(X_test)
    y_train, y_test = from_numpy(y_train), from_numpy(y_test)
    cls_model = fit_cls_model(X_train, y_train, fast_train=True)
    #cls_model = fit_cls_model_good(X_train, y_train, X_test, y_test)

    y_predicted = argmax(cls_model(X_test), dim = 1)
    acc = accuracy_score(y_test, y_predicted)
    print(acc)

    #cls_model.load_state_dict(torch.load('classifier_model_SOTA.pt', map_location=torch.device('cpu')))

    #y_predicted = argmax(cls_model(X_test), dim = 1)
    #acc = accuracy_score(y_test, y_predicted)
    #print(acc)
    
    assert acc > 0.9
