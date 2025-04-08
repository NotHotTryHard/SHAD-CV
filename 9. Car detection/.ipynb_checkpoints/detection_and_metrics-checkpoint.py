# ============================== 1 Classifier model ============================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import numpy as np
import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import accuracy_score
import copy

LR = 5e-4
OUTPUT_SIZE = 2
BATCH_SIZE = 32
MAX_EPOCHS = 10
NUM_WORKERS = 6
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cuda:0"):
    torch.set_float32_matmul_precision('medium')

augmentations = [
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=(-0.1, 0.2), p=0.25),
    A.GaussianBlur(sigma_limit=(0.2, 1), p=0.4),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(0.01, 0.02), p=0.3),
]

common_transforms = [
    A.Normalize(max_pixel_value=1.0),
    A.pytorch.transforms.ToTensorV2(),
]

MyFitTransform = A.ReplayCompose(augmentations + common_transforms)
MyPredictTransform = A.ReplayCompose(common_transforms)

def ensure_numpy(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    elif not isinstance(image, np.ndarray):
        raise TypeError
    return image

class CarDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        assert self.X.shape[0] == self.y.shape[0] 

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.X[idx][0]
        label = self.y[idx]
        

        if self.transform:
            image = ensure_numpy(image)
            augmented = self.transform(image=image)
            #print(augmented['replay'])
            image = augmented['image'][0]
        #sample = {'image': image, 'label': label}
        
        return image.unsqueeze(0), label

class MyModel1(nn.Module):
    def __init__(self):
        super().__init__()
        
        def convpool(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels//2, 3, stride=1, padding=1),
                nn.LeakyReLU(0.3),
                nn.Conv2d(out_channels//2, out_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(0.3),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            )

        self.model = nn.Sequential(
            convpool(1, 32),
            convpool(32, 128),
            nn.Flatten(),
            nn.Linear(128 * 10 * 25, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE)
        )
    def forward(self, X):
        #X = X.unsqueeze(1)
        return self.model(X)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), 
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(),
            nn.Linear(128 * 3 * 10, 256), # 3 x 10
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE)
        )
    def forward(self, X):
        #X = X.unsqueeze(1)
        return self.model(X)

class ModifiedModel(MyModel):
    def __init__(self):
        super(ModifiedModel, self).__init__()

    def forward(self, X):
        return self.model(X)[:, 1, ...]

def get_cls_model(input_shape=(1, 40, 100)):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    classification_model = MyModel()
    return classification_model
    # your code here /\


def fit_cls_model_good(X, y, Xv, yv):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    #ds_train = CarDataset(X, y, MyFitTransform)
    ds_train = CarDataset(X, y, None)
    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    ds_valid = CarDataset(Xv, yv, None)
    dl_valid = DataLoader(
        ds_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )
    
    

    epochs = 100
    max_acc = float('-inf')
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        progress_bar = tqdm.tqdm(dl_train, desc=f'Epoch [{epoch + 1}/{epochs}]')
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': running_loss / (progress_bar.n + 1)})

        model.eval()
        outputs = model(Xv)
        val_loss = loss_fn(outputs, yv)
        scheduler.step(val_loss)

        y_predicted = torch.argmax(outputs, 1)
        acc = accuracy_score(yv, y_predicted)
        print('Accuracy of the model on the validation set: {}%'.format(100 * acc))
        if acc > max_acc:
            torch.save(model.state_dict(), 'classifier_model_SOTA.pt')
    torch.save(model.state_dict(), 'classifier_model.pt')
    return model

def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #ds_train = CarDataset(X, y, MyFitTransform)
    ds_train = CarDataset(X, y, None)
    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    
    model.train()
    if fast_train:
        epochs = 2
    else:
        epochs = MAX_EPOCHS
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm.tqdm(dl_train, desc=f'Epoch [{epoch + 1}/{epochs}]')
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': running_loss / (progress_bar.n + 1)})

    model.eval()
    if not fast_train:
        outputs = model(X)
        y_predicted = torch.argmax(outputs, 1)
        print('Accuracy of the model on the test set: {}%'.format(100 * accuracy_score(y, y_predicted)))

    return model
    # your code here /\

# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    fc_weights = cls_model.model[10].weight.data # (128 * 3 * 10, 256) = (256, 3840) -> (256, 128, 3, 10)
    conv_layer_flatten_lin = nn.Conv2d(128, 256, kernel_size=(3, 10), stride=(1, 1), padding=(0, 0))
    conv_layer_flatten_lin.weight.data = fc_weights.view(256, 128, 3, 10)

    fc_weights = cls_model.model[13].weight.data # (256, 2) = (2, 256) -> (2, 256, 1, 1)
    conv_layer_lin = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    conv_layer_lin.weight.data = fc_weights.view(2, 256, 1, 1)

    detection_model = ModifiedModel()
    detection_model.load_state_dict(cls_model.state_dict())
    detection_model.model[9] = conv_layer_flatten_lin
    detection_model.model[10] = detection_model.model[11]
    detection_model.model[11] = detection_model.model[12]
    detection_model.model[12] = conv_layer_lin
    detection_model.model[13] = nn.Softmax(dim=1)
    #detection_model.model = nn.Sequential(*[layer for layer in detection_model.model if layer is not None])
    return detection_model
    # your code here /\

# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    
    # your code here \/
    return {}
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=1.0):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    return {}
    # your code here /\


'''
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), 
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(),
            nn.Linear(128 * 3 * 10, 256), # 3 x 10
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE)
        )
    def forward(self, X):
        #X = X.unsqueeze(1)
        return self.model(X)

class ModifiedModel(MyModel):
    def __init__(self):
        super(ModifiedModel, self).__init__()

    def forward(self, X):
        return self.model(X)[:, 1, ...]
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    fc_weights = cls_model.model[10].weight.data # (128 * 3 * 10, 256) = (256, 3840) -> (256, 128, 3, 10)
    conv_layer_flatten_lin = nn.Conv2d(128, 256, kernel_size=(3, 10), stride=(1, 1), padding=(0, 0))
    conv_layer_flatten_lin.weight.data = fc_weights.view(256, 128, 3, 10)

    fc_weights = cls_model.model[13].weight.data # (256, 2) = (2, 256) -> (2, 256, 1, 1)
    conv_layer_lin = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    conv_layer_lin.weight.data = fc_weights.view(2, 256, 1, 1)

    detection_model = ModifiedModel()
    detection_model.load_state_dict(cls_model.state_dict())
    detection_model.model[9] = conv_layer_flatten_lin
    detection_model.model[10] = detection_model.model[11]
    detection_model.model[11] = detection_model.model[12]
    detection_model.model[12] = conv_layer_lin
    detection_model.model[13] = nn.Softmax(dim=1)
    #detection_model.model = nn.Sequential(*[layer for layer in detection_model.model if layer is not None])
    return detection_model
    # your code here /\
'''