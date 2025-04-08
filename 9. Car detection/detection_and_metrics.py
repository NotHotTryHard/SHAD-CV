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

'''class MyModel1(nn.Module):
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
odyn v pole voin 
        self.model = nn.Sequential(odyn v pole voin 
            convpool(1, 32),odyn v pole voin 
            convpool(32, 128),odyn v pole voin 
            nn.Flatten(),odyn v pole voin 
            nn.Linear(128 * 10 * 25, 256),odyn v pole voin 
            nn.Dropout(0.2),odyn v pole voin 
            nn.ReLU(),odyn v pole voin 
            nn.Linear(256, OUTPUT_SIZE)odyn v pole voin 
        )
    def forward(self, X):
        #X = X.unsqueeze(1)
        return self.model(X)'''

'''class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(2, 5), stride=(2, 5)),
            
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE)
        )
    def forward(self, X):
        #X = X.unsqueeze(1)
        return self.model(X)'''

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(2, 5), stride=(1, 1)),
            
            nn.Flatten(),
            nn.Linear(128 * 9 * 21, 256),
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
    optimizer = optim.Adam(model.parameters(), lr=8e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.3)

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
    
    

    epochs = 5
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
'''def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    fc_weights = cls_model.model[10].weight.data # (128 * 5 * 5, 256) = (256, 3840) -> (256, 128, 5, 5)
    conv_layer_flatten_lin = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    conv_layer_flatten_lin.weight.data = fc_weights.view(256, 128, 5, 5)

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
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    fc_weights = cls_model.model[10].weight.data # (128 * 5 * 5, 256) = (256, 3840) -> (256, 128, 5, 5)
    conv_layer_flatten_lin = nn.Conv2d(128, 256, kernel_size=(9, 21), stride=(1, 1), padding=(0, 0))
    conv_layer_flatten_lin.weight.data = fc_weights.view(256, 128, 9, 21)

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
def get_bboxes_from_logits(logitts, dh=4, dw=4):
    shape = logitts.shape
    logits = logitts.squeeze()
    if logits.ndim == 1:
        logits = logits.reshape(shape[-2], shape[-1])
    non_zero_indices = np.argwhere(logits > 0)
    bboxes = []
    for idx in non_zero_indices:
        y, x = idx
        logit_value = logits[y, x]
        bbox = [y * dh, x * dw, 40, 100, logit_value]
        bboxes.append(bbox)
    
    return bboxes

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
    threshold = -1
    dictt = {}
    for image_name in dictionary_of_images:
        image = dictionary_of_images[image_name]

        dh = 4
        dw = 4
        
        with torch.no_grad():
            logits = detection_model(torch.from_numpy(image).unsqueeze(0).unsqueeze(0))

        bboxes = get_bboxes_from_logits(logits.cpu().numpy(), dh, dw)
        dictt[image_name] = bboxes

    return dictt
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row1, col1, n_rows1, n_cols1 = first_bbox
    row2, col2, n_rows2, n_cols2 = second_bbox
    
    x_start = max(col1, col2)
    y_start = max(row1, row2)
    x_end = min(col1 + n_cols1, col2 + n_cols2)
    y_end = min(row1 + n_rows1, row2 + n_rows2)
    
    if x_start < x_end and y_start < y_end:
        A = (x_end - x_start) * (y_end - y_start)
    else:
        A = 0
    
    B = n_cols1 * n_rows1 + n_cols2 * n_rows2 - A

    return A / B
    # your code here /\


# =============================== 6 AUC ========================================
from collections import defaultdict
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
    iou_thr = 0.5
    assert pred_bboxes.keys() == gt_bboxes.keys()

    all_tp = []
    all_fp = []
    fn = 0
    for img_name in pred_bboxes.keys():
        tp = []
        fp = []
        pred_bbx = pred_bboxes[img_name]
        gt_bbx = gt_bboxes[img_name]
        sorted_bboxes = sorted(pred_bbx, key=lambda x: x[-1], reverse=True)

        for pred in sorted_bboxes:
            iou_max = 0
            gt_best = []
            for gt in gt_bbx:
                iou = calc_iou(gt, pred[0:4])
                if iou > iou_max:
                    iou_max = iou
                    gt_best = gt

            if iou_max > iou_thr:
                tp.append(pred)
                gt_bbx.remove(gt_best)
            else:
                fp.append(pred)

        fn += len(gt_bbx)
        all_tp += tp
        all_fp += fp
    #print(all_tp)
    #print(all_fp)
    #print(fn)
    list1 = sorted(all_tp + all_fp, key=lambda x: x[-1], reverse=True)
    list2 = sorted(all_tp, key=lambda x: x[-1], reverse=True)
    dict_p = defaultdict(int)
    dict_tp = defaultdict(int)
    for p in list1:
        dict_p[p[-1]] += 1
    for tp in list2:
        dict_tp[tp[-1]] += 1
    #print(dict_tp)
    #print(dict_p)
    
    precrec = []
    precrec.append((0, 1))
    cum_tp, cum_p = 0, 0
    for c in dict_p.keys():
        cum_p += dict_p[c]
        if c in dict_tp:
            cum_tp +=  dict_tp[c]
        c_recall = cum_tp / (len(all_tp) + fn)
        c_precision = cum_tp / cum_p
        dot = (c_recall, c_precision)
        precrec.append(dot)
    #print(precrec)
    auc_pr = 0
    for i in range(1, len(precrec)):
        delta = precrec[i][0] - precrec[i - 1][0]
        auc_pr += 0.5 * delta * (precrec[i][1] + precrec[i - 1][1]) 
    
    #print(auc_pr)
    return auc_pr
    # your code here /\




# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.23): #0.23
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described `using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    res = {}
    for img_name, detections in detections_dictionary.items():
        detections = sorted(detections, key=lambda x: x[-1], reverse=True)
        kept_detections = []

        while detections:
            current_detection = detections.pop(0)
            kept_detections.append(current_detection)
            
            detections = [det for det in detections if calc_iou(current_detection[:4], det[:4]) <= iou_thr]

        res[img_name] = kept_detections

    return res


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