# -*- coding: utf-8 -*-
import csv
import json
import os
import glob
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt 
import cv2
from pathlib import Path

import albumentations as A
import lightning as L
import numpy as np
import scipy
from sklearn.metrics import recall_score
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# !Этих импортов достаточно для решения данного задания

DEBUG_STATE = False # SET TO FALSE


NETWORK_SIZE = (128, 128)
CLASSES_CNT = 205
BATCH_SIZE = 32
SIMPLE_MAX_EPOCHS = 2
SIMPLE_INTERNAL_FEATURES = 1024
NUM_WORKERS = 6
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cuda:0"):
    torch.set_float32_matmul_precision('medium')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
augmentations = [
    A.Rotate(limit=15, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.25),
    A.GaussianBlur(sigma=0.5, p=0.3),
    #A.HorizontalFlip(p=0.5),
    #A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    #A.ElasticTransform(alpha=3, sigma=2, p=0.5),
    A.Perspective(p=0.25)
]

common_transforms = [
    A.Resize(*NETWORK_SIZE),
    A.ToFloat(max_value=255),
    A.Normalize(max_pixel_value=1.0, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.pytorch.transforms.ToTensorV2(),
]

MyFitTransform = A.Compose(augmentations + common_transforms)
MyPredictTransform = A.Compose(common_transforms)



class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        # список пар (путь до картинки, индекс класса)
        
        self.samples = []
        for root in root_folders:
            paths = glob.glob(f'{root}/**/*.png', recursive=True)
            self.samples += [(path, self.class_to_idx[os.path.basename(os.path.dirname(path))]) for path in paths]
        # cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        tmp = {}
        for class_num in self.classes:
            tmp[self.class_to_idx[class_num]] = []
        for idx, pair in enumerate(self.samples):
            class_num = pair[1]
            tmp[class_num].append(idx)
        self.classes_to_samples = tmp
        # аугментации + нормализация + ToTensorV2
        self.transform = MyFitTransform

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_path, class_num = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, img_path, class_num

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        class_to_idx = {class_name: data[class_name]['id'] for class_name in data}
        ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        classes = [0] * 205
        count = 0
        for class_name in data:
            count += 1
            classes[data[class_name]['id']] = class_name
        return classes[:count], class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)

class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        # список путей до картинок
        self.samples = [os.path.basename(path) for path in glob.glob(f'{root}/**/*.png', recursive=True)]
        # преобразования: ресайз + нормализация + ToTensorV2
        self.transform = MyPredictTransform
        self.targets = None
        if annotations_file is not None:
            # словарь, targets[путь до картинки] = индекс класса
            tmp = {}
            with open(annotations_file, mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                
                for row in csv_reader:
                    img_name = row[0] 
                    class_name = row[1]
                    tmp[img_name] = self.class_to_idx[class_name]
            self.targets = tmp

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_name = self.samples[index]
        image = Image.open(os.path.join(self.root, img_name)).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))
        return image['image'], img_name, self.targets[img_name] if self.targets else -1

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)
        
    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        class_to_idx = {class_name: data[class_name]['id'] for class_name in data}
        ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        classes = [0] * 205
        count = 0
        for class_name in data:
            count += 1
            classes[data[class_name]['id']] = class_name
        return classes[:count], class_to_idx


class CustomNetwork(L.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.

    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(
        self,
        features_criterion: (
            typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        internal_features: int = SIMPLE_INTERNAL_FEATURES,
        need_weights=False # ПОМЕНЯТЬ ПЕРЕД ЗАСЫЛКОЙ НА FALSE
    ):
        super().__init__()
        self.features_criterion = features_criterion
        self.model, self.classifier = self.get_model(internal_features, need_weights)
        '''if not need_weights:
            self.load_model('simple_model.pth')'''
        
        self.validation_preds = []
        self.validation_targets = []
        self.max_val_acc = 0
    def get_model(self, internal_features, need_weights):
        if need_weights:
            model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        else:
            model = torchvision.models.resnet50()
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, internal_features)
        
        classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(internal_features, CLASSES_CNT),
        )

        '''for param in model.parameters():
            param.requires_grad = False
        
        for param in model.fc.parameters():
            param.requires_grad = True
            
        for param in classifier.parameters():
            param.requires_grad = True'''
            
        return model, classifier
    
    def save_model(self, name='simple_model.pth'):
        torch.save(self.state_dict(), name)
        #full_model = torch.nn.Sequential(torch.nn.ModuleList([self.model, self.classifier]))
        #torch.save(full_model.state_dict(), 'simple_model.pth')

    def load_model(self, name='simple_model.pth'):
        self.load_state_dict(torch.load(name, map_location="cpu", weights_only=True))
        
    '''def load_model(self, path_to_model='simple_model.pth'):
        full_model = torch.nn.Sequential(torch.nn.ModuleList([self.model, self.classifier]))
        full_model.load_state_dict(torch.load(path_to_model, map_location="cpu", weights_only=True))
        self.model = full_model[0][0]
        self.classifier = full_model[0][1]'''

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        features = self.model(x)
        logits = self.classifier(features)
        return features, logits

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        self.eval()
        with torch.no_grad():
            _, logits = self(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().numpy()

    def print_batch(self, batch):
        images, _, landmarks = batch
        images = images.cpu()
        landmarks = landmarks.cpu()
        for i in range(images.shape[0]):
            image = images[i]
            if not i: 
                print(image)
            landmark = landmarks[i]
            image = (image - image.min()) / (image.max() - image.min())

            plt.figure(figsize=(6, 6))
            plt.imshow(image.numpy().transpose((1, 2, 0)))
            plt.title(landmark)

            plt.savefig(f'./debug/{i}.png')
            plt.close()

    def training_step(self, batch, batch_idx):
        """
        Один шаг обучения, который включает расчет лосса.
        """
        if DEBUG_STATE and batch_idx == 0:
            self.print_batch(batch)

        x, _, y = batch
        
        features, logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        if self.features_criterion is not None:
            features_loss = self.features_criterion(features, y)
            loss += features_loss

        self.running_train_loss += loss.item()
        self.train_batch_count += 1

        running_avg_loss = self.running_train_loss / self.train_batch_count
        self.log("running_train_loss", running_avg_loss, on_step=True, prog_bar=True)

        self.log("train_loss", loss)
        return loss

    def on_train_epoch_start(self):
        """
        Сброс значений накопленного лосса и количества батчей в начале каждой эпохи.
        """
        self.running_train_loss = 0.0
        self.train_batch_count = 0

    def validation_step(self, batch, batch_idx):
        """
        Шаг валидации. Вычисляет и логирует лосс и метрики.
        """
        x, _, y = batch
        features, logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        self.validation_preds.append(preds)
        self.validation_targets.append(y)
        
        return loss

    def on_validation_epoch_end(self):
        """
        Вызывается в конце каждой эпохи валидации для вычисления общей точности.
        """
        all_preds = torch.cat(self.validation_preds, dim=0)
        all_targets = torch.cat(self.validation_targets, dim=0)
        
        acc = (all_preds == all_targets).float().mean()
        self.log("val_acc", acc, prog_bar=True)

        print('acc', acc)
        if acc > self.max_val_acc:
            self.max_val_acc = acc
            self.save_model(name=f'simple_model_val_acc_{acc}.pth')

        self.validation_preds.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        """
        Настройка оптимизаторов.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        return optimizer


def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора с валидацией.
    """
    train_dataset = DatasetRTSD(
        root_folders=["./cropped-train/"],
        path_to_classes_json="./classes.json"
    )

    val_dataset = TestData(
        'smalltest', 
        './classes.json', 
        'smalltest_annotations.csv'
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = CustomNetwork()

    trainer = L.Trainer(
        max_epochs=SIMPLE_MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )
    
    trainer.fit(model, train_loader, val_loader)
    model.save_model()
    return model



def apply_classifier(
    model: CustomNetwork,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    test_dataset = TestData(
        root=test_folder,
        path_to_classes_json=path_to_classes_json,
        annotations_file='smalltest_annotations.csv'
        )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    results = []

    with torch.no_grad():
        for images, img_names, annotations in test_loader:
            #images = images.to(DEVICE)
            predictions = model.predict(images)
            
            for img_name, class_idx, annotation in zip(img_names, predictions, annotations.numpy()):
                class_name = test_dataset.classes[class_idx]
                results.append({"filename": img_name, "class": class_name, "gt": annotation})
    return results

def calc_metric(y_true, y_pred):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        all_cnt += 1
        if t == p:
            ok_cnt += 1
    return ok_cnt / max(1, all_cnt)

def test_classifier(
    model: CustomNetwork,
    test_folder: str = "./smalltest/",
    annotations_file: str = "./smalltest_annotations.csv",
    path_to_classes_json:str = "./classes.json",
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями
    """
    # Используем apply_classifier для получения предсказаний
    predictions = apply_classifier(model, test_folder, path_to_classes_json)
    with open(path_to_classes_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
            
    targets = {}
    with open(annotations_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            img_name = row[0]
            class_name = row[1]
            targets[img_name] = class_name

    all_preds = []
    all_trues = []
    all_types = []
    for pred in predictions:
        img_name = pred['filename']
        predicted_class = pred['class']
        #gt = pred['gt']
        if img_name in targets:
            true_class = targets[img_name]
            all_preds.append(predicted_class)
            all_trues.append(true_class)
            all_types.append(data[true_class]['type'])

    y_pred_rare = []
    y_pred_freq = []
    y_true_rare = []
    y_true_freq = []
    for i in range(len(all_types)):
        if all_types[i] == 'freq':
            y_pred_freq.append(all_preds[i])
            y_true_freq.append(all_trues[i])
        else:
            y_pred_rare.append(all_preds[i])
            y_true_rare.append(all_trues[i])

    total_metric = calc_metric(all_trues, all_preds)
    rare_metric = calc_metric(y_true_rare, y_pred_rare)
    freq_metric = calc_metric(y_true_freq, y_pred_freq)

    return total_metric, rare_metric, freq_metric



class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, icon_path: str, background_path: str = './background_images/') -> None:
        super().__init__()
        self.root = background_path
        self.icon = np.array(Image.open(icon_path).convert("RGBA"))
        ### YOUR CODE HERE

    ### Для каждого из необходимых преобразований над иконками/картинками,
    ### напишите вспомогательную функцию приблизительно следующего вида:
    ###
    ### @staticmethod
    ### def discombobulate_icon(icon: np.ndarray) -> np.ndarray:
    ###     ### YOUR CODE HERE
    ###     return ...
    ###
    ### Постарайтесь не использовать готовые библиотечные функции для
    ### аугментаций и преобразования картинок, а реализовать их
    ### "из первых принципов" на numpy

    @staticmethod
    def resize_icon(icon: np.ndarray) -> np.ndarray:
        new_size = np.random.randint(32, 129)
        new_shape = (new_size, new_size)
        return cv2.resize(icon, new_shape, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def pad_icon(icon: np.ndarray) -> np.ndarray:
        pad_size = np.random.randint(5, 20) / 100
        height, width, c = icon.shape

        padding_height = int(height * pad_size)
        padding_width = int(width * pad_size)

        padded_icon = np.pad(
            icon,
            ((padding_height, padding_height), (padding_width, padding_width), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return padded_icon

    @staticmethod
    def color_shift(icon: np.ndarray, coefs=(0.04, 0.05, 0.05)) -> np.ndarray:
        '''cmax = np.max(icon_clean, axis=-1)
        cmin = np.min(icon_clean, axis=-1)
        delta = cmax - cmin

        s = delta/cmax * 100

        RGB = (cmax - icon_clean) / delta

        new_icon = np.zeros_like(icon_clean)
        new_icon[icon_clean[0, :, :] == cmax] = RGB[2] - RGB[1]
        new_icon[icon_clean[1, :, :] == cmax] = 2 + RGB[0] - RGB[2]
        new_icon[icon_clean[2, :, :] == cmax] = 4 + RGB[1] - RGB[0]
        if cmax = '''
        icon_clean_rgb = icon[..., :3]
        icon_clean_hsv = cv2.cvtColor(icon_clean_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        limits = (180, 255, 255)
        shifts = [np.random.uniform(-coefs[i], coefs[i]) * limits[i] for i in range(3)]

        for i in range(3):
            icon_clean_hsv[..., i] = np.clip(icon_clean_hsv[..., i] + shifts[i], 0, limits[i])
        icon_clean_hsv = icon_clean_hsv.astype(np.uint8)

        icon_clean_rgb = cv2.cvtColor(icon_clean_hsv, cv2.COLOR_HSV2RGB)

        icon[..., :3] = icon_clean_rgb
        return icon

    @staticmethod
    def rotate_icon(icon: np.ndarray, deg_limit=15) -> np.ndarray:
        theta = np.random.uniform(-deg_limit, deg_limit) / 180 *  np.pi 
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        h, w, _ = icon.shape
        center = (w // 2, h // 2)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0]
        ], dtype=np.float32)

        rotated_icon = cv2.warpAffine(icon, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        return rotated_icon


    #УБРААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААТТЬ
    #УБРААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААТТЬ
    #УБРААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААТТЬ
    #УБРААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААТТЬ
    #УБРААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААААТТЬ
    @staticmethod
    def rotate_icon(icon: np.ndarray, deg_limit=15) -> np.ndarray:
        theta = np.random.uniform(-deg_limit, deg_limit)
        h, w, _ = icon.shape
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)
        rotated_icon = cv2.warpAffine(icon, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)

        return rotated_icon

    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG___SPPIZDIL COD
    @staticmethod  #WWWWWWWWWWWWWWWWWWWWWWWWWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNIIIIIIIIIIIINNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGG
    def perspective(icon: np.ndarray, offset=0.3) -> np.ndarray: 
        h, w, c = icon.shape 
        offset = int(w * offset) 
        new_icon = cv2.warpPerspective( 
            icon, 
            cv2.getPerspectiveTransform( 
                np.float32([[0, 0], [0, h], [w, 0], [w, h]]), 
                np.float32([ 
                    [np.random.uniform(0, offset), np.random.uniform(0, offset)], 
                    [np.random.uniform(0, offset), h - np.random.uniform(0, offset)], 
                    [w - np.random.uniform(0, offset), np.random.uniform(0, offset)], 
                    [w - np.random.uniform(0, offset), h - np.random.uniform(0, offset)] 
                ]) 
            ), 
            (h, w) 
        ) 
        return new_icon

    @staticmethod
    def create_motion_blur_kernel(length=15, angle=0):
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1 / length

        theta = angle / 180 *  np.pi 
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0]
        ], dtype=np.float32)
        rotated_kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))

        return rotated_kernel

    @staticmethod
    def apply_motion_blur(icon: np.ndarray, deg_limit=90, len_limit=60) -> np.ndarray:
        angle = np.random.uniform(-deg_limit, deg_limit)
        length = np.random.uniform(0, len_limit)
        height, width, channels = icon.shape

        icon_clean = icon[..., :3]

        length_normed = int(length / 512 * np.min([height, width]))
        if length_normed:
            kernel = SignGenerator.create_motion_blur_kernel(length=length_normed, angle=angle)
            icon_clean = cv2.filter2D(icon_clean, -1, kernel)
        
        icon[..., :3] = icon_clean
        return icon

    @staticmethod
    def blur_alpha_channel(icon: np.ndarray, sigma_lims=(0.3, 2.3), ksize=7) -> np.ndarray:
        mask = icon[..., 3]
        sigma = np.random.uniform(*sigma_lims)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
        
        icon[..., 3] = mask
        return icon

    def get_sample(self) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        ### YOUR CODE HERE
        icon = self.icon
        icon = self.resize_icon(icon)
        icon = self.pad_icon(icon)
        icon = self.color_shift(icon)
        icon = self.rotate_icon(icon)
        icon = self.perspective(icon)
        
        bg = self.get_bg(shape=icon.shape[:2], path=self.root)
        sign = icon[..., :3]
        alpha = icon[..., 3]
        alpha = alpha[..., np.newaxis]
        alpha = alpha / np.max(alpha) 
        icon = sign * alpha + (1 - alpha) * bg
        icon = np.concatenate((icon, alpha), axis=-1)

        icon = self.apply_motion_blur(icon)
        icon = self.blur_alpha_channel(icon)

        sign = icon[..., :3]
        alpha = icon[..., 3]
        alpha = alpha[..., np.newaxis]
        alpha = alpha / np.max(alpha) 
        icon = sign * alpha + (1 - alpha) * bg

        icon = sign * alpha + (1 - alpha) * bg
        return icon.astype(np.uint8)

    @staticmethod
    def get_bg(shape, path='./background_images/'):
        files = [f for f in os.listdir(path) if f.endswith('.jpg')]
        bg = random.choice(files)
        bg_path = os.path.join(path, bg)

        image = Image.open(bg_path)
        img_width, img_height = image.size
        
        x_start = random.randint(0, img_width - shape[1])
        y_start = random.randint(0, img_height - shape[0])
        
        cropped_image = image.crop((x_start, y_start, x_start + shape[0], y_start + shape[1]))

        return np.array(cropped_image)

def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    icon_path, path_output, path_bg, num = args
    icon_name = os.path.splitext(os.path.basename(icon_path))[0]

    k = SignGenerator(icon_path=icon_path, background_path=path_bg)
    for i in range(num):
        new_icon = k.get_sample()
        icon_folder_path = Path(path_output) / icon_name
        icon_folder_path.mkdir(parents=True, exist_ok=True)

        output_filename = os.path.join(icon_folder_path, f'{i+1}.png')

        image = Image.fromarray(new_icon)
        image.save(output_filename)
        
    


def generate_all_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    samples_per_class: int = 1000,
) -> None:
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.

    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на смеси исходных и ситетических данных.
    """
    ### YOUR CODE HERE
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        ### YOUR CODE HERE

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Функция, вычисляющая loss-функцию на признаки предпоследнего слоя нейросети.

        :param outputs: Признаки с предпоследнего слоя нейросети
        :param labels: Реальные метки объектов
        """
        ### YOUR CODE HERE


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.

    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int,
    ) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        ### YOUR CODE HERE

    def __len__(self) -> None:
        """
        Возвращает общее количество батчей.
        """
        ### YOUR CODE HERE


def train_better_model() -> torch.nn.Module:
    """
    Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки.
    """
    ### YOUR CODE HERE
    return model


class ModelWithHead(CustomNetwork):
    """
    Класс, реализующий модель с головой из kNN.

    :param n_neighbors: Количество соседей в методе ближайших соседей
    """

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()
        self.eval()
        ### YOUR CODE HERE

    def load_nn(self, nn_weights_path: str) -> None:
        """
        Функция, загружающая веса обученной нейросети.

        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE

    def load_head(self, knn_path: str) -> None:
        """
        Функция, загружающая веса kNN (с помощью pickle).

        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE

    def save_head(self, knn_path: str) -> None:
        """
        Функция, сохраняющая веса kNN (с помощью pickle).

        :param knn_path: Путь, куда надо сохранить веса kNN
        """
        ### YOUR CODE HERE

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        """
        Функция, обучающая голову kNN.

        :param indexloader: Загрузчик данных для обучения kNN
        """
        ### YOUR CODE HERE

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param imgs: батч с картинками
        """
        ### YOUR CODE HERE - предсказание нейросетевой модели
        features, model_pred = ...
        features = features / np.linalg.norm(features, axis=1)[:, None]
        ### YOUR CODE HERE - предсказание kNN на features
        knn_pred = ...
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        return  ### YOUR CODE HERE

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        ### YOUR CODE HERE


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    """
    Функция для обучения kNN-головы классификатора.

    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE


if __name__ == "__main__":
    # The following code won't be run in the test system, but you can run it
    # on your local computer with `python -m rare_traffic_sign_solution`.

    # Feel free to put here any code that you used while
    # debugging, training and testing your solution.
    
    #train_simple_classifier()
    #SignGenerator(background_path='./background_images/')
    generate_all_data( 
        output_folder='generated', 
        icons_path='icons', 
        background_path='background_images', 
        samples_per_class=1000 
    )
