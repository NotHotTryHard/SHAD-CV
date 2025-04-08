# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import lightning as L
import numpy as np
import scipy
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


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_WORKERS = 0

FULL_TRANSFORMS = A.Compose([
    A.ToFloat(max_value=255),
    A.Resize(128, 128),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.7),
    #A.ToGray(p=0.1),
    A.CoarseDropout(max_holes=3, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
    ToTensorV2(),
])

BASE_TRANSFORMS = A.Compose([
    A.ToFloat(max_value=255),
    A.Resize(128, 128),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
    ToTensorV2(),
])


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
        transforms=FULL_TRANSFORMS,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = [] ### - список пар (путь до картинки, индекс класса)
        self.classes_to_samples = {} ### - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.transform = transforms ### - аугментации + нормализация + ToTensorV2

        sample_idx = 0
        classes_to_add = list(range(len(self.classes)))
        for folder in root_folders:
            for class_name in os.listdir(folder):
                class_idx = self.class_to_idx[class_name]
                classes_to_add.remove(class_idx)
                images = []
                class_path = os.path.join(folder, class_name)
                for image_name in os.listdir(class_path):
                    self.samples.append((os.path.join(class_path, image_name), class_idx))
                    images.append(sample_idx)
                    sample_idx += 1
                self.classes_to_samples[class_idx] = images
        
        for class_idx in classes_to_add:
            self.classes_to_samples[class_idx] = []

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        image_path, class_idx = self.samples[index]
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.transform(image=image)['image']
        return image, image_path, class_idx

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        class_to_idx = {} ### - словарь, class_to_idx['название класса'] = индекс
        classes = [] ### - массив, classes[индекс] = 'название класса'
        with open(path_to_classes_json) as f:
            input_data = json.load(f)
        for name, items in input_data.items():
            class_to_idx[name] = items['id']
            classes.append(name)
        return classes, class_to_idx

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
        transforms=BASE_TRANSFORMS,
    ) -> None:
        super().__init__()
        self.root = root
        self.samples = [] ### - список путей до картинок
        self.transform = transforms ###  преобразования: ресайз + нормализация + ToTensorV2
        
        self.targets = None
        for image_path in os.listdir(root):
            self.samples.append(image_path)
        if annotations_file is not None:
            classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
            with open(annotations_file) as f:
                input_data = f.readlines()
                self.targets = {} ### словарь, targets[путь до картинки] = индекс класса 
                for line in input_data[1:]:
                    line = line.split(',')
                    self.targets[line[0]] = class_to_idx[line[1][:-1] if '\n' in line[1] else line[1]]

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        image_path = self.samples[index]
        image = np.array(Image.open(os.path.join(self.root, image_path)).convert('RGB'))
        image = self.transform(image=image)['image']
        class_idx = None
        if self.targets is not None:
            class_idx = self.targets[image_path]
        return image, image_path, -1 if class_idx is None else class_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class CustomNetwork(torch.nn.Module):
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
        internal_features: int = 1024,
    ):
        super().__init__()
        self.net = torchvision.models.resnet50(weights=None) #torchvision.models.ResNet50_Weights.DEFAULT
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, internal_features)
        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(internal_features, CLASSES_CNT),
        )

        # for parameter in list(self.net.parameters())[:round(freeze * len(list(self.net.parameters())))]:
        #     parameter.requires_grad = False
        # for parameter in list(self.net.parameters())[round(freeze * len(list(self.net.parameters()))):]:
        #     parameter.requires_grad = True

        # for parameter in self.net.parameters():
        #     parameter.requires_grad = False
        
        # for param in self.net.fc.parameters():
        #     param.requires_grad = True

        self.features_criterion = features_criterion

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        x = self.net(x)
        return self.classifier(x), x

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        outputs, _ = self(x)
        #pred = torch.nn.functional.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1)
        return pred.cpu().numpy()


def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на исходных данных.
    """
    # dataset_trn, dataset_vld = torch.utils.data.random_split(
    #     dataset, 
    #     [
    #     int(0.85 * len(dataset)),
    #     int(0.15 * len(dataset)) + 1,
    #     ],
    #     generator=torch.Generator().manual_seed(42),
    # )
    dataset_trn = DatasetRTSD(['cropped-train'], 'classes.json', FULL_TRANSFORMS)
    dataset_vld = TestData('smalltest', 'classes.json', 'smalltest_annotations.csv')
    dataloader_trn = torch.utils.data.DataLoader(dataset_trn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
    dataloader_vld = torch.utils.data.DataLoader(dataset_vld, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    model = CustomNetwork()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.3, patience=2, verbose=True)
    
    val_loss_min = np.Inf
    val_acc_max = 0
    epochs = 20

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}:", flush=True)
        moving_avg_loss = 0.0
        moving_avg_acc = 0.0
        train_losses = []
        train_acc = []

        model.train()
        with tqdm.tqdm() as batch_bar:
            for i, data in enumerate(dataloader_trn):
                inputs, _, gt_outputs = data
                inputs = inputs.to(DEVICE)
                gt_outputs = gt_outputs.to(DEVICE)
                outputs, features = model(inputs)
                #outputs = torch.nn.functional.softmax(outputs, dim=1)
                loss = criterion(outputs, gt_outputs)#, features, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                moving_avg_loss = loss.item() if i == 0 else (0.99 * moving_avg_loss + 0.01 * loss.item())
                train_losses.append(loss.item())
                gt_vectors = torch.zeros((gt_outputs.shape[0], CLASSES_CNT)).to(DEVICE)
                gt_vectors[torch.arange(gt_outputs.shape[0]), gt_outputs] = 1
                acc = (torch.argmax(outputs, dim=1) == torch.argmax(gt_vectors, dim=1)).sum() / BATCH_SIZE
                moving_avg_acc = acc if i == 0 else (0.99 * moving_avg_acc + 0.01 * acc)
                train_acc.append(acc)
                batch_bar.set_postfix_str(f"\tloss = {moving_avg_loss :.10f}, acc = {moving_avg_acc :.10f}")
                batch_bar.update()
        
        print(f"Train loss: {np.mean(train_losses)}\n", flush=True)
        
        # VALIDATION LOOP
        val_losses = []
        val_acc = []
        model.eval()
        with torch.no_grad():
            with tqdm.tqdm() as batch_bar:
                for i, data in enumerate(dataloader_vld):
                    inputs, _, gt_outputs = data
                    inputs = inputs.to(DEVICE)
                    gt_outputs = gt_outputs.to(DEVICE)
                    outputs, _ = model(inputs)
                    #outputs = torch.nn.functional.softmax(outputs, dim=1)
                    loss = criterion(outputs, gt_outputs)#, features, epoch)
                    val_losses.append(loss.item())
                    gt_vectors = torch.zeros((gt_outputs.shape[0], CLASSES_CNT)).to(DEVICE)
                    gt_vectors[torch.arange(gt_outputs.shape[0]), gt_outputs] = 1
                    acc = (torch.argmax(outputs, dim=1) == torch.argmax(gt_vectors, dim=1)).sum() / BATCH_SIZE
                    val_acc.append(acc.cpu().numpy())
                    moving_avg_acc = acc if i == 0 else (0.99 * moving_avg_acc + 0.01 * acc)
                    batch_bar.set_postfix_str(f"\tval loss = {loss.item() :.10f}, acc = {moving_avg_acc :.10f}")
                    batch_bar.update()
        
        cur_loss = np.mean(val_losses)
        cur_acc = np.mean(val_acc)
        print(f"Validation loss: {cur_loss}, Validation acc: {cur_acc}\n", flush=True)
        scheduler.step(cur_loss)
        
        if cur_loss <= val_loss_min or cur_acc >= val_acc_max:
            torch.save(model.state_dict(), f'simple_classifier_{cur_loss :.5f}_{cur_acc :.5f}.ckpt')
            val_loss_min = cur_loss
            val_acc_max = cur_acc
        
    return model


def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    dataset = TestData(test_folder, path_to_classes_json)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    results = [] ### - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    classes, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
    for data in dataloader:
        with torch.no_grad():
            pred = model.predict(data[0])
        for _, image_path, _, idx in zip(*data, pred):
            res = {}
            res['filename'] = image_path
            res['class'] = classes[idx]
            results.append(res)
    return results


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == "all" or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


def test_classifier(
    model: torch.nn.Module,
    test_folder: str,
    annotations_file: str,
    path_to_classes_json ='classes.json',
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def read_csv(filename):
        res = {}
        with open(filename) as fhandle:
            reader = csv.DictReader(fhandle)
            for row in reader:
                res[row['filename']] = row['class']
        return res
    
    result = apply_classifier(model, test_folder, path_to_classes_json)
    output = {res['filename'] : res['class'] for res in result}
    gt = read_csv(annotations_file)
    y_pred = []
    y_true = []
    for k, v in output.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(path_to_classes_json, 'r') as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)

    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path: str) -> None:
        super().__init__()
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

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        ### YOUR CODE HERE
        icon = ...
        ### YOUR CODE HERE - случайное изображение фона
        bg = ...
        return  ### YOUR CODE HERE


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE


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
    train_simple_classifier()

