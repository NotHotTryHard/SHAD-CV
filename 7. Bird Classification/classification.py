import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from os.path import join
from run import read_csv, save_csv
import random
import glob
import albumentations.pytorch.transforms
import albumentations as A
from PIL import Image
import os
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torchvision
from torch import nn
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchmetrics
import matplotlib.pyplot as plt 
from os.path import basename


def get_device():
    if torch.cuda.is_available():
        print("Using the GPU 😊")
        return torch.device("cuda")
    else:
        print("Using the CPU 😞")
        return torch.device("cpu")
    

NETWORK_SIZE = (480, 480)
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_CLASSES = 50
BASE_LR = 5e-5
MAX_EPOCHS = 70

path_train = './tests/00_test_img_input/train/'
path_experiment = './experiment/'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
augmentations = [
    A.Rotate(limit=25, p=0.5),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10 , p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
    A.GaussianBlur(blur_limit=(5, 9), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 20.0), p=0.3),
    A.ElasticTransform(alpha=2, sigma=2, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.25),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=12, max_width=12, p=0.3),
]

common_transforms = [
    A.Resize(*NETWORK_SIZE),
    A.ToFloat(max_value=255),
    A.Normalize(max_pixel_value=1.0, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    A.pytorch.transforms.ToTensorV2(),
]

MyFitTransform = A.Compose(augmentations + common_transforms)
MyPredictTransform = A.Compose(common_transforms)


class ImgDataset(Dataset):
    def __init__(self, img_dir, data, stage, transform=None):
        self._image_dir = img_dir
        self._data = data
        self._stage = stage

        if not transform:
            if stage in ['train', 'fit']:
                self._transform = MyFitTransform
            elif stage in ['validate', 'test', 'predict']:
                self._transform = MyPredictTransform
                
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._stage in ['fit', 'train', 'validate', 'test']:
            img_path, label = self._data[idx]
            image = Image.open(img_path).convert("RGB")
            if self._transform:
                image = self._transform(image=np.array(image))
            return image['image'], label
        elif self._stage == 'predict':
            img_path = self._data[idx]
            image = Image.open(img_path).convert("RGB")
            if self._transform:
                image = self._transform(image=np.array(image))
            return image['image']

class ImgDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir = '', 
        img_dir = '',
        gt_dict = None, 
        batch_size: int = BATCH_SIZE, 
        num_workers: int = NUM_WORKERS,
        split_seed=42,
        train_share = 0.8,
        valid_share = 0.2,
        # test_share = 0.1,
        transform=common_transforms,
        aug_transform=augmentations,
    ):
        super().__init__()
        self._data_dir = data_dir
        self._img_dir = img_dir
        self._gt_dict = gt_dict
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._split_seed = split_seed
        self._train_share = train_share
        self._valid_share = valid_share
        # self._test_share = test_share
        self._transform = transform
        self._aug_transform = aug_transform
        
    def setup(self, stage):
        if self._data_dir != '':
            img_dir = join(self._data_dir, 'images')
        elif self._data_dir == '' and self._img_dir != '':
            img_dir = self._img_dir
        paths = sorted(glob.glob(f"{img_dir}/*"))
        
        if stage in ['fit', 'train', 'validate', 'test']:
            if self._data_dir != '':
                gt_dict = read_csv(join(self._data_dir, 'gt.csv'))
            elif self._data_dir == '' and self._gt_dict is not None:
                gt_dict = self._gt_dict
            
            labels = [gt_dict[path.split('/')[-1]] for path in paths]
            
            path_train, path_val, label_train, label_val = train_test_split(paths, labels, test_size=self._valid_share, random_state=self._split_seed, stratify=labels)
            #path_train = path_train[:32] #для засыла в систему!
            #label_train = label_train[:32] #для засыла в систему!
            #path_val = path_val[:32] #для засыла в систему!
            #label_val = label_val[:32] #для засыла в систему!
            #path_train, path_test, label_train, label_test = train_test_split(paths, labels, test_size=('''self._test_share + '''self._valid_share), random_state=self._split_seed, stratify=labels)
            #path_val, path_test, label_val, label_test = train_test_split(path_test, label_test, test_size=(self._test_share / self._valid_share), random_state=self._split_seed, stratify=label_test)
            
            self._train_data = list(zip(path_train, label_train))
            self._val_data = list(zip(path_val, label_val))
            #self._test_data = list(zip(path_test, label_test))
            #print(Counter(labels))
            #print(Counter(label_train))
            #print(Counter(label_val))
            #print(Counter(label_test))

        elif stage == 'predict':
            self._pred_data = paths
            
        else:
            raise RuntimeError(f"Invalid stage: {stage!r}")
        
        self._img_dir = img_dir

    def train_dataloader(self):
        ds = ImgDataset(img_dir=self._img_dir, data=self._train_data, stage='train')
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True, drop_last=True, num_workers=self._num_workers)

    def val_dataloader(self):
        ds = ImgDataset(img_dir=self._img_dir, data=self._val_data, stage='validate')
        return DataLoader(ds, batch_size=self._batch_size, shuffle=False, drop_last=False, num_workers=self._num_workers)

    '''def test_dataloader(self):
        ds = ImgDataset(img_dir=self._img_dir, data=self._test_data, stage='test')
        return DataLoader(ds, batch_size=self._batch_size, shuffle=False, drop_last=False, num_workers=self._num_workers)'''
    
    def predict_dataloader(self):
        ds = ImgDataset(img_dir=self._img_dir, data=self._pred_data, stage='predict')
        return DataLoader(ds, batch_size=self._batch_size, shuffle=False, drop_last=False, num_workers=self._num_workers)
    
class LightningBirdClassifier(L.LightningModule):
    def __init__(self, *, transfer=False, lr=BASE_LR, model_path='./birds_model.pt', **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.transfer = transfer
        self.num_classes = 50
        self.model_path = model_path
        self.model = self.get_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
        )

    '''def get_model(self):
        if not self.transfer:
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, self.num_classes),
                nn.Softmax(dim=1)
            )
            
            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True
            return model
        else:
            self.load_model()
            return self.model
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        #print(f"🟢 Training Epoch Start: {self.current_epoch}")
        
        if self.current_epoch == 10:
            print("🔓 Разблокируем последний слой features")
            for param in self.model.features[-2:].parameters():
                param.requires_grad = True'''

    
    def get_model(self):
        if not self.transfer:
            print(1)
            checkpoint_path = './experiment/lightning_logs/version_6/checkpoints/epoch=49-valid_acc=0.832.ckpt'
            model = LightningBirdClassifier.load_from_checkpoint(
                checkpoint_path,
                lr=BASE_LR,  # При необходимости задайте параметры
                transfer=True
            ).model
            print(2)
            for param in model.parameters():
                param.requires_grad = False

            for param in model.features[-2:].parameters():
                param.requires_grad = True

            for param in model.classifier.parameters():
                param.requires_grad = True

            
            return model
        else:
            self.load_model()
            return self.model

    def save_model(self):
        torch.save(self.model, 'birds_model.pt')

    def load_model(self):
        self.model = torch.load(self.model_path, map_location=get_device())
        #self.model.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",  # Мониторинг значения валидации
                "interval": "epoch",      # Обновление каждую эпоху
                "frequency": 1            # Частота обновления
            },
        }

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")

    def _step(self, batch, kind):
        x, y = batch
        p = self.model(x)
        loss = self.loss_fn(p, y)
        acc = self.accuracy(p.argmax(dim=-1), y)

        return self._log_metrics(loss, acc, kind)

    def _log_metrics(self, loss, acc, kind):
        metrics = {}
        if loss is not None:
            metrics[f"{kind}_loss"] = loss
        if acc is not None:
            metrics[f"{kind}_acc"] = acc
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True, #для засыла в систему!
            on_step=(kind == "train"),
            on_epoch=True,
        )
        return loss
    
    def forward(self, x):
        return self.model(x)


def train_model(
    experiment_path,
    data_module,
    model,
    max_epochs=MAX_EPOCHS,
    **trainer_kwargs,
):

    callbacks = [
        L.pytorch.callbacks.TQDMProgressBar(leave=True),
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            filename="{epoch}-{valid_acc:.3f}",
            monitor="valid_acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
    ]

    trainer = L.Trainer( 
        callbacks=callbacks,  #для засыла в систему!
        max_epochs=max_epochs,
        default_root_dir=experiment_path,  #для засыла в систему!
        **trainer_kwargs,
    )
    
    data_module.setup(stage='fit')
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

    
def train_classifier(train_gt, train_img_dir, fast_train=False, num_epochs=MAX_EPOCHS):
    if fast_train:
        data_module = ImgDataModule(
            img_dir=train_img_dir,
            gt_dict=train_gt,
            batch_size=BATCH_SIZE,
            train_share=0.3,
            valid_share=0.7,
            #test_share=0.3 # если нужно тестирование
        )
        
        torch.set_float32_matmul_precision('medium')
        
        model = LightningBirdClassifier(
            transfer=True,
            lr=BASE_LR,
        )
        
        train_model(
            experiment_path=path_experiment, #УБРАТЬ ДО ЗАСЫЛА В СИСТЕМУУУУ или нетт...
            data_module=data_module,
            model=model,
            max_epochs=1,      
            accelerator="cpu" if get_device() == torch.device('cpu') else 'gpu',
            devices=1,
            precision=16,
            logger=False, #для засыла в систему!
            enable_checkpointing=False, #для засыла в систему!
        )
        
    else:
        data_module = ImgDataModule(
            img_dir=train_img_dir,
            gt_dict=train_gt,
            batch_size=BATCH_SIZE,
            train_share=0.8,
            valid_share=0.2,
            #test_share=0.02 # если нужно тестирование
        )
        
        torch.set_float32_matmul_precision('medium')
        model = LightningBirdClassifier(
            transfer=False,
            lr=BASE_LR,
        )
        
        train_model(
            experiment_path=path_experiment, #УБРАТЬ ДО ЗАСЫЛА В СИСТЕМУУУУ или нетт...
            data_module=data_module,
            model=model,
            max_epochs=num_epochs,      
            accelerator="cpu" if get_device() == torch.device('cpu') else 'gpu',
            devices=1,
            precision=16,
            enable_progress_bar=True,
            logger=True, 
            enable_checkpointing=True,
        )
        model.save_model()
        
    return model


def classify(model_path, test_img_dir):
    data_module = ImgDataModule(
            img_dir=test_img_dir,
            batch_size=BATCH_SIZE,
        )
    data_module.setup('predict')
    model = LightningBirdClassifier(
            transfer=True,
            lr=BASE_LR,
            model_path=model_path,
        )
    
    trainer = L.Trainer()
    #trainer = L.Trainer(logger=False, enable_checkpointing=False) #для засыла в систему!
 
    predictions_batches = trainer.predict(model.eval(), data_module.predict_dataloader())
    
    results = {}
    image_paths = data_module._pred_data

    predictions = torch.cat([torch.argmax(batch, dim=1) for batch in predictions_batches]).cpu().numpy()

    results = {
        basename(image_path): int(pred_class)
        for image_path, pred_class in zip(image_paths, predictions)
    }

    return results