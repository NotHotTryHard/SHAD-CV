import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import numpy as np
import PIL
from tqdm import tqdm 
import os
import glob
import random
from torch.utils import data
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau


path_to_train = './tests/00_test_img_input/train/'
path_to_test = './tests/00_test_img_input/test/'

NETWORK_SIZE = (256, 256)
OUTPUT_SIZE = 28
BATCH_SIZE = 32
NUM_WORKERS = 10

def get_device():
    if torch.cuda.is_available():
        print("Using the GPU 😊")
        return torch.device("cuda")
    else:
        print("Using the CPU 😞")
        return torch.device("cpu")


def flip_keypoints(keypoints):
    flipped_keypoints = keypoints.copy()
    flip_keys = (
        (0, 3),
        (1, 2),
        (4, 9),
        (5, 8),
        (6, 7),
        (11, 13)
    )
    
    for i, j in flip_keys:
        flipped_keypoints[i], flipped_keypoints[j] = keypoints[j], keypoints[i]
        
    return flipped_keypoints


DEFAULT_AUGMENTATION = A.ReplayCompose(
    [
        A.Rotate(limit=(0, 180), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        #A.RandomGamma(p=0.2), 
        #A.CLAHE(p=0.1),
        #A.GaussNoise(p=0.2),
        #A.ElasticTransform(p=0.2), 
        #A.OneOf([ 
        #   A.HueSaturationValue(p=0.3), 
        #   A.RGBShift(p=0.3)
        #], p=0.2)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TRANSFORM = A.Compose(
    [
        A.Resize(height=NETWORK_SIZE[0], width=NETWORK_SIZE[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ]
)


def DEFAULT_LABEL_TRANSFORM(original_shape, label):
    original_height, original_width = original_shape
    label[::2] /= original_width
    label[1::2] /= original_height
    return label


class ImgDataset(data.Dataset):
    def __init__(
        self,
        mode,
        gt_dict,
        img_dir,
        train_fraction=0.8,
        split_seed=42,
        transform=None,
        label_transform=None,
        aug_transform=None,
    ):
        if mode != "test":
            rng = random.Random(split_seed)
            

            paths = sorted(glob.glob(f"{img_dir}/*"))
            split = int(train_fraction * len(paths))
            rng.shuffle(paths)
            if mode == "train":
                paths = paths[:split]
            elif mode == "valid":
                paths = paths[split:]
            elif mode == "all":
                pass
            else:
                raise RuntimeError(f"Invalid mode: {mode!r}")

            labels = [gt_dict[path.split('/')[-1]] for path in paths]

            self._labels = np.array(labels)
        else:  # mode == "test"
            paths = sorted(glob.glob(f"{img_dir}/*"))
        
        self._paths = paths
        self._mode = mode
        self._len = len(paths)
        if transform is None:
            transform = DEFAULT_TRANSFORM
        self._transform = transform
        if label_transform is None:
            label_transform = DEFAULT_LABEL_TRANSFORM
        self._label_transform = label_transform
        if aug_transform is None:
            aug_transform = DEFAULT_AUGMENTATION
        self._aug_transform = aug_transform

    def __len__(self):
        self._len = len(self._paths)
        return self._len

    def __getitem__(self, index):
        if self._mode in ["train", "all"]:
            img_path = self._paths[index]
            label = self._labels[index].copy()

            image = PIL.Image.open(img_path).convert("RGB")

            augmented = self._aug_transform(image=np.array(image), keypoints=label.reshape((14, 2)))
            image = augmented['image']
            label = augmented['keypoints']

            if augmented['replay']['transforms'][2]['applied']:
                label = flip_keypoints(label)

            label = self._label_transform(image.shape[:2], label)
            transformed = self._transform(image=image)
            
            return transformed['image'], label.flatten()
        elif self._mode == "valid":
            img_path = self._paths[index]
            label = self._labels[index].copy()

            image = np.array(PIL.Image.open(img_path).convert("RGB"))

            label = self._label_transform(image.shape[:2], label.reshape((14, 2)))
            transformed = self._transform(image=image)
            
            return transformed['image'], label.flatten()
        elif self._mode == "test":
            img_path = self._paths[index]
            image = PIL.Image.open(img_path).convert("RGB")

            original_shape = image.size
            img_name = img_path.split('/')[-1]
            
            transformed = self._transform(image=np.array(image))
            return transformed['image'], np.array(original_shape), img_name

'''class MyModel(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.relu4 = nn.ReLU()

        self.fc2  = nn.Linear(64, OUTPUT_SIZE)'''
        

'''class MyModel(nn.Sequential):
    def __init__(self):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, OUTPUT_SIZE)
        )'''
        
        
class MyModel(nn.Sequential):
    def __init__(self):
        super().__init__()
        
        def convpool(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.3),
                nn.Dropout(0.3),
            )
        def conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.3)
            )
        self.convpool1 = convpool(3, 16)
        self.conv1 = conv(16, 16)
        self.convpool2 = convpool(16, 32)
        self.conv21 = conv(32, 32)
        self.conv22 = conv(32, 32)
        self.convpool3 = convpool(32, 64)
        self.conv31 = conv(64, 64)
        self.conv32 = conv(64, 64)
        self.convpool4 = convpool(64, 128)
        self.conv41 = conv(128, 128)
        self.conv42 = conv(128, 128)
        self.convpool5 = convpool(128, 256)
        self.conv5 = conv(256, 256)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE)
        )
        

    
def train_detector(train_gt, train_img_dir, fast_train=False, num_epochs=1):
    if fast_train:
        ds_train = ImgDataset(mode="train", gt_dict=train_gt, img_dir=train_img_dir)
        ds_valid = ImgDataset(mode="valid", gt_dict=train_gt, img_dir=train_img_dir)

        dl_train = data.DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )
        dl_valid = data.DataLoader(
            ds_valid,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )

        device = get_device()
        model = MyModel().to(device)
        loss_fn = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        
        
        num_epochs = 1
        for epoch in range(num_epochs):
            model = model.train()
            train_loss = []
            progress_train = tqdm(
                total=len(dl_train),
                desc=f"Epoch {epoch}",
                leave=False,
            )
            for x_batch, y_batch in dl_train:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                p_batch = model(x_batch)
                loss = loss_fn(p_batch, y_batch.float())
                train_loss.append(loss.detach())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                progress_train.update()
            progress_train.close()

            train_loss = torch.stack(train_loss).mean()
            print(
                f"Epoch {epoch},",
                f"train_loss: {train_loss.item():.8f}",
            )

            # Validation loss for scheduler
            model.eval()
            valid_loss = []
            with torch.no_grad():
                for x_batch, y_batch in dl_valid:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    p_batch = model(x_batch)
                    acc = loss_fn(p_batch, y_batch.float())
                    valid_loss.append(acc.item())
                
            valid_loss = np.mean(valid_loss) 
            print(f"Epoch {epoch}, valid_loss: {valid_loss:.8f}")
    else:
        ds_train = ImgDataset(mode="train", gt_dict=train_gt, img_dir=train_img_dir)
        ds_valid = ImgDataset(mode="valid", gt_dict=train_gt, img_dir=train_img_dir)

        dl_train = data.DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )
        dl_valid = data.DataLoader(
            ds_valid,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )

        device = get_device()   
        model = MyModel().to(device)
        sd = torch.load(
            "facepoints_model_SOTA.pt",
            map_location=get_device(),
            weights_only=True,
        )
        model.load_state_dict(sd)
        loss_fn = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.85, patience=10, verbose=True
        )
        
        valid_loss_hist = []
        for epoch in range(num_epochs):
            model = model.train()
            train_loss = []
            progress_train = tqdm(
                total=len(dl_train),
                desc=f"Epoch {epoch}",
                leave=False,
            )
            for x_batch, y_batch in dl_train:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                p_batch = model(x_batch)
                loss = loss_fn(p_batch, y_batch.float())
                train_loss.append(loss.detach())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                progress_train.update()
            progress_train.close()

            train_loss = torch.stack(train_loss).mean()
            print(
                f"Epoch {epoch},",
                f"train_loss: {train_loss.item():.8f}",
            )

            # Measure metrics on validation
            model = model.eval()
            valid_loss = []
            progress_valid = tqdm(
                total=len(dl_valid),
                desc=f"Epoch {epoch}",
                leave=False,
            )
            for x_batch, y_batch in dl_valid:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                with torch.no_grad():
                    p_batch = model(x_batch)

                acc = loss_fn(p_batch, y_batch.float())
                valid_loss.append(acc.item())

                progress_valid.update()
            progress_valid.close()

            valid_loss = np.mean(valid_loss)
            if valid_loss_hist != [] and valid_loss < np.min(valid_loss_hist):
                torch.save(model.state_dict(), 'facepoints_model_SOTA.pt')
            valid_loss_hist.append(valid_loss)
            print(
                f"Epoch {epoch},",
                f"valid_loss: {valid_loss.item():.8f}",
            )
            scheduler.step(valid_loss)
        torch.save(model.state_dict(), 'facepoints_model.pt')
    return model


def detect(detect_file_name, test_img_dir):
    device = get_device()
    model = MyModel().to(device)
    model.load_state_dict(torch.load(detect_file_name, map_location=get_device()))
    model.eval()
    
    ds_test = ImgDataset(mode="test", gt_dict=None, img_dir=test_img_dir)
    dl_test = data.DataLoader(
            ds_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )
    
    results = {}

    with torch.no_grad():
        for x_batch, old_shapes, file_names in tqdm(dl_test):
            x_batch = x_batch.to(device)

            p_batch = model(x_batch)

            for i in range(x_batch.shape[0]):
                h, w = old_shapes[i].numpy()
                predicted_marks = p_batch[i].cpu().numpy()
                
                predicted_marks[::2] *= w
                predicted_marks[1::2] *= h

                results[file_names[i]] = predicted_marks
    return results