from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from run import read_csv
from tqdm import tqdm 
from os.path import abspath, dirname, join
import os

path_to_train = './tests/00_test_img_input/train/'
path_to_test = './tests/00_test_img_input/test/'


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        #print("Using CPU")
    return device


def format_number(number):
    return f"{number:05d}.jpg"


class ImDataset(Dataset):
    def __init__(self, train_gt, img_dir, train=True):
        self.data = []
        self.img_dir = img_dir
        self.train = train
        
        if self.train:
            self.train_gt = train_gt
            self.data = list(train_gt.keys())
        else:
            self.data = os.listdir(self.img_dir)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index, plot=False):
        if self.train:
            image_filename = self.data[index]
            marks = self.train_gt[image_filename]
            image = Image.open(self.img_dir + f"/{image_filename}")
            
            #добавляем каналов
            if image.mode != 'RGB':
                #print(index)
                image = image.convert('RGB')
            
            original_height, original_width = image.size
            new_size = (256, 256)
            
            transform = transforms.Compose([
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor_image = transform(image)
            resized_marks = marks.copy()
            resized_marks[::2] /= original_width
            resized_marks[1::2] /= original_height
            tensor_resized_marks = torch.Tensor(resized_marks)

            if plot:
                plt.figure(figsize=(4.5, 4.5))
                plt.imshow(image)
                plt.scatter(marks[::2], marks[1::2], color='red', s=10)  # Рисуем точки на исходном изображении
                plt.title(f'{image_filename}')
                plt.show()
            return (tensor_image, tensor_resized_marks, image.size)
        else:
            image_filename = self.data[index]
            image = Image.open(self.img_dir + f"/{image_filename}")
            
            #добавляем каналов
            if image.mode != 'RGB':
                #print(index)
                image = image.convert('RGB')
            
            original_height, original_width = image.size
            new_size = (256, 256)
            
            transform = transforms.Compose([
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor_image = transform(image)
            return (tensor_image, image.size, image_filename)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 32 * 32, 64),
            nn.ReLU(),

            nn.Linear(64, 28),
        )
        
    def forward(self, x):
        return self.model(x)


def train_detector(train_gt, train_img_dir, fast_train=False):
    if fast_train:
        train_dataset = ImDataset(train_gt, train_img_dir, train=True)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

        device = get_device()
        model = Model().to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 1

        for epoch in range(num_epochs):
            model.train()
            running_loss = None
            for images, marks, old_shapes in tqdm(train_loader):
                images, marks = images.to(device), marks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                tmp = zip(old_shapes[0], old_shapes[1])
                for i in range(len(images)):
                    print(tmp[i])
                    h, w = tmp[i]
                    outputs[i, 1::2] *= h
                    outputs[i, ::2] *= w
                loss = criterion(outputs, marks)
                loss.backward()
                optimizer.step()
                running_loss = loss.item() if running_loss is None else (0.99 * running_loss + 0.01 * loss.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    else:
        train_dataset = ImDataset(train_gt, train_img_dir, train=True)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

        device = get_device()
        model = Model().to(device)
        model.train()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 20
        for epoch in range(num_epochs):
            running_loss = None
            for images, marks, old_shapes in tqdm(train_loader):
                images, marks = images.to(device), marks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                tmp = zip(old_shapes[0], old_shapes[1])
                for i in range(len(images)):
                    print(tmp[i])
                    h, w = tmp[i]
                    outputs[i, 1::2] *= h
                    outputs[i, ::2] *= w
                loss = criterion(outputs, marks)
                loss.backward()
                optimizer.step()
                running_loss = loss.item if running_loss is None else (0.99 * running_loss + 0.01 * loss.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        #torch.save(model.state_dict(), 'model.pth')
    return model

def detect(detect_file_name, test_img_dir):
    device = get_device()
    model = Model().to(device)
    model.load_state_dict(torch.load(detect_file_name))
    model.eval()
    
    test_dataset = ImDataset(None, test_img_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
    
    results = {}

    with torch.no_grad():
        for images, old_shapes, file_names in tqdm(test_loader):
            images = images.to(device)

            outputs = model(images)

            for i in range(len(images)):
                h, w = old_shapes[i]
                predicted_marks = outputs[i].cpu().numpy()
                
                predicted_marks[::2] *= w
                predicted_marks[1::2] *= h

                results[file_names[i]] = predicted_marks
    return results