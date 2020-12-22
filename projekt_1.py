import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models, utils
from torchvision.utils import make_grid

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import copy

from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Setting seed
np.random.seed(10)
torch.manual_seed(10)

# Example of images from both classes before augmentation
display(Image.open('./dataset/with_mask/0_0_0 copy 10.jpg'))
display(Image.open('./dataset/without_mask/1.jpg'))


# Transform performing resizing, normalization and augmentation
image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
    #transforms.RandomHorizontalFlip,
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Loading all images and transforming
images_data = datasets.ImageFolder(('./dataset'), transform=image_transform)


class_names = images_data.classes

# Dividing data into test and train sets
num_train = len(images_data)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]


# Dividing data into train and validation sets
num_train2 = len(train_idx)
indices2 = list(range(num_train2))
split2 = int(np.floor(0.1 * num_train2))

np.random.shuffle(indices2)
train_idx, val_idx = indices[split2:], indices[:split2]


train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx) 
val_sampler = SubsetRandomSampler(val_idx)

# data containing information about tensor and class
train_data = DataLoader(images_data, batch_size = 30, sampler = train_sampler)
test_data = DataLoader(images_data, batch_size = 30, sampler = test_sampler)
val_data = DataLoader(images_data, batch_size = 30, sampler = val_sampler)

#-----------------
#data_examples = []
#for i in range(4):
#    data_examples.append(next(iter(train_data))[0])
#    data_examples.append(next(iter(test_data))[0])

#image_examples = utils.make_grid(data_examples, nrow=4)

#plt.imshow(image_examples[0].permute(1,2,0))







#------------------------------------------------------------------
# VGG - transfer learning

res18_mod = models.resnet18(pretrained=True)
#res18_mod_copy = res18_mod

print(res18_mod)

# Freezing parameters
for param in res18_mod.parameters():
    param.requires_grad = False


# Creating new fully connected layer
fc_features_num = res18_mod.fc.in_features
res18_mod.fc = nn.Linear(fc_features_num, 2)
res18_mod = res18_mod.to(device)


#res18_mod.fc = nn.Sequential(
#    nn.Linear(fc_features_num,2), nn.LogSoftmax(dim=1)
#)
#res18_mod = res18_mod.to(device)

# Setting criterion - CrossEntropyLoss, Stochastic Gradient Descent parameters and
# changes of learning rate
criterion = nn.CrossEntropyLoss()
optimizer_res18 = optim.SGD(res18_mod.fc.parameters(), lr = 0.001)
lr_change = lr_scheduler.StepLR(optimizer_res18, step_size = 5, gamma = 0.1)




def train_res18_model(model, criterion, optimizer, scheduler, num_epochs):

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Training mode
                data_input = train_data
            else:
                model.eval()    # Evaluation mode
                data_input = val_data

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_input:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_input)
            epoch_acc = running_corrects.double() / len(data_input)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                    
    return model, train_loss, train_acc, val_loss, val_acc         




res18_mod, train_loss, train_acc, val_loss, val_acc = train_res18_model(res18_mod, criterion, optimizer_res18, lr_change, 2)


train_data



example = next(iter(train_data))

outputs = res18_mod(example[0])
_, predictions = torch.max(outputs,1)

loss = criterion(outputs, example[1])
loss

outputs
predictions