import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.optim as optim
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import math
import copy
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.image as mpimg

class FruitsDataset(Dataset):

  def __init__(self, csv_file, root_dir, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_name = self.annotations.iloc[index, 0]  # Quince/r_305_100.jpg
    path = os.path.join(self.root_dir, img_name)
    img = Image.open(path).convert('RGB')
    label = torch.tensor(self.annotations.iloc[index, 1])

    if self.transform:
      img = self.transform(img)

    return (img, label)

class make_class_names(dict):

  def __init__(self):
    self = dict()

  def add(self, key, value):
    self[key] = value


def read_img_list(dir = './fruits-360_dataset/fruits-360/Training'):
    i = 0
    headerlist = ['image_name', 'target']
    with open('./fruit_train_cleaned.csv', 'w') as f:
        f.write(headerlist[0] + ',' + headerlist[1] + '\n')
        for folder in os.listdir(dir):
            folder_path = os.path.join(dir, folder)
            for image in os.listdir(folder_path):
                f.write(folder + '/' + image + ',' + str(i) + '\n')
            i += 1
    print('Done.')

def read_data():
    replaced_names = {0: 'Apple', 1: 'Apple', 2: 'Apple', 3: 'Apple', 4: 'Apple',
                   5: 'Apple', 6: 'Apple', 7: 'Apple', 8: 'Apple', 9: 'Apple',
                   10: 'Apple', 11: 'Apple', 12: 'Apple', 13: 'Apricot', 14: 'Avocado',
                   15: 'Avocado', 16: 'Banana', 17: 'Banana', 18: 'Banana', 19: 'Beetroot', 20: 'Blueberry',
                   21: 'Cactus', 22: 'Cantaloupe', 23: 'Cantaloupe', 24: 'Carambula', 25: 'Cauliflower', 26: 'Cherry',
                   27: 'Cherry', 28: 'Cherry', 29: 'Cherry', 30: 'Cherry', 31: 'Cherry',
                   32: 'Chestnut', 33: 'Clementine', 34: 'Cocos', 35: 'Corn', 36: 'Corn', 37: 'Cucumber',
                   38: 'Cucumber', 39: 'Dates', 40: 'Eggplant', 41: 'Fig', 42: 'Ginger', 43: 'Granadilla',
                   44: 'Grape', 45: 'Grape', 46: 'Grape', 47: 'Grape', 48: 'Grape',
                   49: 'Grape', 50: 'Grapefruit', 51: 'Grapefruit', 52: 'Guava', 53: 'Hazelnut', 54: 'Huckleberry',
                   55: 'Persimmon', 56: 'Kiwi', 57: 'Kohlrabi', 58: 'Kumquats', 59: 'Lemon', 60: 'Lemon Meyer',
                   61: 'Limes', 62: 'Lychee',
                   63: 'Mandarine', 64: 'Mango', 65: 'Mango Red', 66: 'Mangostan', 67: 'Maracuja',
                   68: 'Melon Piel de Sapo',
                   69: 'Mulberry', 70: 'Nectarine', 71: 'Nectarine', 72: 'Nut Forest', 73: 'Pecan Nut', 74: 'Onion',
                   75: 'Onion', 76: 'Onion', 77: 'Orange', 78: 'Papaya', 79: 'Passion', 80: 'Peach',
                   81: 'Peach', 82: 'Peach', 83: 'Pear', 84: 'Pear', 85: 'Pear', 86: 'Pear', 87: 'Pear',
                   88: 'Pear', 89: 'Pear', 90: 'Pear', 91: 'Pear', 92: 'Pepino', 93: 'Pepper',
                   94: 'Pepper', 95: 'Pepper', 96: 'Pepper', 97: 'Physalis', 98: 'Physalis',
                   99: 'Pineapple', 100: 'Pineapple', 101: 'Pitahaya', 102: 'Plum', 103: 'Plum', 104: 'Plum',
                   105: 'Pomegranate', 106: 'Pomelo', 107: 'Potato', 108: 'Potato', 109: 'Potato',
                   110: 'Potato', 111: 'Quince', 112: 'Rambutan', 113: 'Raspberry', 114: 'Redcurrant', 115: 'Salak',
                   116: 'Strawberry', 117: 'Strawberry', 118: 'Tamarillo', 119: 'Tangelo', 120: 'Tomato', 121: 'Tomato',
                   122: 'Tomato', 123: 'Tomato', 124: 'Tomato', 125: 'Tomato', 126: 'Tomato',
                   127: 'Tomato', 128: 'Tomato', 129: 'Walnut', 130: 'Watermelon'}
    dir = './fruits-360_dataset/fruits-360/Training'
    csv_file = './fruit_train_cleaned.csv'
    sample_trans = transforms.Compose([
        # transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    dataset = FruitsDataset(csv_file=csv_file, root_dir=dir, transform=sample_trans)
    print(len(dataset))
    dir = './fruits-360_dataset/fruits-360/Training'
    fruit_names = make_class_names()
    i = 0
    for folder in os.listdir(dir):
        fruit_names.add(i, replaced_names[i])
        i += 1
    print(fruit_names)
    mean, std = cal_normalize(dataset)  # dataset should be original, no resize
    print(mean, std)
    mean = torch.tensor([0.6840, 0.5786, 0.5037])
    std = torch.tensor([0.3035, 0.3600, 0.3914])

    train_trans = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomPerspective(distortion_scale=0.5),
        transforms.RandomAffine(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_trans = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dir = './fruits-360_dataset/fruits-360/Training'
    csv_file = './fruit_train_cleaned.csv'
    train_dataset = FruitsDataset(csv_file=csv_file, root_dir=dir, transform=train_trans)
    print(len(train_dataset))

    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = \
        torch.utils.data.random_split(dataset=train_dataset,
                                      lengths=[train_size, val_size])
    print(len(train_dataset), len(val_dataset))

    num_epochs = 40
    begin_epoch = 0
    batch_size = 128
    learning_rate = 0.01
    load_model = False
    load_model_file = './model/googlenet.pth'
    num_classes = len(list(set(fruit_names.values())))
    print(num_classes)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    tolerate = 3
    t = 0
    model = torchvision.models.googlenet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    n_iters1 = math.ceil(len(train_dataset) / batch_size)
    n_iters2 = math.ceil(len(val_dataset) / batch_size)

    if load_model:
        state = torch.load(load_model_file)
        best_model_wts = model.load_state_dict(state['model_state_dict'])
    else:
        best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start = time.time()

    loss_cache_train = []
    acc_cache_train = []
    loss_cache_val = []
    acc_cache_val = []

    for epoch in range(begin_epoch, num_epochs):

        model.train()
        start_epoch = time.time()

        running_corrects = 0
        running_loss = 0.0

        for idx1, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            y_predicted = model(images)
            loss = criterion(y_predicted, labels)
            #     print(loss1)
            #     loss2 = criterion(aux_y_predicted, labels)
            #     print(loss2)
            #     loss = loss1 + 0.4*loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, index = torch.max(y_predicted, 1)
            running_corrects += torch.sum(index == labels)
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)

        loss_cache_train.append(epoch_loss)
        acc_cache_train.append(epoch_acc.item())

        model.eval()

        running_corrects = 0
        running_loss = 0.0

        for idx2, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                y_predicted = model(images)
                loss = criterion(y_predicted, labels)

                _, index = torch.max(y_predicted, 1)
                running_corrects += torch.sum(index == labels)
                running_loss += loss.item() * images.size(0)

        epoch_loss_val = running_loss / len(val_dataset)
        epoch_acc_val = running_corrects / len(val_dataset)

        is_best = bool(epoch_acc_val.item() > best_acc)
        if is_best:
            best_acc = epoch_acc_val.item()
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, './model/googlenet.pth')
            t = 0
        else:
            t += 1

        loss_cache_val.append(epoch_loss_val)
        acc_cache_val.append(epoch_acc_val.item())

        end_epoch = time.time()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Step {idx1 + 1}/{n_iters1}, train Loss = {epoch_loss:.2f},  train Acc = {epoch_acc:.2f}')
        print(f'Step {idx2 + 1}/{n_iters2}, val loss = {epoch_loss_val:.2f},  val acc = {epoch_acc_val:.2f}')

        epoch_elapse = end_epoch - start_epoch
        print(f'Time spent for this epoch -----> {int(epoch_elapse // 60)}m {int(epoch_elapse % 60)}s')
        print("")

        if t > tolerate:
            print(f'Early Stopping at tolerance {tolerate}')
            break

    end = time.time()
    duration = end - start
    print(f'Training completes in {int(duration // 60)}m {int(duration % 60)}s')
    print(f'Best val Acc: {best_acc:.4f}')
    torch.save(model, './model/googlenet.pt')
    model.load_state_dict(best_model_wts)
    try:
        visualize_cost(epoch, loss_cache_train, acc_cache_train, loss_cache_val, acc_cache_val)
    except:
        print(loss_cache_train, acc_cache_train, loss_cache_val, acc_cache_val)
    with open('./model/training_curves_google.txt', 'w') as f:
        f.write(str(loss_cache_train))
        f.write(str(acc_cache_train))
        f.write(str(loss_cache_val))
        f.write(str(acc_cache_val))


def visualize_cost(num_epochs, loss_train, acc_train, loss_val, acc_val):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.subplot(1, 2, 1)

    plt.plot(range(1, num_epochs + 1), np.array(loss_train), '-o', label='train', linewidth=2)
    plt.plot(range(1, num_epochs + 1), np.array(loss_val), '-o', label='val', linewidth=2)
    plt.xlabel('$Epochs$', size=20)
    plt.ylabel('$Loss$', size=20)
    plt.legend(loc='best', fontsize=20)

    plot2 = plt.subplot(1, 2, 2)
    plot2.plot(range(1, num_epochs + 1), np.array(acc_train), '-o', label='train', linewidth=2)
    plot2.plot(range(1, num_epochs + 1), np.array(acc_val), '-o', label='val', linewidth=2)
    plot2.set_xlabel('$Epochs$', size=20)
    plot2.set_ylabel('$Acc$', size=20)
    plot2.legend(loc='best', fontsize=20)
    plot2.grid(True)

    plt.show()

def cal_normalize(dataset):
  num_channels = 3
  psum = torch.zeros(num_channels)
  psum_sq = torch.zeros(num_channels)

  for image, _ in dataset: # unpack images only
    psum += image.sum(axis=[1, 2]) # torch.Size([3, 100, 100]) C x M x N w.r.t C
    psum_sq += (image**2).sum(axis=[1, 2])

  total_pxl = len(dataset) * 100 * 100

  mean = psum / total_pxl
  var = (psum_sq / total_pxl) - (mean**2)
  std = torch.sqrt(var)

  return mean, std

if __name__ == '__main__':
    read_data()
