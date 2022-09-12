import torch
import torch.nn as nn
from PIL import Image
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   # 继承__init__功能
        ## 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,28,28]
            nn.Conv2d(
                in_channels=3,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            # 经过卷积层 输出[16,28,28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 经过池化 输出[16,14,14] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,    # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)  # 经过池化 输出[32,7,7] 传入输出层
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # 同上
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 经过池化 输出[32,7,7] 传入输出层
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # 同上
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 经过池化 输出[32,7,7] 传入输出层
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # 同上
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 经过池化 输出[32,7,7] 传入输出层
        )
        ## 输出层
        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 70)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape[1])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x = F.log_softmax(x, dim=1)
        return x

class FruitsDataset():

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

def get_predictions(loader, model):

  model.eval()

  predicted_class = np.array([])
  targets = np.array([])

  with torch.no_grad():
    for idx, (images, labels) in enumerate(loader):

      labelsnp = labels.cpu().numpy()
      targets = np.concatenate((targets, labelsnp), axis=None)

      images = images.to(device) # torch.Size([3, 3, 224, 224]) train_dataset
      labels = labels.to(device) # torch.Size([3])

      y_predicted = model(images) # torch.Size([3, 5])
      _, index = torch.max(y_predicted, 1)

      convert = index.cpu().numpy()
      predicted_class = np.concatenate((predicted_class, convert), axis=None)

  return targets, predicted_class

def visualize_model(labels_map, num_rows, num_cols, dataset, mean, std, predicted_class):

  plt.figure(figsize=(10, 10))

  for i in range(num_rows*num_cols):
    plt.subplot(num_rows, num_cols, i+1)
    index = torch.randint(len(dataset), size=(1,)).item()
    image, label = dataset[index]
    image = image.cpu().numpy().transpose(1,2,0)
    image = image * np.array(std) + np.array(mean)
    # image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.axis('off')
    if label.item() == predicted_class[index]:
      check = 'green'
    else: check = 'red'
    print(label.item(), predicted_class[index])
    plt.title(f'Pred: {labels_map[predicted_class[index]]}', color='white',
              backgroundcolor=check, fontsize=15)

  plt.show()


def check_accuracy(loader, loader_dataset, model, labels_map):
  print('Val Acc')

  running_corrects = 0
  num_samples = 0
  absent_class = []

  num_classes = len(list(labels_map.values()))

  n_correct_class = [0 for i in range(num_classes)]
  n_class_samples = [0 for i in range(num_classes)]

  model.eval()

  with torch.no_grad():

    for images, labels in loader:

      images = images.to(device)
      labels = labels.to(device)

      y_predicted = model(images)
      _, index = torch.max(y_predicted, 1)

      running_corrects += (index == labels.data).sum()

      temp_ = index.cpu().numpy()
      num_samples += temp_.shape[0]

      temp = labels.cpu().numpy()

      for i in range(temp.shape[0]):

        label = temp[i]
        index_i = temp_[i]

        if label == index_i:
          n_correct_class[label] += 1
        n_class_samples[label] += 1

    convert = running_corrects.double()
    acc = convert / len(loader_dataset)
    print(f'Got {int(convert.item())}/{num_samples} correct samples over {acc.item() * 100:.2f}%')

    for i in range(num_classes):
      if n_class_samples[i] != 0:
        acc_ = 100 * n_correct_class[i] / n_class_samples[i]
        print(f'Accuracy of {labels_map[i]}: {acc_:.2f}%')
      else:
        absent_class.append(i)
        print(f'Class {labels_map[i]} does not have its sample in this dataset.')

  return absent_class
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
i=0
dir = './fruits-360_dataset/fruits-360/Test'

mean = torch.tensor([0.6840, 0.5786, 0.5037])
std = torch.tensor([0.3035, 0.3600, 0.3914])
batch_size = 64
test_trans = transforms.Compose([
                                  transforms.Resize((100, 100)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean, std)
])
dir = './fruits-360_dataset/fruits-360/Test'
test_csv = './fruit_test_cleaned.csv'
test_dataset = FruitsDataset(csv_file=test_csv, root_dir=dir, transform=test_trans)
print(len(test_dataset))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
model = torch.load('./model/googlenet.pt')
fruit_names = {0: 'Apple', 1: 'Apricot', 2: 'Avocado', 3: 'Banana', 4: 'Beetroot', 5: 'Blueberry',
                           6: 'Cactus', 7: 'Cantaloupe',
                           8: 'Carambula', 9: 'Cauliflower', 10: 'Cherry', 11: 'Chestnut', 12: 'Clementine',
                           13: 'Cocos', 14: 'Corn',
                           15: 'Cucumber', 16: 'Dates', 17: 'Eggplant', 18: 'Fig', 19: 'Ginger', 20: 'Granadilla',
                           21: 'Grape',
                           22: 'Grapefruit', 23: 'Guava', 24: 'Hazelnut', 25: 'Huckleberry', 26: 'Persimmon',
                           27: 'Kiwi', 28: 'Kohlrabi',
                           29: 'Kumquats', 30: 'Lemon', 31: 'Lemon Meyer', 32: 'Limes', 33: 'Lychee', 34: 'Mandarine',
                           35: 'Mango',
                           36: 'Mango Red', 37: 'Mangosteen', 38: 'Passion fruit', 39: 'Melon Piel de Sapo',
                           40: 'Mulberry', 41: 'Nectarine',
                           42: 'Nut Forest', 43: 'Pecan Nut', 44: 'Onion', 45: 'Orange', 46: 'Papaya', 47: 'Passion',
                           48: 'Peach', 49: 'Pear',
                           50: 'Pepino', 51: 'Pepper', 52: 'Physalis', 53: 'Pineapple', 54: 'Pitahaya', 55: 'Plum',
                           56: 'Pomegranate',
                           57: 'Pomelo', 58: 'Potato', 59: 'Quince', 60: 'Rambutan', 61: 'Raspberry', 62: 'Redcurrant',
                           63: 'Salak',
                           64: 'Strawberry', 65: 'Tamarillo', 66: 'Tangelo', 67: 'Tomato', 68: 'Walnut',
                           69: 'Watermelon'}
targets, preds = get_predictions(test_loader, model)
print(preds)
visualize_model(fruit_names, 3, 3, test_dataset, mean, std, preds)

print(classification_report(targets, preds))
print(confusion_matrix(targets, preds))
import seaborn as sns
cf_matrix = confusion_matrix(targets, preds, normalize='true')
plt.figure(figsize=(30,20))
y_test = sorted(fruit_names.values())
sns.heatmap(cf_matrix, annot=True,
            xticklabels = y_test, #we put this to see labels
            yticklabels = y_test
           )
plt.title('Normalized Confusion Matrix')
plt.show()
check_accuracy(test_loader, test_dataset, model, fruit_names)