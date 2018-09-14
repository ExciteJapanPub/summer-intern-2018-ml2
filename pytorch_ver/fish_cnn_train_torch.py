import torch
import torch. nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import json
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os
import sys
import pathlib
import pandas as pd
from skimage import io

num_epochs = 10
batch_size = 100
learning_rate = 3e-3

#　データのパス
DATA_PATH1 = pathlib.Path('./FISH_data1/processed/train/')
DATA_PATH2 = pathlib.Path('./FISH_data1/processed/test/')
TRAIN_IMAGES_PATH = DATA_PATH1 / 'train/'
TRAIN_LABELS_PATH = DATA_PATH1 / 'train.csv'
TEST_IMAGES_PATH = DATA_PATH2 / 'test/'
TEST_LABELS_PATH = DATA_PATH2 / 'test.csv'

#　バッチノーマライゼーション
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

# 取り込んだデータに施す処理を指定
preprocess = transforms.Compose([

    transforms.Resize((56,56)),
    transforms.ToTensor(),
    normalize])



class MyDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, transform):
        #pandasでcsvデータの読み出し
        self.image_dataframe = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        #画像データへの処理
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
    #dataframeから画像へのパスとラベルを読み出す
        label = self.image_dataframe.iat[idx, 1]
        img_name = os.path.join(self.root_dir,
                        self.image_dataframe.iloc[idx, 0])

    #画像の読み込み
        image = Image.open(img_name)
       #画像へ処理を加える

        image = self.transform(image)
        return image, label

train_dataset = MyDataset(TRAIN_LABELS_PATH, DATA_PATH1, preprocess)
test_dataset = MyDataset(TEST_LABELS_PATH, DATA_PATH2, preprocess)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# モデル定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        con1 = nn.Conv2d(3, 16, kernel_size=2, padding = 2, stride = 2) # n=56
        norm1 = nn.BatchNorm2d(16)
        relu1 = nn.ReLU()
        maxpooling1 = nn.MaxPool2d(2) # n=56
        self.layer1 = nn.Sequential(con1, norm1, relu1, maxpooling1)

        con2 = nn.Conv2d(16, 32, kernel_size=4, padding = 2, stride = 3) #n=28
        norm2 = nn.BatchNorm2d(32)
        relu2 = nn.ReLU()
        maxpooling2 = nn.MaxPool2d(2) #n=32
        self.layer2 = nn.Sequential(con2, norm2, relu2, maxpooling2)

        con3 = nn.Conv2d(32, 64, kernel_size=4, padding = 2, stride = 3) #n=15
        norm3 = nn.BatchNorm2d(64)
        relu3 = nn.ReLU()
        maxpooling3 = nn.MaxPool2d(2) #n= 14
        self.layer3 = nn.Sequential(con3, norm3, relu3, maxpooling3)

        self.fc = nn.Linear(64, 7)

    def forward(self, x):
        #print('1:', x.size())
        out = self.layer1(x)
        #print('2:', out.size())
        out = self.layer2(out)
        #print('3:', out.size())
        out = self.layer3(out)
        #print('4:', out.size())
        out = out.view(out.size(0), -1)
        #print('5:', out.size())
        out = self.fc(out)
        #print('6:', out.size())
        return out

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def train(train_loader):
    model.train()
    runnning_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion (outputs, labels)
        runnning_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    train_loss = runnning_loss/len(train_loader)
    return train_loss

def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(test_loader):

        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()
        total += labels.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc

loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(num_epochs):
    loss = train(train_loader)
    val_loss, val_acc = valid(test_loader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f'
          % (epoch, loss, val_loss, val_acc))

    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

# save the trained model
np.save('loss_list.npy', np.array(loss_list))
np.save('val_loss_list.npy', np.array(val_loss_list))
np.save('val_acc_list.npy', np.array(val_acc_list))
torch.save(model.state_dict(), 'cnn.pkl')
