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

dic = {'0':'イッテンフエダイ', '1':'ハオコゼ', '2':'ゴンズイ', '3':'ソウシハギ', '4':'ギギ', '5':'アイゴ', '6':'その他'}
param = torch.load('cnn.pkl')
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

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize(56),
    transforms.ToTensor(),
    normalize])

def predict(image_file):
    image = Image.open(image_file)
    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)
    model = CNN()
    model.load_state_dict(param)
    out = model(Variable(img_tensor))
    out = nn.functional.softmax(out, dim=1)
    out = out.data.numpy()

    #labels = {int(key) : value for (key,value) in dic}
    target  = np.argmax(out)
    what = dic[str(target)]
    prob = np.max(out)
    print(f'prediction={what} probas={prob} image={image_file}')
    return  what, prob
if __name__ == '__main__':
    # 予測
    path_list = [
        '0_045_000.jpg',
    ]
    for path in path_list:
        predict('./FISH_data1/processed/test/'+ path)
