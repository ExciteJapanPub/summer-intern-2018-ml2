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
import cv2
import tensorflow as tf
from tflearn.layers.normalization import batch_normalization

num_epochs = 10
batch_size = 100
learning_rate = 3e-3

dic = {'0':'イッテンフエダイ', '1':'ハオコゼ', '2':'ゴンズイ', '3':'ソウシハギ', '4':'ギギ', '5':'アイゴ', '6':'その他'}
############################################################################################################
############################################################################################################
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

def predict_takao1(image_path):
    image = Image.open(image_file)
    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)
    model = CNN()
    model.load_state_dict(param)
    out = model(Variable(img_tensor))
    out = nn.functional.softmax(out, dim=1)
    out = out.data.numpy()

    #labels = {int(key) : value for (key,value) in dic}
    #label_takao  = np.argmax(out)
    #what = dic[str(target)]
    #prob_takao = np.max(out)
    #print(f'prediction={label_takao} probas={prob_takao}')
    return  out
############################################################################################################
############################################################################################################
CHANNEL_NUM = 3
CLASS_NUM = 7

def image2array(image_path, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # リサイズ&正規化
    image = cv2.resize(image, (image_size, image_size))
    image = image.flatten().astype(np.float32) / 255.0

    return image

def predict_takao2(image_path):
    def inference(x, keep_prob, image_size):
        # 重みを標準偏差0.1の正規分布で初期化する
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        # バイアスを0.1の定数で初期化する
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # 畳み込みを行う
        def conv2d(x, W):
            # 縦横ともにストライドは1でゼロパディングを行う
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # 畳み込み層を作成する
        def conv_layer(x, filter_size, filter_in, filter_out):
            # 重み
            W = weight_variable([filter_size, filter_size, filter_in, filter_out])
            # バイアス
            b = bias_variable([filter_out])
            # 活性化関数
            return tf.nn.relu(conv2d(x, W) + b)

        # プーリング層を作成する
        def pool_layer(x, image_size):
            # MAXプーリング（カーネルサイズ2px*2px、縦横ともにストライドは2、ゼロパディング）
            h = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 画像サイズは半分になる
            return h, int(image_size / 2)

        # 全結合層を作成する
        def dense_layer(x, dense_in, dense_out):
            # 重み
            W = weight_variable([dense_in, dense_out])
            # バイアス
            b = bias_variable([dense_out])
            # 結合
            return tf.matmul(x, W) + b

        # 平坦化されたベクトルを画像に戻す
        x_image = tf.reshape(x, [-1, image_size, image_size, CHANNEL_NUM])

        # 畳み込み層のフィルターサイズ 畳み込みは3px*3pxのカーネルサイズで行う
        filter_size = 3

        # 第1畳み込み層
        conv1_in = CHANNEL_NUM
        conv1_out = 32
        conv1 = conv_layer(x_image, filter_size, conv1_in, conv1_out)
        # 第1プーリング層
        pool1, out_size = pool_layer(conv1, image_size)

        # 第2畳み込み層
        conv2_in = conv1_out
        conv2_out = 64
        conv2 = conv_layer(pool1, filter_size, conv2_in, conv2_out)
        # batch normalization
        batch = batch_normalization(conv2)
        # 第2プーリング層
        pool2, out_size = pool_layer(batch, out_size)

        # 画像を平坦化してベクトルにする
        dimension = out_size * out_size * conv2_out
        x_flatten = tf.reshape(pool2, [-1, dimension])

        # 全結合層
        fc = dense_layer(x_flatten, dimension, conv2_out)
        # 活性化関数
        fc = tf.nn.relu(fc)

        # ドロップアウト
        drop = tf.nn.dropout(fc, keep_prob)

        # モデル出力
        y = dense_layer(drop, conv2_out, CLASS_NUM)

        return y

    # 画像の準備
    # モデル作成時の画像サイズの設定値に書き換えてください
    image_size = 56
    image = image2array(image_path, image_size)
    if image is None:
        print('not image:', image_path)
        return None

    with tf.Graph().as_default():
        # 予測
        x = np.asarray([image])
        y = tf.nn.softmax(inference(x, 1.0, image_size))
        class_label = tf.argmax(y, 1)

        # 保存の準備
        saver = tf.train.Saver()
        # セッションの作成
        sess = tf.Session()
        # セッションの開始及び初期化
        sess.run(tf.global_variables_initializer())

        # モデルの読み込み
        model = './checkpoint/fish_cnn.ckpt'
        saver.restore(sess, model)

        # 実行
        probas, predicted_label = sess.run([y, class_label])

        # 結果
        label = predicted_label[0]
        probas = [f'{p:5.3f}' for p in probas[0]]
        #print(f'prediction={label} probas={probas} image={image_path}')

        return probas
############################################################################################################
############################################################################################################
if __name__ == '__main__':
    img_path = './FISH_data/raw/images/test/0_045.jpg'
    # predict_xxxをコピーして自分の作成したモデルに合わせて編集
    proba_takao1 = predict_takao1(img_path)
    proba_takao2 = predict_takao2(img_path)
