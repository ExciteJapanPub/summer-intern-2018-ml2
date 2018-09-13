import os
import argparse
import pickle
import pathlib
import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm
from tflearn.layers.normalization import batch_normalization

#from model.model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 28
CHANNEL_NUM = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM

CLASS_NUM = 7

DATA_PATH = pathlib.Path('./FISH_data/processed/')
TRAIN_IMAGES_PATH = DATA_PATH / 'images/train/'
TRAIN_LABELS_PATH = DATA_PATH / 'labels/train_distortion.csv'
TEST_IMAGES_PATH = DATA_PATH / 'images/test/'
TEST_LABELS_PATH = DATA_PATH / 'labels/test_distortion.csv'

CHECKPOINT = './checkpoint/fish_cnn.ckpt'

def get_args():
    '''
    init_load: データセットを作成し直すかどうか（デフォルトはFalse（過去に作成したものを使う））
    tune_num:　サンプリングを何回行うか（デフォルトは５回）
    same_param_num:　過去に同じパラメータで実行したことがある時、何回まで実行を許すか（デフォルトは３回）
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_load', action='store_true')
    parser.add_argument('--tune_num', default=1, type=int)
    parser.add_argument('--same_param_num', default=3, type=int)
    parser.add_argument('--result_dir', default='./result', type=str)
    args = parser.parse_args()
    return args

def get_hyperparams():
    '''
    lr: learning rate
    dr: dropout rate
    epoch: epoch num
    batch: batch size
    '''
    hyperparams = {
        "lr": random.choice([random.uniform(0.0005, 0.002)]),
        "dr": random.choice([0.4, 0.5, 0.6]),
        "epoch": random.choice([20, 30]),
        "batch": random.choice([32, 64]),
    }
    return hyperparams


def image2array(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # リサイズ&正規化
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.flatten().astype(np.float32) / 255.0

    return image


def load_images(images_path, labels_path):
    images = []
    labels = []

    print('\n- load', labels_path.name)

    with labels_path.open() as f:
        # 各行のファイル名と正解ラベルを取り出しリスト化する
        for line in tqdm(f):
            filename, label = line.rstrip().split(',')
            image_path = str(images_path / filename)
            image = image2array(image_path)
            if image is None:
                print('not image:', image_path)
                continue
            images.append(image)
            labels.append(int(label))

    assert len(images) == len(labels)

    return images, labels


def inference(x, keep_prob, image_size, channel_num, class_num):
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
    x_image = tf.reshape(x, [-1, image_size, image_size, channel_num])

    # 畳み込み層のフィルターサイズ 畳み込みは3px*3pxのカーネルサイズで行う
    filter_size = 3

    # 第1畳み込み層
    conv1_in = channel_num
    conv1_out = 32
    conv1 = conv_layer(x_image, filter_size, conv1_in, conv1_out)
    # batch normalization
    batch1 = batch_normalization(conv1)
    # 第1プーリング層
    pool1, out_size = pool_layer(batch1, image_size)

    # 第2畳み込み層
    conv2_in = conv1_out
    conv2_out = 32
    conv2 = conv_layer(pool1, filter_size, conv2_in, conv2_out)
    # batch normalization
    batch2 = batch_normalization(conv2)
    # 第2プーリング層
    pool2, out_size = pool_layer(batch2, out_size)

    '''
    # 第3畳み込み層
    conv3_in = conv2_out
    conv3_out = 32
    conv3 = conv_layer(pool2, filter_size, conv3_in, conv3_out)
    # batch normalization
    batch3 = batch_normalization(conv3)
    # 第3プーリング層
    pool3, out_size = pool_layer(batch3, out_size)
    '''

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
    y = dense_layer(drop, conv2_out, class_num)

    return y


def loss(onehot_labels, logits):
    # 損失関数はクロスエントロピーとする
    return tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


def training(loss_value, learning_rate):
    # 勾配降下アルゴリズム(Adam)を用いてクロスエントロピーを最小化する
    return tf.train.AdamOptimizer(learning_rate).minimize(loss_value)


def train(trains, tests, hyperparams, dir):
    # データとラベルに分ける
    train_x, train_y = trains
    test_x, test_y = tests

    # ハイパーパラメータの読み込み
    learning_rate = hyperparams['lr']
    dropout_rate = hyperparams['dr']
    epoch_num = hyperparams['epoch']
    batch_size = hyperparams['batch']

    with tf.Graph().as_default():
        # dropout率
        keep_prob = tf.placeholder(tf.float32)
        # 画像データ
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        # 出力データ
        y = inference(x, keep_prob, IMAGE_SIZE, CHANNEL_NUM, CLASS_NUM)
        #y = prot_inference(x, keep_prob, IMAGE_SIZE, CHANNEL_NUM, CLASS_NUM)
        # 正解データ
        labels = tf.placeholder(tf.int64, [None])
        y_ = tf.one_hot(labels, depth=CLASS_NUM, dtype=tf.float32)

        # 損失関数
        loss_value = loss(y_, y)
        # 学習
        train_step = training(loss_value, learning_rate)

        # 予測値と正解値を比較してbool値にする
        prediction = tf.argmax(tf.nn.softmax(y), 1)
        correct_prediction = tf.equal(prediction, labels)
        # これを正解率とする
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 保存の準備
        saver = tf.train.Saver()
        # セッションの作成
        sess = tf.Session()
        # セッションの開始及び初期化
        sess.run(tf.global_variables_initializer())

        of_count = 0
        pre_train_accuracy = 150

        # 学習
        comment = '\n- start training'
        print('\n- start training')
        for epoch in range(epoch_num):
            print(f"epoch: {epoch}/{epoch_num}")
            # ミニバッチ法
            keys = list(range(len(train_x)))
            random.shuffle(keys)
            for i in range(len(keys) // batch_size):
                if i % 100 == 0:
                    print(f"\tbatch: {i}/{len(keys) // batch_size}")
                batch_keys = keys[batch_size*i:batch_size*(i+1)]
                batch_x = np.asarray([train_x[key] for key in batch_keys])
                batch_y = np.asarray([train_y[key] for key in batch_keys])
                # 確率的勾配降下法によりクロスエントロピーを最小化するような重みを更新する
                sess.run(train_step, feed_dict={x: batch_x, labels: batch_y, keep_prob: dropout_rate})
            # 指定したepoch毎に学習データに対して精度を出す
            if epoch%1 == 0:
                train_accuracy, train_loss = sess.run([accuracy, loss_value], feed_dict={x: train_x, labels: train_y, keep_prob: 1.0})
                print(f'\n[epoch {epoch+1:02d}] acc={train_accuracy:12.10f} loss={train_loss:12.10f}')
                comment += f'\n[epoch {epoch+1:02d}] acc={train_accuracy:12.10f} loss={train_loss:12.10f}'
                print(f"abs( {pre_train_accuracy} - {train_accuracy} ) = {abs(pre_train_accuracy - train_accuracy)}")
                if (epoch > 3) and abs(pre_train_accuracy - train_accuracy) < 0.01:
                    # 以下のエポックの間、学習が進まなかったら終了
                    if of_count > 2:
                        early_stopcomment = f"Early stop: {epoch}epochs"
                        print(early_stopcomment)
                        with open(os.path.join(dir, '_early_stop.txt'), mode="w") as f:
                            f.write(early_stopcomment)
                        break
                    else:
                        of_count += 1
                else:
                    of_count = 0
                    pre_train_accuracy = train_accuracy

        # 学習が終わったら評価データに対して精度を出す
        test_accuracy, test_loss, prediction_y = sess.run([accuracy, loss_value, prediction], feed_dict={x: test_x, labels: test_y, keep_prob: 1.0})
        comment += '\n- test accuracy'
        comment += f'\nacc={test_accuracy:12.10f} loss={test_loss:12.10f}'
        print('\n- test accuracy')
        print(f'acc={test_accuracy:12.10f} loss={test_loss:12.10f}')

        comment += '\n- report'
        comment += f"{metrics.classification_report(test_y, prediction_y, target_names=[f'class {c}' for c in range(CLASS_NUM)])}"
        comment += f"{metrics.confusion_matrix(test_y, prediction_y)}"

        print('\n- report')
        print(metrics.classification_report(test_y, prediction_y, target_names=[f'class {c}' for c in range(CLASS_NUM)]))
        print(metrics.confusion_matrix(test_y, prediction_y))

        # 精度等の出力情報を保存する
        with open(os.path.join(dir, 'accuracy.txt'), mode="w") as f:
            f.write(comment)

        fn = f"_test_accuracy_{test_accuracy:12.10f}"
        with open(os.path.join(dir, f'{fn}.txt'), mode="w") as f:
            f.write(f"test accuracy is {test_accuracy:12.10f}")

        # 完成したモデルを保存する
        saver.save(sess, os.path.join(dir, "fish_cnn.ckpt"))


if __name__ == '__main__':
    '''
    実行前に以下のコマンドを実行
    $ mkdir /Users/excite3/Work/summer-intern-2018-ml2/pkl
    $ mkdir /Users/excite3/Work/summer-intern-2018-ml2/result
    
    初めての実行時は必ず　--init_load を指定して実行する
    usage: $ python fish_cnn_train_fortune.py --init_load --tune_num 1
    '''
    args = get_args()
    print(args)
    # 学習データと評価データをロードする
    '''
    # 書き込み先サイズが2GBを超える場合は下を利用. 画像サイズ128を指定した場合は2GBを超えた
    train_list = load_images(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
    test_list = load_images(TEST_IMAGES_PATH, TEST_LABELS_PATH)
    '''
    # 書き込み先サイズが2GBを超えない場合は下を利用. 画像サイズ100は問題なし
    if args.init_load:
        train_list = load_images(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
        test_list = load_images(TEST_IMAGES_PATH, TEST_LABELS_PATH)
        # pickleファイルによる出力
        with open(os.path.join('./pkl', f'train{IMAGE_SIZE}.pkl'), mode="wb") as f:
            pickle.dump(train_list, f)
        with open(os.path.join('./pkl', f'test{IMAGE_SIZE}.pkl'), mode="wb") as f:
            pickle.dump(test_list, f)
    else:
        print("load pickle file")
        with open(os.path.join('./pkl', f'train{IMAGE_SIZE}.pkl'), mode="rb") as f:
            train_list = pickle.load(f)
        with open(os.path.join('./pkl', f'test{IMAGE_SIZE}.pkl'), mode="rb") as f:
            test_list = pickle.load(f)

    # hyperparameter tuning
    for i in range(args.tune_num):
        print(f"tuning {i+1}/{args.tune_num} ...")
        # 学習開始
        hyperparams = get_hyperparams()
        print(hyperparams)
        dir = os.path.join(args.result_dir, f"lr{hyperparams['lr']}zdr{hyperparams['dr']}ep{hyperparams['epoch']}ba{hyperparams['batch']}")
        comment = f"learning_rate: {hyperparams['lr']}, dropout_rate: {hyperparams['dr']}, epoch: {hyperparams['epoch']}, batch size: {hyperparams['batch']}"
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            for i in range(args.same_param_num):
                tmp_dir = os.path.join(args.result_dir,f"{i}_lr{hyperparams['lr']}zdr{hyperparams['dr']}ep{hyperparams['epoch']}ba{hyperparams['batch']}")
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                else:
                    with open(os.path.join('./result', '_memo.txt'), mode="w") as f:
                        comment = f"\ncomment is over {args.same_param_num} times."
                        f.write(comment)
                    continue
                dir = tmp_dir
        with open(os.path.join(dir, '_memo.txt'), mode="w") as f:
            f.write(comment)
        train(train_list, test_list, hyperparams, dir)
