import numpy as np
import tensorflow as tf
import fish_cnn_train1 as train

dic = {'0':'イッテンフエダイ', '1':'ハオコゼ', '2':'ゴンズイ', '3':'ソウシハギ', '4':'ギギ', '5':'アイゴ', '6':'その他'}
def predict(image_path):
    # 画像の準備
    image = train.image2array(image_path)
    if image is None:
        print('not image:', image_path)
        return None

    with tf.Graph().as_default():
        # 予測
        x = np.asarray([image])
        y = tf.nn.softmax(train.inference(x, 1.0))
        class_label = tf.argmax(y, 1)

        # 保存の準備
        saver = tf.train.Saver()
        # セッションの作成
        sess = tf.Session()
        # セッションの開始及び初期化
        sess.run(tf.global_variables_initializer())

        # モデルの読み込み
        saver.restore(sess, train.CHECKPOINT)

        # 実行
        probas, predicted_label = sess.run([y, class_label])

        # 結果
        label = predicted_label[0]
        probas = [f'{p:5.3f}' for p in probas[0]]
        print(f'prediction={dic[str(label)]} probas={probas} image={image_path}')

        return label


if __name__ == '__main__':
    # 予測
    path_list = [
        '0_045.jpg',
        '1_046.jpg',
        '2_053.jpg',
        '2_054.jpg',
        '3_048.jpg',
        '5_049.jpg',
        '6_068.jpg',

    ]
    for path in path_list:
        predict(str(train.TEST_IMAGES_PATH / path))
