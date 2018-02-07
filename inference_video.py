from __future__ import print_function

import os
import cv2
import tensorflow as tf
import numpy as np

from model import ICNet, ICNet_BN
from tools import decode_labels

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
# define setting & model configuration
ADE20k_class = 150  # predict: [0~149] corresponding to label [1~150], ignore class 0 (background)
cityscapes_class = 19

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy',
               'trainval': './model/icnet_cityscapes_trainval_90k.npy',
               'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
               'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
               'others': './model/'}
# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}
snapshot_dir = './snapshots'
SAVE_DIR = './output/'


def preprocess(img):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    img = tf.expand_dims(img, dim=0)
    return img


def check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]

    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h / 32) + 1) * 32
        new_w = (int(ori_w / 32) + 1) * 32
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)

        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape

W_RES = 2048 # width resolution
H_RES = 1024 # height resolution
C_RES = 3    # channel resolution
def main():
    num_classes = cityscapes_class
    MODEL = 'train'
    root_dir = "D:/ANNOTATION/CityScape/leftImg8bit/demoVideo/stuttgart_00"
    images = os.listdir(root_dir)

    # shape = img.shape[0:2]
    # x = tf.placeholder(dtype=tf.float32, shape=img.shape)
    shape = (H_RES,W_RES)
    x = tf.placeholder(dtype=tf.float32, shape=(H_RES,W_RES,C_RES))
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)

    model = model_config[MODEL]
    net = model({'data': img_tf}, num_classes=num_classes, filter_scale=1)
    print(net.inputs)
    for k,v in net.layers.items():
        print("{:30} => {:}".format(k,v))

    raw_output = net.layers['conv6_cls']

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = decode_labels(raw_output_up, shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()

    model_path = model_paths[MODEL]
    net.load(model_path, sess)
    print('Restore from {}'.format(model_path))

    i = 0
    speed = 1
    while i < len(images):
        #img_read = misc.imread(root_dir + '/' + images[i], mode='RGB')
        img_read = cv2.imread(root_dir + '/' + images[i])
        img_read = cv2.resize(img_read,(W_RES,H_RES))
        preds = sess.run(pred, feed_dict={x: img_read})
        input = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        out = cv2.cvtColor(cv2.convertScaleAbs(preds[0]), cv2.COLOR_BGR2RGB)
        out = cv2.addWeighted(input, 0.1, out, 0.9, 0.0)
        cv2.imshow("out", out)
        key = cv2.waitKey(5)
        if key == 27:
            break
        elif key == 56:
            i += 100
        i+=speed
        print("frame= %d key=%d" % (i,key))


if __name__ == '__main__':
    main()
