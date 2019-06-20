#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 下午6:57
# @Author  : future 2846440252@qq.com
# @File    : predict.py

import cv2
import os
import numpy as np
from model import densenet161
import tensorflow as tf


val_id = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,3]
trainId_to_id = {v:val_id[v] for v in range(20)}
id_to_trainId_map_func = np.vectorize(trainId_to_id.get)


if __name__ == "__main__":
    img_dir = "./imgs"
    imgs = os.listdir(img_dir)

    input = tf.placeholder(dtype=tf.float32,shape=(1,1024,2048,3))
    model = densenet161(input, num_classes=19, output_stride=8, is_training=False, reuse=False)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    tf.train.Saver().restore(sess, "./weight/model0.ckpt")
    for file in imgs:
        img_path = img_dir+"/"+file
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        max = img.max()
        min = img.min()
        img = (img - min) / (max - min)
        image = (img - np.array([0.290101, 0.328081, 0.286964])) / np.array([0.182954, 0.186566, 0.184475])
        img = image[np.newaxis,:].astype(np.float32)
        cla = sess.run(model,feed_dict={input:img})
        tt = np.argmax(cla[0], axis=-1)
        pre = np.asanyarray(tt).astype(np.uint8)
        # cv2.imshow("pre",pre)
        # cv2.waitKey()
        out_pre = id_to_trainId_map_func(pre)
        cv2.imwrite("./result/"+file,out_pre)
        print(file)
    print("end")