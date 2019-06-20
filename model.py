#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 下午6:47
# @Author  : future 2846440252@qq.com
# @File    : model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim


@slim.add_arg_scope
def Upsampling(inputs,scale):
    new_image = tf.image.resize_bilinear(
        inputs,
        [tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
        align_corners=True)
    return new_image


def _DenseLayer(input,num1,num2,keep_prob,scope,dilation=1):
    with tf.variable_scope(scope, 'denseLayer', [input]) as sc:
        bn1 = slim.batch_norm(input, scope="norm1")
        relu1 = tf.nn.relu(bn1,name="relu1")
        conv1 = slim.conv2d(relu1, num1, kernel_size=1, scope='conv1')
        bn2 = slim.batch_norm(conv1, scope="norm2")
        relu2 = tf.nn.relu(bn2, name="relu2")
        relu2 = tf.pad(relu2, [[0, 0], [dilation, dilation], [dilation, dilation], [0, 0]])
        conv2 = slim.conv2d(relu2, num2, kernel_size=3, scope='conv2', rate=dilation, padding="VALID")
        dropout = slim.dropout(conv2, keep_prob=keep_prob)
    return tf.concat([input,dropout],axis=-1)


def _DenseAsppBlock(layer, num1, num2, dilation_rate, keep_prob, scope, bn_start=True):
    with tf.variable_scope(scope, 'ASPP', [layer]) as sc:
        if bn_start:
            layer = slim.batch_norm(layer, scope="norm1")
        layer = tf.nn.relu(layer,name="relu1")
        layer = slim.conv2d(layer, num1, kernel_size=1, scope='conv1')
        layer = slim.batch_norm(layer, scope="norm2")
        layer = tf.nn.relu(layer, name="relu2")
        layer = tf.pad(layer, [[0, 0], [dilation_rate, dilation_rate], [dilation_rate, dilation_rate], [0, 0]])
        layer = slim.conv2d(layer, num2, kernel_size=3, scope='conv2',rate=dilation_rate,padding="VALID")
        layer = slim.dropout(layer,keep_prob=keep_prob)
    return layer


def _DenseBlock(layer,num_layers, num1, num2, keep_prob, scope,dilation=1):
    with tf.variable_scope(scope, 'denseBlock', [layer]) as sc:
        for i in range(num_layers):
            layer = _DenseLayer(layer,num1,num2,keep_prob,"denselayer"+str(i+1),dilation=dilation)
    return layer


def _Transition(layer, scope, num_output_features, stride=2):
    with tf.variable_scope(scope, 'transition', [layer]) as sc:
        bn = slim.batch_norm(layer, scope="norm")
        relu = tf.nn.relu(bn, name="relu")
        layer = slim.conv2d(relu,num_output_features,kernel_size=1,scope="conv")
        if stride == 2:
            layer = slim.avg_pool2d(layer,kernel_size=2,scope="pool")
    return layer


def classification(layer,keep_prob,num_class, scope=None):
    with tf.variable_scope(scope, 'classification', [layer]) as sc:
        drop = slim.dropout(layer,keep_prob=keep_prob)
        conv = slim.conv2d(drop,num_class,kernel_size=1)
        output = Upsampling(conv,8)
    return output

def densenet(inputs,
             num_classes=20,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=1,
             data_format='NHWC',
             is_training=True,
             labels=None,
             reuse=None,
             output_stride=8,
             scope=None):
    assert reduction is not None
    assert growth_rate is not None
    assert num_filters is not None
    assert num_layers is not None

    feature_size = int(output_stride / 8)
    d_feature0 = 512
    d_feature1 = 128

    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    with tf.variable_scope('features', 'features', [inputs, num_classes],
                           reuse=reuse) as sc:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=False), \
             slim.arg_scope([slim.batch_norm],
                            epsilon=1e-5,scale=True,fused=False), \
             slim.arg_scope([slim.batch_norm,slim.conv2d],
                            trainable=False,activation_fn=None):
            net = inputs

            # initial convolution
            net = tf.pad(net,[[0,0],[3,3],[3,3],[0,0]])
            net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv0',padding="VALID")
            net = slim.batch_norm(net,scope="norm0")
            net = tf.nn.relu(net,name="relu0")
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, 3, stride=2, padding='VALID',scope="pool0")
            # Each denseblock
            num_features = num_filters
            # block1*****************************************************************************************************
            net = _DenseBlock(net,num_layers[0],4*growth_rate,growth_rate,dropout_rate,scope='denseblock1')
            num_features = num_features + num_layers[0] * growth_rate
            net = _Transition(net,scope="transition1",num_output_features=num_features//2)
            num_features = num_features // 2

            # block2*****************************************************************************************************
            net = _DenseBlock(net,num_layers[1],4*growth_rate,growth_rate,dropout_rate,scope='denseblock2')
            num_features = num_features + num_layers[1] * growth_rate
            net = _Transition(net,scope="transition2",num_output_features=num_features//2,stride=feature_size)
            num_features = num_features // 2

            # block3*****************************************************************************************************
            net = _DenseBlock(net,num_layers[2],4*growth_rate,growth_rate,dropout_rate,dilation=int(2 / feature_size),scope='denseblock3')
            num_features = num_features + num_layers[2] * growth_rate
            net = _Transition(net,scope="transition3",num_output_features=num_features//2,stride=1)
            num_features = num_features // 2

            # block4*****************************************************************************************************
            net = _DenseBlock(net, num_layers[3], 4 * growth_rate, growth_rate, dropout_rate,
                              dilation=int(4 / feature_size), scope='denseblock4')
            num_features = num_features + num_layers[3] * growth_rate
            net = _Transition(net, scope="transition4", num_output_features=num_features // 2, stride=1)
            num_features = num_features // 2

            # Final batch norm
            net = slim.batch_norm(net, scope="norm5")
            if feature_size > 1:
                net = Upsampling(net,2)
    with tf.variable_scope('ASPP', 'ASPP', [inputs, num_classes],
                           reuse=reuse) as sc1:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=False), \
             slim.arg_scope([slim.batch_norm],
                            epsilon=1e-5, scale=True,fused=False), \
             slim.arg_scope([slim.batch_norm, slim.conv2d],
                            trainable=False, activation_fn=None):

            ASPP_3= _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=3, keep_prob=1,
                                     scope="ASPP_3", bn_start=False)
            net = tf.concat([ASPP_3,tf.nn.relu(net)], axis=-1)

            ASPP_6 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=6, keep_prob=1,
                                     scope="ASPP_6", bn_start=True)
            net = tf.concat([ASPP_6, net], axis=-1)

            ASPP_12 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=12, keep_prob=1,
                                     scope="ASPP_12", bn_start=True)
            net = tf.concat([ASPP_12, net], axis=-1)

            ASPP_18 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=18, keep_prob=1,
                                     scope="ASPP_18", bn_start=True)
            net = tf.concat([ASPP_18, net], axis=-1)

            ASPP_24 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=24, keep_prob=1,
                                     scope="ASPP_24", bn_start=True)
            net = tf.concat([ASPP_24, net], axis=-1)

            cls = classification(net,keep_prob=1,num_class=num_classes)

    return cls


def densenet161(inputs, num_classes=19, output_stride=8, data_format='NHWC', is_training=True, reuse=None):
    return densenet(inputs,
                    num_classes=num_classes,
                    reduction=0.5,
                    growth_rate=48,
                    num_filters=96,
                    num_layers=[6,12,36,24],
                    data_format=data_format,
                    is_training=is_training,
                    output_stride=8,
                    reuse=reuse,
                    scope='densenet161')





