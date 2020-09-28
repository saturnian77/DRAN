import tensorflow as tf
import numpy as np
import os

def conv2d(x, filter_num, kernel_size, scope, activation):
    with tf.variable_scope(scope):
        a = tf.layers.conv2d(x, filter_num, kernel_size, padding = 'same')
        if activation == 'ReLU':
            a = tf.nn.relu(a)
    return a

def pReLU(_x, name):
    alpha = tf.get_variable(name+'alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    out = tf.nn.relu(_x)+alpha * (_x - abs(_x)) * 0.5
    return out

def conv_rare(x, filter_num, name):
    x3 = pReLU(x, name+'3x3_1')
    x3 = tf.layers.conv2d(x3, filter_num, 3, padding='same') # 3x3
    x3 = pReLU(x3, name+'3x3_2')
    out = tf.layers.conv2d(x3, filter_num, 3, padding='same') # 3x3
    return out


def resatt_p(img, param, name):
    inp = tf.layers.conv2d(img, param, 3, padding='same')
    inp = tf.nn.max_pool(inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    inp = pReLU(inp, name + 'resatt_1')
    inp = tf.layers.conv2d(inp, param * 2, 3, padding='same')
    inp = tf.nn.max_pool(inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    inp = pReLU(inp, name + 'resatt_2')
    inp = tf.layers.conv2d(inp, param * 4, 3, padding='same')
    out = tf.reduce_mean(inp, axis=[1, 2], keepdims=True)

    return out

def upsample_x2(inputs, feature):
    outputs = conv2d(inputs, feature * 4, [3, 3], 'upsample_mid', None)
    outputs = tf.depth_to_space(outputs, 2)
    return outputs

def upsample_x3(inputs, feature):
    outputs = conv2d(inputs, feature * 9, [3, 3], 'upsample_mid', None)
    outputs = tf.depth_to_space(outputs, 3)
    return outputs

def upsample_x2x2(inputs, feature):
    outputs = conv2d(inputs, feature * 4, [3, 3], 'upsample_mid1', None)
    outputs = tf.depth_to_space(outputs, 2)
    outputs = conv2d(outputs, feature * 4, [3, 3], 'upsample_mid2', None)
    outputs = tf.depth_to_space(outputs, 2)
    return outputs

class DRAN(object):
    def __init__(self, x, scale):
        self.conv = x
        self.scale = scale
        self.output = self.build_model()

    def build_model(self):
        print("Build Model")
            
        with tf.variable_scope('Model'):

            res_att = resatt_p(self.conv, 34, 'resatt1')
            conv = conv2d(self.conv, 64, [3, 3], 'conv_first', activation = None)

            conv_blocks = {}
            temp = conv
            conv_blocks[0] = conv
            K = 0

            for i in range(16):

                conv = conv_rare(conv,64,'conv_%d' %(i+1))
                for j in range(i + 1):
                    conv = conv + conv_blocks[j] * res_att[:, :, :, K, None]
                    K = K + 1
                conv_blocks[i+1] = conv
                
            self.conv = conv2d(conv, 64, [3, 3], 'conv_last', activation = None)
            aa = tf.add(temp,self.conv)
            
            ###########Upsampler
            if self.scale == 'x2':
                aa = upsample_x2(aa, 64)
            elif self.scale == 'x3':
                aa = upsample_x3(aa,64)
            elif self.scale == 'x4':
                aa = upsample_x2x2(aa, 64)
            ###########
            out = conv2d(aa, 3, [3, 3], 'upsampler_x2_last', None)

        return out
