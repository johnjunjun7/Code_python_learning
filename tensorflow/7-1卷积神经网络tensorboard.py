# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial, name=name)

#初始化偏置值
def biases_variable(shape, name):
    # initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')#28x28
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x-image'):
        #改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x-iamge')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和向量
    with tf.name_scope('w_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], name='w_conv1')  # 5x5的采样窗口，32个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = biases_variable([32], name='b_conv1')  # 每个卷积核一个偏置值

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        con2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(con2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('Conv2'):
    #初始化第二个卷积层
    with tf.name_scope('w_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='w_conv2')#5x5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = biases_variable([64], name='b_conv2')#每个卷积核一个偏置值

    #把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

#28*28的图片第一次卷积后还是28*28，池化后变为14*14
#第二次卷积后为14*14，第二次池化后为7*7
#操作完成后得到64张7*7的平面

with tf.name_scope('fc1'):
    #初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')#上一层有7*7*64个神经元， 全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = biases_variable([1024], name='b_fc1')
    with tf.name_scope('h_pool2_flat'):
        #池化层2的输出扁平化为1维
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    with tf.name_scope('Wx_plus_b1'):
        Wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        #求第一个全连接层的输出
        h_fc1 = tf.nn.relu(Wx_plus_b1)
    with tf.name_scope('keep_drop'):
        #keep_drop用来表示神经元的输出概率
        keep_drop = tf.placeholder(tf.float32, name='keep_drop')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层的权值
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='w_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = biases_variable([10], name='b_fc2')
    with tf.name_scope('Wx_plus_b2'):
        Wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # #计算输出
        prediction = tf.nn.softmax(Wx_plus_b2)

with tf.name_scope('cross_entropy'):
    #交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    #使用Adam优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    #结果存放在一个布尔列表中
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        #准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(r'C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\logs1\train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(r'C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\logs1\test',
                                        sess.graph)
    for i in range(3501):
        # 100 samples for every batch
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x: batch_xs, y: batch_ys, keep_drop: 0.7})
        #记录训练集参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_drop: 1.0})
        train_writer.add_summary(summary, i)
        #记录测试集参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_drop: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_drop: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_drop: 1.0})
            print("Iter:" + str(i) + "  Test Accuracy:" + str(test_acc) + "  Training Accuracy:" + str(train_acc))
