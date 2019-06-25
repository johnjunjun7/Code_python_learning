# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#输入图片的大小28*28
batch_size = 50
n_batch = mnist.train.num_examples // batch_size
n_inputs = 28  #每一行输入28个像素
max_time = 28  #一共28行
lstm_size = 100  #有100个隐藏层神经单元，其实就是100个block
n_classes = 10 #10个类别


#初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial, name=name)

#初始化偏置值
def biases_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')#28x28
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('RNN'):
    with tf.name_scope('w_RNN'):
        w_RNN = weight_variable([lstm_size, n_classes], name='w_RNN')  # 100个 block
    with tf.name_scope('b_cRNN'):
        b_RNN = biases_variable([n_classes], name='b_RNN')  # 每一类一个偏置值


#定义RNN网络
def RNN(X, weights, biases):
    # input = [batchsize, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本的CELL
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    lstm_cell = tf.keras.layers.LSTMCell(lstm_size)
    #final_state[0]是cell_state
    #final_state[1]是hidden_sate
    #final_state[state, batch_size, cell.state_size]=【两种状态， 每一个批次的大小， 隐藏层的个数】
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    #outputs, final_state = tf.keras.layers.RNN(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


#计算RNN的返回结果
prediction = RNN(x, w_RNN, b_RNN)
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter:" + str(epoch) + "  Test Accuracy:" + str(acc) )
