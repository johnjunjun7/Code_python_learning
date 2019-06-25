# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

##2-1创建和使用图
# #创建常量op
# m1 = tf.constant([[3, 3]])
# m2 = tf.constant([[2], [3]])
#
# #创建矩阵乘法
# product1 = tf.matmul(m1, m2)
# print(product1)
#
# #定义会话，启动默认的图
# #调用sess的run方法来执行矩阵乘法op
# #run(product1)触发图中的三个op
# with tf.Session() as sess:
#     result = sess.run(product1)
#     print(result)




##2-2变量
# x = tf.Variable([1, 2])
# a = tf.constant([3, 3])
# state = tf.Variable(0,name='conter')
# #增加一个减法的op
# sub = tf.subtract(x, a)
# #增加一个加法的op
# add = tf.add(x, sub)
# new_value = tf.add(state, 1)
# #赋值op
# update = tf.assign(state, new_value)
#
# #所有的变量需要初始化
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))
#     print('\n')
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(state))
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(state))




##2-3 Fetch and Feed
#Fetch
# iuput1 = tf.constant(3.0)
# iuput2 = tf.constant(2.0)
# iuput3 = tf.constant(5.0)
#
# add = tf.add(iuput2, iuput3)
# mul = tf.multiply(iuput1, add)
#
# with tf.Session() as sess:
#     result = sess.run([mul, add])
#     print(result)

# #Feed
# #创建占位符
# iuput1 = tf.placeholder(tf.float32)
# iuput2 = tf.placeholder(tf.float32)
# output = tf.multiply(iuput1, iuput2)
# with tf.Session() as sess:
#     #以字典形式传入数据
#     print(sess.run(output, feed_dict={iuput1:[7.0], iuput2:[2.0]}))




##2-4 tensorflow 简单使用案例
# x_data = np.random.rand(100)
# y_data = x_data * 0.1 + 0.2
#
# #构建一个线性模型
# b = tf.Variable(1.)
# k = tf.Variable(2.)
# y = k*x_data + b
#
# #定义一个二次代价函数
# loss = tf.reduce_mean(tf.square(y_data-y))
# #定义一个梯度下降法来进行训练的优化器
# optimizer = tf.train.GradientDescentOptimizer(0.2)
# #定义最小化代价函数
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(201):
#         sess.run(train)
#         if step%20 == 0:
#             print(step, sess.run([k, b]))




# #3-1非线性回归
# x_data = np.linspace(-2, 2, 1000)[:, np.newaxis]
# noise = np.random.normal(0, 0.02, x_data.shape)
# # f = np.poly1d([0, 1, -2, 1])
# # y_data = f(x_data) + noise
# y_data = np.square(x_data) + noise
#
# #定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 1])
# y = tf.placeholder(tf.float32, [None, 1])
#
# #定义神经网络中间层
# Weight_1 = tf.Variable(tf.random_normal([1, 20]))
# biases_1 = tf.Variable(tf.zeros([1, 20]))
# Wx_plus_b_1 = tf.matmul(x, Weight_1) + biases_1
# L1 = tf.nn.tanh(Wx_plus_b_1)
#
# #定义神经网络输出层
# Weight_2 = tf.Variable(tf.random_normal([20, 1]))
# biases_2 = tf.Variable(tf.zeros([1, 1]))
# Wx_plus_b_2 = tf.matmul(L1, Weight_2) + biases_2
# prediction = tf.nn.tanh(Wx_plus_b_2)
#
# #二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# #使用梯度下降函数训练
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(2000):
#         sess.run(train_step, feed_dict={x: x_data, y: y_data})
#
#     #获得预测值
#     prediction_value = sess.run(prediction, feed_dict={x: x_data})
#     #画图
#     plt.figure()
#     plt.scatter(x_data, y_data)
#     plt.plot(x_data, prediction_value, 'r-', lw=5)
#     plt.show()




# #3-2 MNIST数据集分类简单版本
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #每个批次的大小
# batch_size = 100
# #计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# #定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
#
# # #创建神经网络隐藏层
# # W_1 = tf.Variable(tf.random_normal([784, 100]))
# # b_1 = tf.Variable(tf.zeros([1, 100]))
# # L = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
# #创建简单的神经网络
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(x, W)+b)
#
# #二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
#
# #对数似然代价函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# #使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# #计算正确率，结果存放在布尔值列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(31):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
#
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))




#4-1 Dropout方法
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #每个批次的大小
# batch_size = 100
# #计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# #定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)
#
# # #创建神经网络隐藏层
# W_1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
# b_1 = tf.Variable(tf.zeros([2000]) + 0.1)
# L1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
# L1_drop = tf.nn.dropout(L1, keep_prob)
#
# W_2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
# b_2 = tf.Variable(tf.zeros([2000]) + 0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W_2) + b_2)
# L2_drop = tf.nn.dropout(L2, keep_prob)
#
# W_3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
# b_3 = tf.Variable(tf.zeros([1000]) + 0.1)
# L3 = tf.nn.tanh(tf.matmul(L2_drop, W_3) + b_3)
# L3_drop = tf.nn.dropout(L3, keep_prob)
#
# #创建输出层的神经网络
# W = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
# b = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(L3_drop, W)+b)
#
# #二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
#
# #对数似然代价函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# #使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# #计算正确率，结果存放在布尔值列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(31):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
#
#         test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
#         print("Iter" + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy" + str(train_acc))




# #4-2优化器
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #每个批次的大小
# batch_size = 100
# #计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# #定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)
#
# # #创建神经网络隐藏层
# W_1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
# b_1 = tf.Variable(tf.zeros([500]) + 0.1)
# L1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
# L1_drop = tf.nn.dropout(L1, keep_prob)
#
# W_2 = tf.Variable(tf.truncated_normal([500, 500], stddev=0.1))
# b_2 = tf.Variable(tf.zeros([500]) + 0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W_2) + b_2)
# L2_drop = tf.nn.dropout(L2, keep_prob)
#
# #创建输出层的神经网络
# W = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
# b = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(L2_drop, W)+b)
#
# #二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
#
# #对数似然代价函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# #使用梯度下降法
# #train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# train_step = tf.train.AdadeltaOptimizer(0.2).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# #计算正确率，结果存放在布尔值列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(21):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
#
#         test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
#         print("Iter" + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy" + str(train_acc))




## 5-1 准确率达到98%
#
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #每个批次的大小
# batch_size = 100
# #计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# #定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)
# lr = tf.Variable(0.001, tf.float32)
#
# # #创建神经网络隐藏层
# W_1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
# b_1 = tf.Variable(tf.zeros([500]) + 0.1)
# L1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
# L1_drop = tf.nn.dropout(L1, keep_prob)
#
# W_2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
# b_2 = tf.Variable(tf.zeros([300]) + 0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W_2) + b_2)
# L2_drop = tf.nn.dropout(L2, keep_prob)
#
# #创建输出层的神经网络
# W = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
# b = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(L2_drop, W)+b)
#
# #二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
#
# #对数似然代价函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# #使用梯度下降法
# #train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# #计算正确率，结果存放在布尔值列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(51):
#         sess.run(tf.assign(lr, 0.001*(0.95 ** epoch)))
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
#
#         learning_rate = sess.run(lr)
#         test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         #train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
#         print("Iter" + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Learning Rate" + str(learning_rate))




#5-2 Tensorboard网络结构
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #每个批次的大小
# batch_size = 100
# #计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# #命名空间
# with tf.name_scope('Input'):
#     #定义两个placeholder
#     x = tf.placeholder(tf.float32, [None, 784], name='x_input')
#     y = tf.placeholder(tf.float32, [None, 10], name='y_input')
#
# with tf.name_scope('layer'):
#     # 创建简单的神经网络
#     with tf.name_scope('wights'):
#         W = tf.Variable(tf.zeros([784, 10]))
#     with tf.name_scope('biases'):
#         b = tf.Variable(tf.zeros([10]))
#     with tf.name_scope('wx_plus_b'):
#         wx_plus_b = tf.matmul(x, W) + b
#     with tf.name_scope('softmax'):
#         prediction = tf.nn.softmax(wx_plus_b)
# #二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
#
# #对数似然代价函数
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#
# #使用梯度下降法
# with tf.name_scope('train'):
#     train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
#         # 计算正确率，结果存放在布尔值列表中
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#     with tf.name_scope('accuracy'):
#         #求准确率
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter(r'C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\logs', sess.graph)
#     for epoch in range(1):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
#
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))
# ##使用此网址打开tensorboard==http://localhost:6006
# ## tensorboard --logdir=your_Path



##5-3 tensorboard网络运行图
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #每个批次的大小
# batch_size = 100
# #计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
#
# #参数概要
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean) #平均值
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('syddev', stddev)#标准差
#         tf.summary.scalar('max', tf.reduce_max(var))#最大值
#         tf.summary.scalar('min', tf.reduce_min(var))#最小值
#         tf.summary.histogram('histogram', var)#直方图
#
#
# #命名空间
# with tf.name_scope('Input'):
#     #定义两个placeholder
#     x = tf.placeholder(tf.float32, [None, 784], name='x_input')
#     y = tf.placeholder(tf.float32, [None, 10], name='y_input')
#
# with tf.name_scope('layer'):
#     # 创建简单的神经网络
#     with tf.name_scope('wights'):
#         W = tf.Variable(tf.zeros([784, 10]))
#         variable_summaries(W)
#     with tf.name_scope('biases'):
#         b = tf.Variable(tf.zeros([10]))
#         variable_summaries(b)
#     with tf.name_scope('wx_plus_b'):
#         wx_plus_b = tf.matmul(x, W) + b
#     with tf.name_scope('softmax'):
#         prediction = tf.nn.softmax(wx_plus_b)
# #二次代价函数
# # loss = tf.reduce_mean(tf.square(y-prediction))
#
# #对数似然代价函数
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#     tf.summary.scalar('loss', loss)
#
# #使用梯度下降法
# with tf.name_scope('train'):
#     train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
#         # 计算正确率，结果存放在布尔值列表中
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#     with tf.name_scope('accuracy'):
#         #求准确率
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         tf.summary.scalar('accuracy', accuracy)
#
# #合并所有的summary
# merged = tf.summary.merge_all()
#
#
# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter(r'C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\logs', sess.graph)
#     for epoch in range(51):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
#
#         writer.add_summary(summary, epoch)
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))
# ##使用此网址打开tensorboard==http://localhost:6006
# ## tensorboard --logdir=your_Path




# ## 5-4 Tensorboard 可视化
#
# #load dataset
# mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#
# #number of cycles
# max_steps = 1001
#
# #number of pictures
# image_num = 3000
#
# #file directory
# DIR = "C://Users//Administrator//Desktop//CodeCpp-learning//tensorflow//"
#
# #define session
# sess = tf.Session()
#
# #load pictures
# embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')
#
# #parameter summary
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram', var)
#
# #name scope
# with tf.name_scope('input'):
#     #none here means the first dimention can be any number
#     x = tf.placeholder(tf.float32, [None, 784], name='x-input')
#     #correct label
#     y = tf.placeholder(tf.float32, [None, 10], name='y-input')
#
# #show images
# with tf.name_scope('input_reshape'):
#     image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
#     tf.summary.image('input', image_shaped_input, 10)
#
# with tf.name_scope('layer'):
#     # create a simple neuronet
#     with tf.name_scope('weights'):
#         W = tf.Variable(tf.zeros([784, 10]), name='W')
#         variable_summaries(W)
#     with tf.name_scope('biases'):
#         b = tf.Variable(tf.zeros([10]), name='b')
#         variable_summaries(b)
#     with tf.name_scope('wx_plus_b'):
#         wx_plus_b = tf.matmul(x, W) + b
#     with tf.name_scope('softmax'):
#         prediction = tf.nn.softmax(wx_plus_b)
#
# with tf.name_scope('loss'):
#     # cross entropy cost
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#     tf.summary.scalar('loss', loss)
# with tf.name_scope('train'):
#     # gradient descent
#     train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
# # initialize variables
# # if 'session' in locals() and session is not None:
# #     print('Close interactive session')
# #     session.close()
#
# sess.run(tf.global_variables_initializer())
#
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
#         # result is stored in a boolean list
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction,1))
#         # argmax returns the position of the greatest number in a list
#     with tf.name_scope('accuracy'):
#         # find accuracy
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         # change correct_prediction into float 32 type
#         tf.summary.scalar('accuracy', accuracy)
#
# # create metadata file
# if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
#     tf.gfile.DeleteRecursively(DIR + 'projector/projector')
#     tf.gfile.MkDir(DIR + 'projector/projector')
# with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
#     labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
#     for i in range(image_num):
#         f.write(str(labels[i]) + '\n')
#
# # combine all summaries
# merged = tf.summary.merge_all()
#
# projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
# saver = tf.train.Saver()
# config = projector.ProjectorConfig()
# embed = config.embeddings.add()
# embed.tensor_name = embedding.name
# embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
# embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
# embed.sprite.single_image_dim.extend([28, 28])
# projector.visualize_embeddings(projector_writer, config)
#
# for i in range(max_steps):
#     # 100 samples for every batch
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
#                           run_metadata=run_metadata)
#     projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
#     projector_writer.add_summary(summary, i)
#
#     if i % 100 == 0:
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("Iter " + str(i) + ", Testing Accuracy = " + str(acc))
#
# saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
# projector_writer.close
# sess.close





##6-1 CNN神经网络
mnist = input_data.read_data_sets(r"C:\Users\Administrator\Desktop\CodeCpp-learning\tensorflow\MNIST_data", one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial)

#初始化偏置值
def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])#28x28
y = tf.placeholder(tf.float32, [None, 10])

#改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

#初始化第一个卷积层的权值和向量
W_conv1 = weight_variable([5, 5, 1, 32])#5x5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = biases_variable([32])#每个卷积核一个偏置值

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#初始化第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])#5x5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2 = biases_variable([64])#每个卷积核一个偏置值

#把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#28*28的图片第一次卷积后还是28*28，池化后变为14*14
#第二次卷积后为14*14，第二次池化后为7*7
#操作完成后得到64张7*7的平面

#初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64, 1024])#上一层有7*7*64个神经元， 全连接层有1024个神经元
b_fc1 = biases_variable([1024])

#池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#keep_drop用来表示神经元的输出概率
keep_drop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop)

#初始化第二个全连接层的权值
W_fc2 = weight_variable([1024, 10])
b_fc2 = biases_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#使用Adam优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_drop: 0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_drop: 1.0})
        print("Iter" + str(epoch) + "Test Accuracy:" + str(acc))
