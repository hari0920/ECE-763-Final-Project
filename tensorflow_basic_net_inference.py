import numpy as np
import cv2
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './trained-models/my_test_model')

