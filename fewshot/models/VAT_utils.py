import tensorflow as tf
import numpy
import sys, os

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_float('VAT_epsilon', 1.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('VAT_num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('VAT_xi', 1e-2, "small constant for finite difference")

def entropy_y_x(logit):
    with tf.name_scope('entropy_x_y'):
        p = tf.nn.softmax(logit)
        return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))

def relative_entropy_y_x(logit):
    p = tf.nn.softmax(logit,1)
    w = tf.reduce_sum(p, 0) #/ tf.reduce_sum(p)
    # w = tf.Print(w, [w], '\n-----\n', summarize=5)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.log(tf.nn.softmax(logit, 0)  ), 1))

def reverse_relative_entropy_y_x(logit):
    p = tf.nn.softmax(logit,0)
    w = tf.reduce_sum(p, 0) / tf.reduce_sum(p)
    w = tf.Print(w, [w], '\n-----\n', summarize=5)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.log(w * tf.nn.softmax(logit, 1)), 1))

def entropy_y_x_weighted(logit):
    p = tf.nn.softmax(logit)
    # class_weights = tf.reduce_sum(p, )
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))

def logsoftmax(x, axis=1):
    with tf.name_scope('Log-of-Softmax'):
        xdev = x - tf.reduce_max(x, axis, keep_dims=True)
        lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), axis, keep_dims=True))
        # return tf.log(tf.nn.softmax(x))
    return lsm

def kl_divergence_with_logit(q_logit, p_logit):
    with tf.name_scope('KL-with-logits'):
        # tf.assert_equal(tf.shape(q_logit), tf.shape(p_logit))
        p_logit=tf.squeeze(p_logit)
        q_logit=tf.squeeze(q_logit)
        # p_logit = tf.expand_dims(p_logit, 0)
        # q_logit = tf.expand_dims(q_logit, 0)
        q = tf.nn.softmax(q_logit)
        qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
        qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp

def kl_divergence_with_logit2(q_logit, p_logit):
    with tf.name_scope('KL-with-logits'):
        # tf.assert_equal(tf.shape(q_logit), tf.shape(p_logit))
        p_logit=tf.squeeze(p_logit)
        q_logit=tf.squeeze(q_logit)
        # p_logit = tf.expand_dims(p_logit, 0)
        # q_logit = tf.expand_dims(q_logit, 0)
        q = tf.nn.softmax(q_logit)
        qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
        qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp

def get_normalized_vector(d):
    with tf.name_scope('Normalize-vector'):
        d /= (1e-12 + tf.reduce_max(tf.abs(d), list(range(1, len(d.get_shape()))), keep_dims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), list(range(1, len(d.get_shape()))) , keep_dims=True))
    return d
