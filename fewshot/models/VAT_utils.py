import tensorflow as tf
import numpy
import sys, os

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_float('epsilon', 8.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('xi', 1e-2, "small constant for finite difference")


def logsoftmax(x):
    with tf.name_scope('Log-of-Softmax'):
        xdev = x - tf.reduce_max(x, 1, keep_dims=True)
        lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
        # return tf.log(tf.nn.softmax(x))
    return lsm

def kl_divergence_with_logit(q_logit, p_logit):
    with tf.name_scope('KL-with-logits'):
        # tf.assert_equal(tf.shape(q_logit), tf.shape(p_logit))
        # tf.assert_greater(tf.reduce_sum(tf.abs(q_logit[0]-p_logit[0])), 0.0)
        p_logit=tf.squeeze(p_logit)
        q_logit=tf.squeeze(q_logit)
        q = tf.nn.softmax(q_logit)
        # q = tf.Print(q, [q, p, tf.shape(p), tf.shape(q)], '\n---------------------\n::')
        qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
        qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp

def get_normalized_vector(d):
    with tf.name_scope('Normalize-vector'):
        d /= (1e-12 + tf.reduce_max(tf.abs(d), list(range(1, len(d.get_shape()))), keep_dims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), list(range(1, len(d.get_shape()))) , keep_dims=True))
    return d