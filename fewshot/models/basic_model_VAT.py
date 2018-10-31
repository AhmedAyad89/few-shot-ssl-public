
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.refine_model import RefineModel
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *


log = logger.get()

@RegisterModel("basic-VAT")
class BasicModelVAT(RefineModel):

	def get_train_op(self, logits, y_test):
		loss, train_op = super().get_train_op(logits, y_test)
		config = self.config
		VAT_weight = config.VAT_weight
		x_unlabel = tf.reshape(self.x_unlabel, [-1, config.height, config.width, config.num_channel])
		with tf.control_dependencies([self.protos]):
			vat_loss =self.virtual_adversarial_loss(x_unlabel, self.predict(VAT_run=True)[0])

		vat_opt = tf.train.AdamOptimizer(VAT_weight * self.learn_rate)
		vat_grads_and_vars = vat_opt.compute_gradients(vat_loss)
		vat_train_op = vat_opt.apply_gradients(vat_grads_and_vars)

		loss += vat_loss
		train_op = tf.group(train_op, vat_train_op)
		return loss, train_op

	def generate_virtual_adversarial_perturbation(self, x, logit, is_training=True):
		# x = tf.Print(x, [tf.shape(x)])
		with tf.name_scope('Gen-adv-perturb'):
			d = tf.random_normal(shape=tf.shape(x))
			for _ in range(FLAGS.VAT_num_power_iterations):
				d = FLAGS.VAT_xi * get_normalized_vector(d)
				logit_p = logit
				logit_m = self.predict(True, eps=d)[0]
				dist = kl_divergence_with_logit(logit_p, logit_m)
				self.summaries.append(tf.summary.scalar('VAT-loss', dist))
				grad = tf.gradients(dist, [d], aggregation_method=2, name='Adv-grads')[0]
				d = tf.stop_gradient(grad)
			return FLAGS.VAT_epsilon * get_normalized_vector(d)

	def virtual_adversarial_loss(self, x, logit, is_training=True, name="vat_loss"):
		with tf.name_scope('VAT'):
			r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, is_training=is_training)
			self.summaries.append(tf.summary.histogram('adv-norm', r_vadv))
			logit = tf.stop_gradient(logit)
			logit_p = logit
			logit_m = self.predict(True, eps=r_vadv)[0]
			loss = kl_divergence_with_logit(logit_p, logit_m)
			self.summaries.append(tf.summary.scalar('kl-loss',loss))
			# loss = tf.Print(loss, [loss], 'KL loss: ')
		return tf.identity(loss, name=name)

	def predict(self, VAT_run=False, eps=tf.constant(0.0)):
		"""See `model.py` for documentation."""
		if VAT_run:
			with tf.name_scope('VAT-predict'):
				inp = tf.add(self.x_unlabel, eps)
				h_unlbl = self.get_encoded_inputs(inp)[0]
				# h_unlbl_v = self.get_encoded_inputs(self.x_unlabel)[0]
				# h_unlbl = tf.Print(h_unlbl, [h_unlbl-h_unlbl_v], '++++\n', summarize=50)
				# h_unlbl = tf.Print(h_unlbl,[inp-self.x_unlabel, tf.reduce_sum(h_unlbl-h_unlbl_v),
				# 										tf.reduce_sum(inp-self.x_unlabel)],
				# 									 	'x_unlabel, h_unlabel-h_unlabel_v, ')
				logits = compute_logits(self.protos, h_unlbl)
		else:
			with tf.name_scope('Predict'):
				h_train, h_test = self.get_encoded_inputs(self.x_train, self.x_test)
				y_train = self.y_train
				nclasses = self.nway
				protos = self._compute_protos(nclasses, h_train, y_train)
				self.protos = protos
				logits = compute_logits(protos, h_test)
		return [logits]