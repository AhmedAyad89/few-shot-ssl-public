
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits, assign_cluster, assign_cluster_persistent, update_cluster
from fewshot.models.model import Model
from fewshot.models.refine_model import RefineModel
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity

log = logger.get()

@RegisterModel("persistent")
class PersistentModel(RefineModel):
	def __init__(self,
               config,
               nway=1,
               nshot=1,
               num_unlabel=10,
               candidate_size=10,
               is_training=True,
               dtype=tf.float32):

		if not False:
			self.training_data = tf.placeholder(
				dtype, [None, None, config.height, config.width, config.num_channel], name="training_data")
			self.training_labels = tf.placeholder(tf.int64, [None, None], name="training_labels")
			# shape = tf.shape(self._logits)
			d = config.proto_dim
			shape = [1, config.n_train_classes, config.proto_dim]
			self._persistent_protos = tf.Variable(name='persistent_protos', initial_value=np.zeros(shape), dtype=tf.float32)

			self._class_num = tf.placeholder(tf.int32, shape = 1, name="class_num")
		super().__init__(config, nway, nshot, num_unlabel, candidate_size, is_training, dtype)
		self.proto = self.get_persistent_proto()
		self.embeddings = self.embed()



	def embed(self):
		return self.get_encoded_inputs(self.training_data)[0]

	def get_persistent_proto(self):
		encoded = self.get_encoded_inputs(self.training_data)[0]
		proto = tf.reduce_mean(encoded, 1)
		return proto

	# @property
	# def proto(self):
	# 	return self.get_persistent_proto()

	def predict(self):
		"""See `model.py` for documentation."""
		nclasses = self.nway
		num_cluster_steps = self.config.num_cluster_steps
		h_train, h_unlabel, h_test = self.get_encoded_inputs(
			self.x_train, self.x_unlabel, self.x_test)
		y_train = self.y_train
		protos = self._compute_protos(nclasses, h_train, y_train)
		logits = compute_logits(protos , h_test)

		# Hard assignment for training images.
		prob_train = [None] * nclasses
		for kk in range(nclasses):
			# [B, N, 1]
			prob_train[kk] = tf.expand_dims(
				tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
		prob_train = concat(prob_train, 2)

		h_all = concat([h_train, h_unlabel], 1)

		logits_list = []
		logits_list.append(compute_logits(protos, h_test))

		# Run clustering.
		for tt in range(num_cluster_steps):
			# Label assignment.
			prob_unlabel = assign_cluster_persistent(protos, self._persistent_protos, h_unlabel)
			multipliers = 0.05 / tf.reduce_sum(prob_unlabel, axis=1)
			multipliers = tf.expand_dims(multipliers, 0)
			prob_unlabel = tf.Print(prob_unlabel, [tf.shape(multipliers),  tf.reduce_sum(prob_unlabel, axis=1)], '\n---------88888------------\n', summarize=20)

			prob_unlabel = prob_unlabel * multipliers

			entropy = tf.reduce_sum(
				-prob_unlabel * tf.log(prob_unlabel), [2], keep_dims=True)
			prob_all = concat([prob_train, prob_unlabel], 1)
			prob_all = tf.stop_gradient(prob_all)
			protos = update_cluster(h_all, prob_all)
			# protos = tf.cond(
			#     tf.shape(self._x_unlabel)[1] > 0,
			#     lambda: update_cluster(h_all, prob_all), lambda: protos)
			logits_list.append(compute_logits(protos, h_test))

		return logits_list