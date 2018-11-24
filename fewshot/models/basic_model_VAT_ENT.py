from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.refine_model import RefineModel
from fewshot.models.basic_model_VAT import BasicModelVAT
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *

l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
log = logger.get()


@RegisterModel("basic-VAT-ENT")
class BasicModelVAT_ENT(BasicModelVAT):
	def get_train_op(self, logits, y_test):
		loss, train_op = super().get_train_op(logits, y_test)
		config = self.config
		ENT_weight = config.ENT_weight

		ENT_loss = entropy_y_x(self._unlabel_logits)
		ENT_opt = tf.train.AdamOptimizer(ENT_weight * self.learn_rate, name="Entropy-optimizer")
		ENT_grads_and_vars = ENT_opt.compute_gradients(ENT_loss)
		ENT_train_op = ENT_opt.apply_gradients(ENT_grads_and_vars)

		loss += ENT_loss
		train_op = tf.group(train_op, ENT_train_op)

		self.summaries.append(tf.summary.scalar('entropy loss', ENT_loss))

		return loss, train_op