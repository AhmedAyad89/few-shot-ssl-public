"""
This model applies VAT noise directly to the prototypes.
The stability penalty then can applied to only unlabelled or labeled+unlabelled samples
"""

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

@RegisterModel("basic-VAT-prototypes")
class BasicModelVAT_prototypes(BasicModelVAT):
	def get_train_op(self, logits, y_test):
		train_op, loss = RefineModel.get_train_op(self, logits, y_test)


	def virtual_adversarial_loss(self, x, logit, is_training=True, name="vat_loss"):
		with tf.name_scope('VAT'):
			r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, is_training=is_training)
			logit = tf.stop_gradient(logit)
			logit_p = logit
			logit_m = self.forward(x + r_vadv)
			loss = kl_divergence_with_logit(logit_p, logit_m)
			self.summaries.append(tf.summary.scalar('kl-loss',loss))
		return tf.identity(loss, name=name)