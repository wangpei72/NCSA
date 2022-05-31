# Pei, Kexin, et al. "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." (2017).#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")
import sys

from load_model.network import *
from load_model.layer import *
from load_model.tutorial_models import *

sys.path.append("../")
from coverage_criteria.utils import init_coverage_tables, neuron_covered, update_coverage, model_load

def neuron_coverage(datasets, X_test, model_name, input_shape, model_path):
    # Object used to keep track of (and return) key accuracies

    samples = X_test
    n_batches = 1
    for i in range(n_batches):
        print(i)
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name,
                                                         input_shape=input_shape, model_path=model_path)
        model_layer_dict = init_coverage_tables(model)
        model_layer_dict = update_coverage(sess, x, samples, model, model_layer_dict, feed_dict, threshold=0)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()

        result = neuron_covered(model_layer_dict)[2]
        print('covered neurons percentage %d neurons %f'
              % (len(model_layer_dict), result))
        return result