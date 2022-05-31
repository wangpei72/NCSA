import shutil
import sys

from tensorflow.python.platform import flags

from coverage_criteria.sc import sc

sys.path.append("../")

import os

from load_model.tutorial_models import *

FLAGS = flags.FLAGS

tf.reset_default_graph()
X_train = np.load('../data/student-aif360preproc-done/features-train.npy')
X_test = np.load('../data/student-aif360preproc-done/test/1/featrures-test-15%.npy')
input_shape = (None, X_test.shape[1])
model_name = "dnn5"
samples_path = "test"
datasets = "student"
model_path = '../original_models/'
if os.path.exists('../suprise/'):
    shutil.rmtree('../suprise/')

# neuron_coverage(datasets, X_test, model_name, input_shape, model_path)

sc(datasets, model_name, model_path, X_train, X_test, input_shape, samples_path)
