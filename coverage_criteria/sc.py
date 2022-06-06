# Guiding Deep Learning System Testing using Surprise Adequacy, https://arxiv.org/pdf/1808.08444.pdf#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import shutil
import sys
import gc

from tensorflow.python.platform import flags

sys.path.append("../")
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

from coverage_criteria.utils import at_training, model_load

from nmutant_util.utils_file import get_data_file
from nmutant_util.utils_tf import model_argmax, model_prediction
import os
import math
from load_model.tutorial_models import *

FLAGS = flags.FLAGS


def sc(datasets, model_name, model_path, X_train, X_test, input_shape, samples_path, layer=-3, num_section=1000):
    train_samples = X_train

    store_path = "../suprise/" + datasets + "/" + model_name + "/ori/"

    if not os.path.exists(store_path):
        os.makedirs(store_path)
        a_n_train = []
        train_labels = []
        n_batches = 1
        tf.reset_default_graph()
        for num in range(n_batches):
            print(num)
            start = 0
            end = len(train_samples)
            tf.reset_default_graph()
            sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name,
                                                             input_shape=input_shape, model_path=model_path)
            train_labels = train_labels + model_argmax(sess, x, preds, train_samples[start:end], feed_dict).tolist()
            a_n_train = a_n_train + at_training(sess, x, train_samples[start:end], model, feed_dict, layer).tolist()
            sess.close()
            del sess, preds, x, y, model, feed_dict
            gc.collect()

        a_n_train = np.asarray(a_n_train)
        train_labels = np.asarray(train_labels)
        np.save(store_path + "a_n_train.npy", a_n_train)
        np.save(store_path + "train_labels.npy", train_labels)
    else:
        a_n_train = np.load(store_path + "a_n_train.npy")
        train_labels = np.load(store_path + "train_labels.npy")

    class_inds = {}
    for i in range(2):  # 二分类,N分类的话range括号内填N
        class_inds[i] = np.where(train_labels == i)[0]

    kdes_store_path = store_path + 'kdes.npy'
    if os.path.exists(kdes_store_path):
        kdes = np.load(kdes_store_path, allow_pickle=True).item()
    else:
        kdes = {}
        for i in range(2):  # 二分类,N分类的话range括号内填N
            print(a_n_train)
            print(len(a_n_train[class_inds[i]]))
            print(-1.0/(len(a_n_train[0])+4))
            if len(a_n_train[class_inds[i]]) == 0:
                return 0, 0
            else:
                scott_bw = pow(len(a_n_train[class_inds[i]]), -1.0/(len(a_n_train[0])+4))

                kdes[i] = KernelDensity(kernel='gaussian',
                                    bandwidth=scott_bw).fit(a_n_train[class_inds[i]])
        np.save(kdes_store_path, kdes)

    lsa = []
    dsa = []
    c = set(range(len(train_labels)))

    lsa_test_store_path = store_path + "lsa_test.npy"
    dsa_test_store_path = store_path + "dsa_test.npy"

    if os.path.exists(lsa_test_store_path) and os.path.exists(dsa_test_store_path):
        lsa = np.load(lsa_test_store_path).tolist()
        dsa = np.load(dsa_test_store_path).tolist()
    else:
        X_test = X_test

        n_batches = 1
        for num in range(n_batches):
            print(num)
            start = 0
            end = len(X_test)
            batch_samples = X_test[start:end]
            tf.reset_default_graph()
            sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name,
                                                             input_shape=input_shape, model_path=model_path)
            batch_labels = model_argmax(sess, x, preds, batch_samples, feed=feed_dict)
            a_n_test = at_training(sess, x, batch_samples, model, feed_dict, layer)
            sess.close()
            del sess, preds, x, y, model, feed_dict
            gc.collect()

            for i in range(len(batch_samples)):
                kd_value = kdes[batch_labels[i]].score_samples(np.reshape(a_n_test[i], (1, -1)))[
                    0]  # / len(a_n_train[class_inds[batch_labels[i]]])
                lsa.append(-kd_value)
                data = np.asarray([a_n_test[i]], dtype=np.float32)
                batch = np.asarray(a_n_train[class_inds[batch_labels[i]]], dtype=np.float32)

                # dist = np.linalg.norm(data - batch, axis=age)
                dist = cdist(data, batch)[0]
                dist_a = min(dist)
                alpha_a = np.asarray([batch[np.argmin(dist)]])

                c_i = set(class_inds[batch_labels[i]])
                c_ni = list(c ^ c_i)
                batch = np.asarray(a_n_train[c_ni], dtype=np.float32)
                # dist_b = min(np.linalg.norm(alpha_a - batch, axis=age))
                dist_b = min(cdist(alpha_a, batch)[0])
                dsa.append(dist_a / dist_b)

        np.save(store_path + "lsa_test.npy", np.asarray(lsa))
        np.save(store_path + "dsa_test.npy", np.asarray(dsa))

    upper_lsa_test = max(lsa)
    lower_lsa_test = min(lsa)
    upper_dsa_test = max(dsa)
    lower_dsa_test = min(dsa)

    if samples_path not in ['test']:
        lsa = []
        dsa = []
        [image_list, _, _, _] = get_data_file(samples_path)  # image_files, real_labels, predicted_labels
        samples_adv = np.asarray(image_list)
        n_batches = int(np.ceil(1.0 * samples_adv.shape[0] / 512))
        for num in range(n_batches):
            print(num)
            start = 0
            end = len(samples_adv)
            batch_samples = samples_adv[start:end]

            tf.reset_compas_graph()
            sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name,
                                                             input_shape=input_shape, model_path=model_path)
            batch_labels = model_argmax(sess, x, preds, batch_samples, feed=feed_dict)
            a_n_adv = at_training(sess, x, batch_samples, model, feed_dict, layer)

            sess.close()
            del sess, preds, x, y, model, feed_dict
            gc.collect()

            for i in range(len(batch_samples)):
                kd_value = kdes[batch_labels[i]].score_samples(np.reshape(a_n_adv[i], (1, -1)))[
                    0]  # / len(a_n_train[class_inds[batch_labels[i]]])
                lsa.append(-kd_value)
                data = np.asarray([a_n_adv[i]], dtype=np.float32)
                batch = np.asarray(a_n_train[class_inds[batch_labels[i]]], dtype=np.float32)

                # dist = np.linalg.norm(data - batch, axis=age)
                dist = cdist(data, batch)[0]
                dist_a = min(dist)
                alpha_a = np.asarray([batch[np.argmin(dist)]])

                c_i = set(class_inds[batch_labels[i]])
                c_ni = list(c ^ c_i)
                batch = np.asarray(a_n_train[c_ni], dtype=np.float32)
                # dist_b = min(np.linalg.norm(alpha_a - batch, axis=age))
                dist_b = min(cdist(alpha_a, batch)[0])
                dsa.append(dist_a / dist_b)

    for i in range(len(lsa)):
        lsa[i] = (lsa[i] - lower_lsa_test) / (upper_lsa_test - lower_lsa_test)
        dsa[i] = (dsa[i] - lower_dsa_test) / (upper_dsa_test - lower_dsa_test)

    lsa_mean = np.mean(lsa)
    lsa_std = np.std(lsa)
    dsa_mean = np.mean(dsa)
    dsa_std = np.std(dsa)

    half_section = int(num_section / 2)
    n_section_lsa = np.zeros(num_section).astype('int64')
    n_section_dsa = np.zeros(num_section).astype('int64')
    for i in range(len(lsa)):
        l = lsa[i] * half_section
        d = dsa[i] * half_section
        if math.ceil(l) < num_section and math.floor(l) >= 0:
            if math.ceil(l) == math.floor(l):
                n_section_lsa[int(l) - 1] = 1
            else:
                n_section_lsa[int(l)] = 1
        if math.ceil(d) < num_section and math.floor(d) >= 0:
            if math.ceil(d) == math.floor(d):
                n_section_dsa[int(d) - 1] = 1
            else:
                n_section_dsa[int(d)] = 1

    cov_lsa_1 = 1.0 * sum(n_section_lsa[:half_section]) / (half_section)
    cov_dsa_1 = 1.0 * sum(n_section_dsa[:half_section]) / (half_section)

    cov_lsa_2 = 1.0 * sum(n_section_lsa[half_section:]) / (half_section)
    cov_dsa_2 = 1.0 * sum(n_section_dsa[half_section:]) / (half_section)

    print([lsa_mean, lsa_std, cov_lsa_1, cov_lsa_2, upper_lsa_test, lower_lsa_test, dsa_mean, dsa_std, cov_dsa_1,
           cov_dsa_2, upper_dsa_test, lower_dsa_test])


    return lsa_mean, dsa_mean


