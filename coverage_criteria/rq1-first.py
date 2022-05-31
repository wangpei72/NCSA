# Pei, Kexin, et al. "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." (2017).#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import os
import shutil
import sys

from tensorflow.python.platform import flags

from coverage_criteria.multi_testing_criteria import multi_testing_criteria
from coverage_criteria.neuron_coverage import neuron_coverage
from coverage_criteria.sc import sc
from coverage_criteria.utils import datasize

sys.path.append("../")
import sys
from load_model.tutorial_models import *

sys.path.append("../")

FLAGS = flags.FLAGS


def main(argv=None):
    f = open('../coverage_result/rq1_first/' + FLAGS.datasets + '/' + FLAGS.model + '/original-relative.csv', 'w',
             encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["size", "nc", "KMN", "NB", "SNA", "TKNC", "TKNP", "lsa", "dsa", "nc_mean", "KMN_mean", "NB_mean", "SNA_mean",
         "TKNC_mean", "lsa_mean", "dsa_mean"])
    model_path = '../original_models/'
    X_train = np.load('../data/' + FLAGS.datasets + '-aif360preproc-done/features-train.npy')
    for i in range(20):  # Relative dnn5%+ or 100+
        n = i + 1
        m = str(n * 5)
        # a = str(n * 100)
        nc = 0
        kmn = 0
        nb = 0
        sna = 0
        tknc = 0
        lsa = 0
        dsa = 0
        for k in range(5):  # 每个阶梯大小的TestSet随机选取五遍
            print('-----------------------' + m + 'steps' + '-' + str(k + 1) + 'times' + '-----------------------')
            X_test = np.load('../data/' + FLAGS.datasets + '-aif360preproc-done/test/' + str(
                k + 1) + '/featrures-test-' + m + '%.npy')
            # X_test = np.load('../data/' + FLAGS.datasets + '-aif360preproc-done/test/' + str(
            #     k + 1) + '/featrures-test-' + a + '.npy')
            NC = neuron_coverage(datasets=FLAGS.datasets,
                                     model_name=FLAGS.model,
                                     model_path=model_path,
                                     X_test=X_test,
                                     input_shape=datasize(FLAGS.datasets)
                                     )
            if os.path.exists('../multi_testing_criteria/'):
                shutil.rmtree('../multi_testing_criteria/')
            KMN, NB, SNA, TKNC, TKNP = multi_testing_criteria(datasets=FLAGS.datasets,
                                                              model_name=FLAGS.model,
                                                              samples_path=FLAGS.samples,
                                                              std_range=FLAGS.std_range,
                                                              k_n=FLAGS.k_n,
                                                              k_l=FLAGS.k_l,
                                                              input_shape=datasize(FLAGS.datasets),
                                                              model_path=model_path,
                                                              X_test=X_test,
                                                              X_train=X_train,
                                                              )
            if os.path.exists('../suprise/'):
                shutil.rmtree('../suprise/')
            LSA, DSA = sc(datasets=FLAGS.datasets,
                          model_name=FLAGS.model,
                          samples_path=FLAGS.samples,
                          layer=FLAGS.layer,
                          num_section=FLAGS.sections,
                          input_shape=datasize(FLAGS.datasets),
                          X_train=X_train,
                          X_test=X_test,
                          model_path=model_path
                          )
            nc = nc +NC
            kmn = kmn + KMN
            nb = nb + NB
            sna = sna + SNA
            tknc = tknc + TKNC
            lsa = lsa + LSA
            dsa = dsa + DSA

            if k == 4:
                nc = nc / 5.0
                kmn = kmn / 5.0
                nb = nb / 5.0
                sna = sna / 5.0
                tknc = tknc / 5.0
                lsa = lsa / 5.0
                dsa = dsa /5.0

                csv_writer.writerow([m + '%', NC, KMN, NB, SNA, TKNC, TKNP, LSA, DSA,nc, kmn, nb, sna, tknc,lsa,dsa])
            else:
                csv_writer.writerow([m + '%', NC, KMN, NB, SNA, TKNC, TKNP, LSA, DSA])



    f.close()


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'meps15', 'The target datasets.')
    flags.DEFINE_string('model', 'dnn5', 'The name of model')
    flags.DEFINE_string('samples', 'test', 'The path to load samples.')
    flags.DEFINE_float('std_range', 0.0, 'The parameter to difine boundary with std')
    flags.DEFINE_integer('k_n', 1000, 'The number of sections for neuron output')
    flags.DEFINE_integer('k_l', 2, 'The number of top-k neurons in one layer')
    flags.DEFINE_integer('layer', -3, 'the layer for calculating activation trace')
    flags.DEFINE_integer('sections', 1000, 'the number of sections for calculating coverage')
    tf.app.run()
