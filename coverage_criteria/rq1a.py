import csv
import shutil

from tensorflow.python.platform import flags
from coverage_criteria.multi_testing_criteria import multi_testing_criteria
from coverage_criteria.neuron_coverage import neuron_coverage
from coverage_criteria.sc import sc
from coverage_criteria.utils import datasize
import sys, os
sys.path.append("../")
from aif360.algorithms.preprocessing.optim_preproc_helpers.structure_dataset_helper.metric_work_flow import *
from load_model.tutorial_models import *

FLAGS = flags.FLAGS

dataset_list = dataset_list()

data_shape_list = data_shape_list()
model_type_list = ['dnn1',
                   'dnn2',
                   'dnn3',
                   'dnn4',
                   'dnn5']


def main():
    for j in range(len(model_type_list)):
        model_type = model_type_list[j]
        for dataset_i in range(len(dataset_list)):
            dataset = dataset_list[dataset_i]
            print('=====original model: current dataset under nc sa calcu is %s ======' % dataset)
            print('=====model type is %s =====' % model_type)

            out_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                        'coverage_result', 'rq1_a', dataset, model_type)
            if not os.path.exists(out_csv_path):
                os.makedirs(out_csv_path)
            if os.path.exists(out_csv_path + '/rq1a-origin-coverage.csv'):
                continue
            f = open(out_csv_path + '/rq1a-origin-coverage.csv', 'w+', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["size", "nc", "KMN", "NB", "SNA", "TKNC", "TKNP", "lsa", "dsa", "nc_mean", "KMN_mean", "NB_mean",
                 "SNA_mean", "TKNC_mean", "lsa_mean", "dsa_mean"])
            model_path = '../original_models/'
            X_train = np.load('../data/' + dataset + '-aif360preproc-done/features-train.npy')

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
                    print('-----------------------' + m + 'steps' + '-' + str(
                        k + 1) + 'times' + '-----------------------')
                    X_test = np.load('../data/' + dataset + '-aif360preproc-done/test/' + str(
                        k + 1) + '/featrures-test-' + m + '%.npy')
                    # X_test = np.load('../data/' + dataset + '-aif360preproc-done/test/' + str(
                    #     k + 1) + '/featrures-test-' + a + '.npy')
                    NC = neuron_coverage(datasets=dataset,
                                         model_name=model_type,
                                         model_path=model_path,
                                         X_test=X_test,
                                         input_shape=datasize(dataset)
                                         )
                    if os.path.exists('../multi_testing_criteria/'):
                        shutil.rmtree('../multi_testing_criteria/')
                    KMN, NB, SNA, TKNC, TKNP = multi_testing_criteria(datasets=dataset,
                                                                      model_name=model_type,
                                                                      samples_path='test',
                                                                      std_range=0.0,
                                                                      k_n=1000,
                                                                      k_l=2,
                                                                      input_shape=datasize(dataset),
                                                                      model_path=model_path,
                                                                      X_test=X_test,
                                                                      X_train=X_train,
                                                                      )
                    if os.path.exists('../suprise/'):
                        shutil.rmtree('../suprise/')
                    LSA, DSA = sc(datasets=dataset,
                                  model_name=model_type,
                                  samples_path='test',
                                  layer=-3,
                                  num_section=1000,
                                  input_shape=datasize(dataset),
                                  X_train=X_train,
                                  X_test=X_test,
                                  model_path=model_path
                                  )
                    nc = nc + NC
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
                        dsa = dsa / 5.0

                        csv_writer.writerow(
                            [m + '%', NC, KMN, NB, SNA, TKNC, TKNP, LSA, DSA, nc, kmn, nb, sna, tknc, lsa, dsa])
                    else:
                        csv_writer.writerow([m + '%', NC, KMN, NB, SNA, TKNC, TKNP, LSA, DSA])


if __name__ == '__main__':
    # flags.DEFINE_float('std_range', 0.0, 'The parameter to difine boundary with std')
    # flags.DEFINE_integer('k_n', 1000, 'The number of sections for neuron output')
    # flags.DEFINE_integer('k_l', 2, 'The number of top-k neurons in one layer')
    # flags.DEFINE_integer('layer', -3, 'the layer for calculating activation trace')
    # flags.DEFINE_integer('sections', 1000, 'the number of sections for calculating coverage')
    # tf.app.run()
    main()