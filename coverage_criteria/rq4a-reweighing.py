import csv

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
dataset_list_compat = dataset_list_compat()
dataset_d_list = dataset_d_list()
data_with_d_shape_list = data_with_d_shape_list()
model_type_list = ['dnn1',
                   'dnn2',
                   'dnn3',
                   'dnn4',
                   'dnn5']


def main():
    for j in range(len(model_type_list)):
        model_type = model_type_list[j]
        for i in range(len(dataset_list_compat)):
            print('=====reweighing model: current dataset under nc sa calcu is %s ======' % dataset_d_list[i])
            print('=====model type is %s =====' % model_type)
            out_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                        'coverage_result', 'rq4a-reweighing-debias', dataset_with_d_attr_list[i],
                                        model_type)
            if not os.path.exists(out_csv_path):
                os.makedirs(out_csv_path)
            if os.path.exists(out_csv_path + '/reweighing-coverage-cri.csv'):
                continue
            f = open(out_csv_path + '/reweighing-coverage-cri.csv', 'w+', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["size", "nc", "KMN", "NB", "SNA", "TKNC", "TKNP", "lsa", "dsa"])
            print('now dataset compat id %d, %s, model type is %s' % (i + 1,dataset_d_list[i], model_type))
            model_path = '../model_reweighing/'
            X_train = np.load('../data/' + data_set_list_compat[i] + '-aif360preproc-done/features-train.npy')
            X_test = np.load('../data/' + data_set_list_compat[i] + '-aif360preproc-done/features-test.npy')
            NC = neuron_coverage(datasets=dataset_with_d_attr_list[i],
                                 model_name=model_type,
                                 model_path=model_path,
                                 X_test=X_test,
                                 input_shape=datasize(dataset_list_compat[i])
                                 )

            KMN, NB, SNA, TKNC, TKNP = multi_testing_criteria(datasets=dataset_with_d_attr_list[i],
                                                              model_name=model_type,
                                                              samples_path='test',
                                                              std_range=0.0,
                                                              k_n=1000,
                                                              k_l=2,
                                                              input_shape=datasize(dataset_list_compat[i]),
                                                              model_path=model_path,
                                                              X_test=X_test,
                                                              X_train=X_train,
                                                              )
            LSA = 0
            DSA = 0
            # sc还是有点问题，先不跑了

            # LSA, DSA = sc(datasets=dataset_with_d_attr_list[i],
            #               model_name=model_type,
            #               samples_path='test',
            #               layer=-3,
            #               num_section=1000,
            #               input_shape=datasize(dataset_list_compat[i]),
            #               X_train=X_train,
            #               X_test=X_test,
            #               model_path=model_path
            #               )
            # csv_writer.writerow(['100%', NC])
            csv_writer.writerow(['100%', NC, KMN, NB, SNA, TKNC, TKNP, LSA, DSA])
            f.close()


if __name__ == '__main__':
    # flags.DEFINE_float('std_range', 0.0, 'The parameter to difine boundary with std')
    # flags.DEFINE_integer('k_n', 1000, 'The number of sections for neuron output')
    # flags.DEFINE_integer('k_l', 2, 'The number of top-k neurons in one layer')
    # flags.DEFINE_integer('layer', -3, 'the layer for calculating activation trace')
    # flags.DEFINE_integer('sections', 1000, 'the number of sections for calculating coverage')
    # tf.app.run()
    main()