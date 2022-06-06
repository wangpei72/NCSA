import csv
import os.path
import sys

sys.path.append("../")
import numpy as np
from aif360.algorithms.preprocessing.optim_preproc_helpers.structure_dataset_helper.metric_work_flow import *
from get_pre_dataset import *


data_set_list = ['adult',
                 'default',
                 'meps15', 'meps16'] # 4个 其他不参与metric计算
data_set_list_compat = ['adult', 'adult',
                        'default',
                        'meps15', 'meps16'] # 5 组
dataset_with_d_attr_list = ['adult_race', 'adult_sex',
                            'default_sex',
                            'meps15_race',
                            'meps16_race'] # 5组
model_type_list= ['dnn1', 'dnn2', 'dnn3', 'dnn4']  # dnn5已经做过了 ，去掉了
two_d_labels_test_rela_path_list, features_test_rela_path_list, labels_test_rela_path_list, pred_rela_path_list =\
gen_relative_sample_feed_path_list(dataset_name='adult')
# 注意，这里rq2，需要对所有数据集的，绝对大小子集进行测试，输入为： x y pred
# pre文件为了方便rw做ranker的学习，我们用的都是二维的，需要先调一次argmax
# 同时y的选用也有讲究，我们用2dlabels，所以，判断的时候也需要先argmax
# 因为我们的2dlabel已经是带了语义的，0就是0，1就是1，不存在根据fav是谁进行判断的问题，可以直接知晓一个案例是正还是负样本


def main():
    for j in range(len(model_type_list)):
        model_type = model_type_list[j]
        for i in range(len(data_set_list_compat)):
            out_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                        'fairness_result', 'rq2b_abso', dataset_with_d_attr_list[i], model_type)
            if not os.path.exists(out_csv_path):
                os.makedirs(out_csv_path)
            f = open(out_csv_path + '/absolute-fair-metric.csv', 'w+', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ["size", "SPD", "EOOD", "EOOP", "DI", "ACC", "THEIL", "SPD_mean",
                 "EOOD_mean", "EOOP_mean", "DI_mean", "ACC_mean", "THEIL_mean"])
            print('now dataset compat id %d' % (i + 1))
            for j in range(20):  # 5%为阶梯或者 100为阶梯
                n = j + 1
                m = str(n * 100)
                spd = 0
                di = 0
                eood = 0
                eoop = 0
                acc = 0
                theil = 0
                for k in range(5):
                    print('=============' + m + 'steps' + '-' + str(k + 1) + 'times=================')
                    X_test = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                  'data', data_set_list_compat[i] + '-aif360preproc-done', 'test',
                                                  str(k + 1), 'featrures-test-' + m + '.npy'))
                    Y_two_test = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                      'data', data_set_list_compat[i] + '-aif360preproc-done', 'test',
                                                      str(k + 1), '2d-labels-test-' + m + '.npy'))
                    pred_two_test = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                         'data', data_set_list_compat[i] + '-aif360preproc-done',
                                                         'test', str(k + 1), model_type, '2d-score-test-' + m + '.npy'))
                    Y_test = np.argmax(Y_two_test, axis=1)[:, np.newaxis]
                    pred_test = np.argmax(pred_two_test, axis=1)[:, np.newaxis]
                    dm = structring_classification_dataset_from_npy_array(X_test, Y_test, pred_test,
                                                                          dataset_name=data_set_list_compat[i],
                                                                          dataset_with_d_name=dataset_with_d_attr_list[i],
                                                                          print_bool=True)
                    SPD = dm.statistical_parity_difference()
                    DI = dm.disparate_impact()
                    EOOD = dm.average_abs_odds_difference()
                    EOOP = dm.equal_opportunity_difference()
                    ACC = dm.accuracy()
                    THEIL = dm.theil_index()
                    spd += SPD
                    di += DI
                    eood += EOOD
                    eoop += EOOP
                    acc += ACC
                    theil += THEIL
                    if k == 4:
                        spd /= 5.
                        di /= 5.
                        eood /= 5.
                        eoop /= 5.
                        acc /= 5.
                        theil /= 5.

                        csv_writer.writerow([m, SPD, EOOD, EOOP, DI, ACC, THEIL, spd, eood, eoop, di, acc, theil])
                    else:
                        csv_writer.writerow([m, SPD, EOOD, EOOP, DI, ACC, THEIL])
            f.close()


if __name__ == '__main__':
    main()
