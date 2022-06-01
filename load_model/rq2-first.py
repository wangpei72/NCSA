import csv
import sys

sys.path.append("../")
import numpy as np
from aif360.algorithms.preprocessing.optim_preproc_helpers.structure_dataset_helper.metric_work_flow import *
from get_pre_dataset import *


dataset_list = dataset_list()
dataset_list_compat = dataset_list_compat()
dataset_d_list = dataset_d_list()
two_d_labels_test_rela_path_list, features_test_rela_path_list, labels_test_rela_path_list, pred_rela_path_list =\
gen_relative_sample_feed_path_list(dataset_name='adult')
# 注意，这里rq2，需要对所有数据集的，相对大小子集进行测试，输入为： x y pred
# pre文件为了方便rw做ranker的学习，我们用的都是二维的，需要先调一次argmax
# 同时y的选用也有讲究，我们用2dlabels，所以，判断的时候也需要先argmax
# 因为我们的2dlabel已经是带了语义的，0就是0，1就是1，不存在根据fav是谁进行判断的问题，可以直接知晓一个案例是正还是负样本
def structure_metric_class_from_npy(x_path, y_path, pre_path,dataset_name='adult',
                                                        dataset_with_d_name='adult_race',
                                                        print_bool=True):
    features_x = np.load(x_path)
    two_d_labels_y = np.load(y_path)
    two_d_pred = np.load(pre_path)
    labels_y = np.argmax(two_d_labels_y, axis=1)
    pred = np.argmax(two_d_pred, axis=1)
    dm = structring_classification_dataset_from_npy_array(features_x, labels_y, pred,
                                                     dataset_name=dataset_name,
                                                     dataset_with_d_name=dataset_with_d_name,
                                                     print_bool=print_bool)
    return dm


def get_all_relative_dm_list():
    dm_overall_tests_list = []
    for dataset_id in range(len(dataset_list_compat)):
        print('calculating fairness metric,current dataset is %s' %
              dataset_list_compat[dataset_id])
        dm_5_test_list = []
        for test_id in range(5):
            print('now test id is %d' % (test_id + 1))
            print('dataset compat is %s ' % dataset_list_compat[dataset_id])
            print('dataset with d is %s' % dataset_with_d_attr_list[dataset_id])
            two_d_labels_test_rela_path_list, features_test_rela_path_list, labels_test_rela_path_list, pred_rela_path_list = \
                gen_relative_sample_feed_path_list(dataset_list_compat[dataset_id], test_id + 1)

            for test_path_i in range(len(features_test_rela_path_list)):
                dm_single_test = structure_metric_class_from_npy(features_test_rela_path_list[test_path_i],
                                                     two_d_labels_test_rela_path_list[test_path_i],
                                                     pred_rela_path_list[test_path_i],
                                                     dataset_name=dataset_list_compat[dataset_id],
                                                     dataset_with_d_name=dataset_with_d_attr_list[dataset_id],
                                                     print_bool=True
                                                     )
                dm_5_test_list.append(dm_single_test)
        assert len(dm_5_test_list) == 5
        dm_overall_tests_list.append(dm_5_test_list)
    assert len(dm_overall_tests_list) == 11
    print('done with get all dm')
    return dm_overall_tests_list

# 求5次实验下，得到的数据的mean值，需要对每一条size的row，根据test id的不同，求和，之除以5
def calc_mean(dm_5_test_list):
    SPD = 0
    DI = 0
    EOOD = 0
    EOOP = 0
    ACC = 0

    spd = 0
    di = 0
    eood = 0
    eoop = 0
    acc = 0
    spd_ls = []
    di_ls = []
    eood_ls = []
    eoop_ls = []
    acc_ls = []
    for i in range(20):
        for k in range(5):
            dm = dm_5_test_list[k, i]
            SPD += dm.statistical_parity_difference()
            DI += dm.disparate_impact()
            EOOD += dm.average_abs_odds_difference()
            EOOP += dm.equal_opportunity_difference()
            ACC += dm.accuracy()
            if k == 4:
                spd = SPD / 5.
                di = DI / 5.
                eood = EOOD / 5.
                eoop = EOOP / 5.
                acc = ACC / 5.
                spd_ls.append(spd)
                di_ls.append(di)
                eood_ls.append(eood)
                eoop_ls.append(eoop)
                acc_ls.append(acc)
    return spd_ls, eood_ls, eoop_ls, di_ls, acc_ls


def wrt_rela_dm_to_csv(dm_overall_list, dataset_name='adult', dataset_d_name='adult_race', model_type='dnn5'):
    idex = dataset_list_compat.index(dataset_name)
    assert dataset_with_d_attr_list.index(dataset_d_name) == idex
    output_dir = '../fairness_result/rq2a_rela/' + dataset_name + '/' + model_type + '/'
    with open(output_dir + 'relative.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["size", "SPD", "EOOD", "EOOP", "DI", "ACC", "SPD_mean", "EOOD_mean", "EOOP_mean", "DI_mean", "ACC_mean"])
        print('now dataset compat id %d' % (idex + 1))
        dm_5_test_list = dm_overall_list[idex]
        spd_ls, eood_ls, eoop_ls, di_ls, acc_ls = calc_mean(dm_5_test_list)
        for j in range(5):
            dm_single_test_list = dm_5_test_list[j]
            for dm_i in range(len(dm_single_test_list)):
                print('test id is %d, dm id is %d' % (j + 1, dm_i + 1))
                SPD = dm_single_test_list[dm_i].statistical_parity_difference()
                DI = dm_single_test_list[dm_i].disparate_impact()
                EOOD = dm_single_test_list[dm_i].average_abs_odds_difference()
                EOOP = dm_single_test_list[dm_i].equal_opportunity_difference()
                ACC = dm_single_test_list[dm_i].accuracy()
                if j == 4:
                    print('test id is %d, writing mean ...' % (j + 1))
                    csv_writer.writerow([str(5 * (dm_i + 1)) + '%', SPD, EOOD, EOOP, DI, spd_ls[dm_i], eood_ls[dm_i],
                                         eoop_ls[dm_i], di_ls[dm_i], acc_ls[dm_i]])
                else:
                    csv_writer.writerow([str(5 * (dm_i + 1)) + '%', SPD, EOOD, EOOP, DI, ACC])
        f.close()


def main():
    dm_overall_rela_list = get_all_relative_dm_list()
    for i in range(len(data_set_list_compat)):
        dataset_name = dataset_list_compat[i]
        dataset_name_d = dataset_with_d_attr_list[i]
        wrt_rela_dm_to_csv(dm_overall_rela_list, dataset_name=dataset_name,
                           dataset_d_name=dataset_name_d,
                           model_type='dnn5')


if __name__ == '__main__':
    main()
