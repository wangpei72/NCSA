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


def main_get_all_relative_metric():
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


def wrt_dm_to_csv(dm, dataset_name='adult', dataset_d_name='adult_race'):
    statistical_parity_diff = dm.statistical_parity_difference()
    disparate_impact = dm.disparate_impact()
    average_abs_odds_difference = dm.average_abs_odds_difference()
    equal_opportunity_difference = dm.equal_opportunity_difference()
    accuracy = dm.accuracy()




if __name__ == '__main__':
    main_get_all_relative_metric()
