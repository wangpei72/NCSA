import os.path
import sys

sys.path.append("../")
import numpy as np
import tensorflow as tf

from tutorial_models import *

dataset_rela_list = ['adult', 'bank', 'compas', 'default',
                     'german', 'heart', 'meps15', 'meps16',
                     'student']
dataset_abso_list = ['adult',
                     'default', 'meps15', 'meps16']  # 变成4个 5组
dataset_shape_map = {
    'adult': (None, 14), 'bank': (None, 20),
    'compas': (None, 21), 'german': (None, 20),
    'default': (None, 23), 'heart': (None, 13),
    'student': (None, 32),
    'meps15': (None, 42), 'meps16': (None, 42)}

model_type_list= ['dnn1', 'dnn2', 'dnn3', 'dnn4', 'dnn5']

def predict_dnn5(sample_feed_path, input_shape=(None, 20), nb_classes=2,
                 model_path='./bank-additional/model/bank-additional/999/' + 'test.model',
                 model_type='dnn5'):
    """
    :param sample_feed_path: 输入的测试集npy文件所在路径
    :param input_shape: 由模型指定
    :param nb_classes: 2
    :param model_path: 模型的ckpt路径
    :return: 返回经过ckpt预测得到的npy文件
    """
    input_shape = input_shape
    nb_classes = nb_classes
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    if model_type == 'dnn1':
        model = dnn1(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn3':
        model = dnn3(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn7':
        model = dnn7(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn9':
        model = dnn9(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn2':
        model = dnn2(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn4':
        model = dnn4(input_shape, nb_classes)
        preds = model(x)
    else:
        assert model_type == 'dnn5'
        model = dnn5(input_shape, nb_classes)
        preds = model(x)
    grad_0 = gradient_graph(x, preds)
    # saver = tf.train.Saver()
    model_path = model_path

    saver = tf.train.import_meta_graph(model_path + '.meta')
    # tf.get_variable_scope().reuse_variables()
    saver.restore(sess, model_path)
    sess.run(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()]
    )
    # predict
    sample_tmp = np.load(sample_feed_path)
    label_tmp = model_argmax(sess, x, preds, sample_tmp)
    ranker_score = model_probab(sess, x, preds, sample_tmp)
    sess.close()
    tf.reset_default_graph()
    return label_tmp, ranker_score


def gen_relative_sample_feed_path_list(dataset_name='adult', test_id=1, model_type='dnn5'):
    two_d_labels_test_rela_path_list = []
    features_test_rela_path_list = []
    labels_test_rela_path_list = []
    pred_rela_path_list = []
    for i in range(20):
        two_d_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/2d-labels-test-' + str((i + 1) * 5) + '%.npy'
        label_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/labels-test-' + str((i + 1) * 5) + '%.npy'
        feat_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/featrures-test-' + str((i + 1) * 5) + '%.npy'
        pred_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/' + model_type + '/' + '2d-score-test-' + str((i + 1) * 5) + '%.npy'
        two_d_labels_test_rela_path_list.append(two_d_path)
        labels_test_rela_path_list.append(label_path)
        features_test_rela_path_list.append(feat_path)
        pred_rela_path_list.append(pred_path)
    if not os.path.exists('../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/' + model_type + '/'):
        os.makedirs('../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/' + model_type + '/')
    return two_d_labels_test_rela_path_list, features_test_rela_path_list, labels_test_rela_path_list, pred_rela_path_list


def gen_absolute_sample_feed_path_list(dataset_name='adult', test_id=1, model_type='dnn5'):
    two_d_labels_test_abso_path_list = []
    features_test_abso_path_list = []
    labels_test_abso_path_list = []
    pred_abso_path_list = []
    for i in range(20):
        two_d_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/2d-labels-test-' + str((i + 1) * 100) + '.npy'
        label_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/labels-test-' + str((i + 1) * 100) + '.npy'
        feat_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/featrures-test-' + str((i + 1) * 100) + '.npy'
        pred_path = '../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/' + model_type + '/' + '2d-score-test-' + str((i + 1) * 100) + '.npy'
        two_d_labels_test_abso_path_list.append(two_d_path)
        labels_test_abso_path_list.append(label_path)
        features_test_abso_path_list.append(feat_path)
        pred_abso_path_list.append(pred_path)
    if not os.path.exists('../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/' + model_type + '/'):
        os.makedirs('../data/' + dataset_name + '-aif360preproc-done/' + 'test/' + str(
            test_id) + '/' + model_type + '/')
    return two_d_labels_test_abso_path_list, features_test_abso_path_list, labels_test_abso_path_list, pred_abso_path_list


def gen_model_path(dataset_name='adult', model_type='dnn5'):
    model_path = '../original_models/' + dataset_name + '/' + model_type + '/999/test.model'
    return model_path


def all_model_path_list(model_type='dnn5'):
    path_list = []
    for i in dataset_rela_list:
        str_tmp = '../original_models/' + i + '/' + model_type + '/999/test.model'
        path_list.append(str_tmp)
    return path_list


def get_rela_pred_npy():
    for j in range(len(model_type_list)):
        model_type = model_type_list[j]
        for dataset_id in range(len(dataset_rela_list)):
            print('>>>>get rela<<<<')
            print('current dataset is %s' % dataset_rela_list[dataset_id])
            for test_id in range(5):
                print('current test id is %d' % (test_id + 1))
                two_d_labels_test_rela_path_list, features_test_rela_path_list, labels_test_rela_path_list, pred_rela_path_list = \
                    gen_relative_sample_feed_path_list(dataset_rela_list[dataset_id], test_id + 1, model_type=model_type)
                for test_path_i in range(len(features_test_rela_path_list)):
                    model_path = gen_model_path(dataset_name=dataset_rela_list[dataset_id], model_type=model_type)
                    if not os.path.exists(pred_rela_path_list[test_path_i]):
                        label_tmp, ranker_score = predict_dnn5(features_test_rela_path_list[test_path_i],
                                                               input_shape=dataset_shape_map[
                                                                   dataset_rela_list[dataset_id]],
                                                               nb_classes=2,
                                                               model_path=model_path,
                                                               model_type=model_type)
                        np.save(pred_rela_path_list[test_path_i], ranker_score)



def get_abso_pre_npy():
    for k in range(len(model_type_list)):
        model_type = model_type_list[k]
        for i in range(len(dataset_abso_list)):
            print('====get abso ===')
            print('current dataset is %s' % dataset_abso_list[i])
            for j in range(5):
                print('current test id is %d' % (j + 1))
                two_d_labels_test_abso_path_list, features_test_abso_path_list, labels_test_abso_path_list, pred_abso_path_list = \
                    gen_absolute_sample_feed_path_list(dataset_abso_list[i], j + 1, model_type=model_type)
                for path_i in range(len(features_test_abso_path_list)):
                    model_path = gen_model_path(dataset_name=dataset_abso_list[i], model_type=model_type)
                    if not os.path.exists(pred_abso_path_list[path_i]):
                        label_tmp, ranker_score = predict_dnn5(features_test_abso_path_list[path_i],
                                                           input_shape=dataset_shape_map[dataset_abso_list[i]],
                                                           nb_classes=2,
                                                           model_path=model_path,
                                                           model_type=model_type)
                        np.save(pred_abso_path_list[path_i], ranker_score)


if __name__ == '__main__':
    get_rela_pred_npy()
    get_abso_pre_npy()

    print('done')
