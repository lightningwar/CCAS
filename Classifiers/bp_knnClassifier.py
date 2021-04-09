import pickle

from flask import url_for
from redis import Redis

import knnClassifier
from Classifiers import Classifiers_blue


@Classifiers_blue.route('/calbestK/', methods=['GET', 'POST'], strict_slashes=False)
def calbestK():
    global vect, best_K, X_train, y_train
    url_for('Classifiers.calbestK')
    cache = Redis(host='127.0.0.1', port=6379)
    vect_bytes = cache.get('vect')
    X_train_bytes = cache.get('X_train')
    y_train_bytes = cache.get('y_train')
    vect = pickle.loads(vect_bytes)
    X_train = pickle.loads(X_train_bytes)
    y_train = pickle.loads(y_train_bytes)
    best_K, best_cross_score, pic_name = knnClassifier.cal_best_K(vect, X_train, y_train)
    print('1')
    return {'best_K': best_K,
            'best_cross_score': best_cross_score,
            'pic_name': pic_name}


@Classifiers_blue.route('/calprescore/', methods=['GET', 'POST'], strict_slashes=False)
def calprescore():
    global vect, pipe, X_test, best_K, y_test, y_pre
    url_for('Classifiers.calprescore')
    pipe = knnClassifier.model_eva(vect, best_K)
    cache = Redis(host='127.0.0.1', port=6379)
    X_test_bytes = cache.get('X_test')
    y_test_bytes = cache.get('y_test')
    X_test = pickle.loads(X_test_bytes)
    y_test = pickle.loads(y_test_bytes)
    score, y_pre = knnClassifier.cal_pre_score(pipe, X_train, X_test, y_train, y_test)
    print('2')
    return score


@Classifiers_blue.route('/confusionmatrix/', methods=['GET', 'POST'], strict_slashes=False)
def confusionmatrix():
    global y_test, y_pre
    url_for('Classifiers.confusionmatrix')
    TP, FP, FN, TN, accurate_rate, recall_rate, pic_name = knnClassifier.confusion_matrix(y_test, y_pre)
    print('3')
    return {'TP': str(TP),
            'FP': str(FP),
            'FN': str(FN),
            'TN': str(TN),
            'accurate_rate': accurate_rate,
            'recall_rate': recall_rate,
            'pic_name': pic_name}


@Classifiers_blue.route('/roc/', methods=['GET', 'POST'], strict_slashes=False)
def roc():
    global pipe, X_test, y_test, y_score
    url_for('Classifiers.roc')
    y_score, roc_auc, pic_name = knnClassifier.ROC(pipe, X_test, y_test)
    print('4')
    return {'roc_auc': roc_auc, 'pic_name': pic_name}


@Classifiers_blue.route('/pr/', methods=['GET', 'POST'], strict_slashes=False)
def pr():
    global y_test, y_score
    url_for('Classifiers.pr')
    pic_name = knnClassifier.pr(y_test, y_score)
    print('5')
    return pic_name
