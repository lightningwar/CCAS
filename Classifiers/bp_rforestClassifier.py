import pickle

from flask import url_for
from redis import Redis

import RForestClassifier
from Classifiers import Classifiers_blue


@Classifiers_blue.route('/RandomForest/calbestN/', methods=['GET', 'POST'], strict_slashes=False)
def calbestN():
    global vect, best_N, X_train, y_train
    url_for('Classifiers.calbestN')
    cache = Redis(host='127.0.0.1', port=6379)
    vect_bytes = cache.get('vect')
    X_train_bytes = cache.get('X_train')
    y_train_bytes = cache.get('y_train')
    vect = pickle.loads(vect_bytes)
    X_train = pickle.loads(X_train_bytes)
    y_train = pickle.loads(y_train_bytes)
    best_N, best_cross_score, pic_name = RForestClassifier.cal_best_N(vect, X_train, y_train)
    print('1')
    return {'best_N': best_N,
            'best_cross_score': best_cross_score,
            'pic_name': pic_name}


@Classifiers_blue.route('/RandomForest/calrfprescore/', methods=['GET', 'POST'], strict_slashes=False)
def calrfprescore():
    global vect, pipe, X_test, best_N, y_test, y_pre
    url_for('Classifiers.calrfprescore')
    pipe = RForestClassifier.model_eva(vect, best_N)
    cache = Redis(host='127.0.0.1', port=6379)
    X_test_bytes = cache.get('X_test')
    y_test_bytes = cache.get('y_test')
    X_test = pickle.loads(X_test_bytes)
    y_test = pickle.loads(y_test_bytes)
    score, y_pre = RForestClassifier.cal_pre_score(pipe, X_train, X_test, y_train, y_test)
    print('2')
    return score


@Classifiers_blue.route('/RandomForest/confusionmatrix/', methods=['GET', 'POST'], strict_slashes=False)
def rfconfusionmatrix():
    global y_test, y_pre
    url_for('Classifiers.rfconfusionmatrix')
    TP, FP, FN, TN, accurate_rate, recall_rate, pic_name = RForestClassifier.confusion_matrix(y_test, y_pre)
    print('3')
    return {'TP': str(TP),
            'FP': str(FP),
            'FN': str(FN),
            'TN': str(TN),
            'accurate_rate': accurate_rate,
            'recall_rate': recall_rate,
            'pic_name': pic_name}


@Classifiers_blue.route('/RandomForest/roc/', methods=['GET', 'POST'], strict_slashes=False)
def rfroc():
    global pipe, X_test, y_test, y_score
    url_for('Classifiers.rfroc')
    y_score, roc_auc, pic_name = RForestClassifier.ROC(pipe, X_test, y_test)
    print('4')
    return {'roc_auc': roc_auc, 'pic_name': pic_name}


@Classifiers_blue.route('/RandomForest/pr/', methods=['GET', 'POST'], strict_slashes=False)
def rfpr():
    global y_test, y_score
    url_for('Classifiers.rfpr')
    pic_name = RForestClassifier.pr(y_test, y_score)
    print('5')
    return pic_name
