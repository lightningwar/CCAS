import pickle

from flask import url_for
from redis import Redis

import bayesClassifier
from Classifiers import Classifiers_blue


@Classifiers_blue.route('/bayes/calbayescrossvalscore/', methods=['GET', 'POST'], strict_slashes=False)
def calbayescrossvalscore():
    global vect, pipe, X_train, y_train
    url_for('Classifiers.calbayescrossvalscore')
    cache = Redis(host='127.0.0.1', port=6379)
    vect_bytes = cache.get('vect')
    X_train_bytes = cache.get('X_train')
    y_train_bytes = cache.get('y_train')
    vect = pickle.loads(vect_bytes)
    X_train = pickle.loads(X_train_bytes)
    y_train = pickle.loads(y_train_bytes)
    pipe, bayes_cross_val_score = bayesClassifier.cal_cross_val_score(vect, X_train, y_train)
    print('1')
    return {'mnb_cross_score': bayes_cross_val_score[0],
            'bnb_cross_score': bayes_cross_val_score[1]}


@Classifiers_blue.route('/bayes/calbayesprescore/', methods=['GET', 'POST'], strict_slashes=False)
def calbayesprescore():
    global vect, pipe, X_test, best_K, y_test, y_pre
    url_for('Classifiers.calbayesprescore')
    cache = Redis(host='127.0.0.1', port=6379)
    X_test_bytes = cache.get('X_test')
    y_test_bytes = cache.get('y_test')
    X_test = pickle.loads(X_test_bytes)
    y_test = pickle.loads(y_test_bytes)
    bayes_pre_score, y_pre = bayesClassifier.cal_pre_score(pipe, X_train, X_test, y_train, y_test)
    print('2')
    return {'mnb_pre_score': bayes_pre_score[0],
            'bnb_pre_score': bayes_pre_score[1]}


@Classifiers_blue.route('/bayes/bayesconfusionmatrix/', methods=['GET', 'POST'], strict_slashes=False)
def bayesconfusionmatrix():
    global y_test, y_pre
    url_for('Classifiers.bayesconfusionmatrix')
    mnb_accurate_rate, mnb_recall_rate, mnb_pic_name = bayesClassifier.confusion_matrix(y_test, y_pre[0])
    # gnb_accurate_rate, gnb_recall_rate, gnb_pic_name = bayesClassifier.confusion_matrix(y_test, y_pre[1])
    bnb_accurate_rate, bnb_recall_rate, bnb_pic_name = bayesClassifier.confusion_matrix(y_test, y_pre[1])
    print('3')
    return {'mnb_accurate_rate': mnb_accurate_rate, 'mnb_recall_rate': mnb_recall_rate, 'mnb_pic_name': mnb_pic_name,
            # 'gnb_accurate_rate':gnb_accurate_rate, 'gnb_recall_rate':gnb_recall_rate, 'gnb_pic_name':gnb_pic_name,
            'bnb_accurate_rate': bnb_accurate_rate, 'bnb_recall_rate': bnb_recall_rate, 'bnb_pic_name': bnb_pic_name}


@Classifiers_blue.route('/bayes/bayesroc/', methods=['GET', 'POST'], strict_slashes=False)
def bayesroc():
    global pipe, X_test, y_test, y_score
    url_for('Classifiers.bayesroc')
    mnb_y_score, mnb_roc_auc, mnb_pic_name = bayesClassifier.ROC(pipe[0], X_test, y_test)
    # gnb_y_score, gnb_roc_auc, gnb_pic_name = bayesClassifier.ROC(pipe[1], X_test, y_test)
    bnb_y_score, bnb_roc_auc, bnb_pic_name = bayesClassifier.ROC(pipe[1], X_test, y_test)
    y_score = [mnb_y_score, bnb_y_score]
    print('4')
    return {'mnb_roc_auc': mnb_roc_auc,
            # 'gnb_roc_auc': gnb_roc_auc,
            'bnb_roc_auc': bnb_roc_auc,
            'mnb_pic_name': mnb_pic_name,
            # 'gnb_pic_name':gnb_pic_name,
            'bnb_pic_name': bnb_pic_name}


@Classifiers_blue.route('/bayes/bayespr/', methods=['GET', 'POST'], strict_slashes=False)
def bayespr():
    global y_test, y_score
    url_for('Classifiers.bayespr')
    pic_name = []
    for i in range(0, 2):
        pic_name.append(bayesClassifier.pr(y_test, y_score[i]))
    print('5')
    return {'mnb_pic_name': pic_name[0],
            # 'gnb_pic_name':pic_name[1],
            'bnb_pic_name': pic_name[1]}
