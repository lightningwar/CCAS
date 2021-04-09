from flask import request, url_for
from redis import Redis
import fenci
from Classifiers import Classifiers_blue
import pickle


@Classifiers_blue.route('/wordcut/', methods=['GET', 'POST'], strict_slashes=False)
def wordcut():
    url_for('Classifiers.wordcut')
    global df
    filename = request.get_data().decode("utf-8")
    df, cut_time = fenci.cut(filename)
    return cut_time


@Classifiers_blue.route('/stopwordscut/', methods=['GET', 'POST'], strict_slashes=False)
def stopwordscut():
    url_for('Classifiers.stopwordscut')
    global df
    df, stopwords_time = fenci.stopwords_cut(df)
    return stopwords_time


@Classifiers_blue.route('/divide/', methods=['GET', 'POST'], strict_slashes=False)
def divide():
    url_for('Classifiers.divide')
    global df, vect, X, y, X_train, X_test, y_train, y_test, divide_time
    vect, X, y, X_train, X_test, y_train, y_test, divide_time = fenci.divide(df)
    vect_bytes = pickle.dumps(vect)
    X_train_bytes = pickle.dumps(X_train)
    y_train_bytes = pickle.dumps(y_train)
    X_test_bytes = pickle.dumps(X_test)
    y_test_bytes = pickle.dumps(y_test)
    cache = Redis(host='127.0.0.1', port=6379)
    vect = cache.set('vect',vect_bytes)
    cache.set('X_train', X_train_bytes)
    cache.set('y_train', y_train_bytes)
    cache.set('X_test', X_train_bytes)
    cache.set('y_test', y_train_bytes)
    return divide_time


@Classifiers_blue.route('/sample/', methods=['GET', 'POST'], strict_slashes=False)
def sample():
    url_for('Classifiers.sample')
    global X, y, X_train, X_test
    sample_size, train_size, test_size, pos_sum, neg_sum = fenci.sample(X, y, X_train, X_test)
    return {'sample_size': sample_size,
            'train_size': train_size,
            'test_size': test_size,
            'pos_sum': pos_sum,
            'neg_sum': neg_sum}


