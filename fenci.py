import time

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def cut(filename):
    # 读取数据
    df = pd.read_csv('static/upload/' + filename, encoding='utf-8-sig')

    # 转换数据类型
    df.score = df.score.astype(int)

    df['emotion'] = df['score'].map(lambda x: 1 if x == 1 else 0)

    # 对评论进行分词, 并以空格隔开
    cut_time_begin = time.time()
    df['cut_jieba'] = df.content.apply(lambda x: ' '.join(jieba.cut(x)))
    cut_time_end = time.time()
    cut_time = cut_time_end - cut_time_begin
    return df, str(cut_time) + 's'


def get_stopwords(path):
    """读取停用词"""
    with open(path, encoding='utf-8-sig') as f:
        stopwords = [i.strip() for i in f.readlines()]
        return stopwords


def stopwords_cut(df):
    path = r'words/stopwords/哈工大停用词表.txt'

    stopwords_time_begin = time.time()
    stopwords = get_stopwords(path)
    # 去除停用词
    df['cut_jieba'] = df.cut_jieba.apply(
        lambda x: ' '.join([w for w in (x.split(' ')) if w not in stopwords]))
    stopwords_time_end = time.time()
    stopwords_time = stopwords_time_end - stopwords_time_begin

    df = df.sort_index(axis=0)
    return df, str(stopwords_time) + 's'


def divide(df):
    # 划分
    X, y = df[['cut_jieba']], df.emotion
    divide_time_begin = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.3)

    vect = TfidfVectorizer(max_df=0.8,
                           min_df=3,
                           token_pattern=r"(?u)\b[^\d\W]\w+\b")
    divide_time_end = time.time()
    divide_time = divide_time_end - divide_time_begin

    # 对cut_jieba进行向量化, 并转化为dataframe.
    vect_matrix_1 = pd.DataFrame(vect.fit_transform(X_train.cut_jieba).toarray(), columns=vect.get_feature_names())
    return vect, X, y, X_train, X_test, y_train, y_test, str(divide_time) + 's'


def sample(X, y, X_train, X_test):
    sample_size = str(X.shape[0])
    train_size = str(X_train.shape[0])
    test_size = str(X_test.shape[0])
    pos_sum = str(y.value_counts()[1])
    neg_sum = str(y.value_counts()[0])
    return sample_size, train_size, test_size, pos_sum, neg_sum
