import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, feature_extraction
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.pipeline import make_pipeline

# 解决 RuntimeError: main thread is not in main loop
plt.switch_backend('agg')

tfidf = feature_extraction.text.TfidfTransformer()


def conversion_score(score):
    score = "%.2f%%" % (score * 100)
    return score


def cal_cross_val_score(vect, X_train, y_train):
    mnb = MultinomialNB()
    # gnb = GaussianNB()
    bnb = BernoulliNB()

    # 利用pipeline管道进行特征向量化
    Multinomial_pipe = make_pipeline(vect, mnb)
    # Gaussian_pile = make_pipeline(vect, gnb)
    Bernoulli_pile = make_pipeline(vect, bnb)

    pipe = [Multinomial_pipe, Bernoulli_pile]

    Multinomial_score = cross_val_score(Multinomial_pipe, X_train.cut_jieba, y_train, cv=5, scoring='accuracy').mean()
    # Gaussian_score = cross_val_score(Gaussian_pile, X_train.cut_jieba, y_train, cv=5, scoring='accuracy').mean()
    Bernoulli_score = cross_val_score(Bernoulli_pile, X_train.cut_jieba, y_train, cv=5, scoring='accuracy').mean()

    Multinomial_score = conversion_score(Multinomial_score)
    # Gaussian_score = conversion_score(Gaussian_score)
    Bernoulli_score = conversion_score(Bernoulli_score)

    bayes_cross_val_score = [Multinomial_score, Bernoulli_score]
    return pipe, bayes_cross_val_score


def cal_pre_score(pipe, X_train, X_test, y_train, y_test):
    # 用训练的模型预测cut_jieba
    y_Mpre = pipe[0].fit(X_train.cut_jieba, y_train).predict(X_test.cut_jieba)
    # y_Gpre = pipe[1].predict(X_test.cut_jieba).predict(X_test.cut_jieba)
    y_Bpre = pipe[1].fit(X_train.cut_jieba, y_train).predict(X_test.cut_jieba)

    y_pre = [y_Mpre, y_Bpre]
    # 评价模型预测结果
    Multinomial_score = conversion_score(metrics.accuracy_score(y_test, y_Mpre))
    # Gaussian_score = conversion_score(metrics.accuracy_score(y_test, y_Gpre))
    Bernoulli_score = conversion_score(metrics.accuracy_score(y_test, y_Bpre))

    bayes_pre_score = [Multinomial_score, Bernoulli_score]
    # print("metrics.accuracy_score")
    print('c')
    return bayes_pre_score, y_pre


# 混淆矩阵图
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
　　 cm:混淆矩阵值
　　 classes:分类标签
　　 """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    pic_name = 'bayes_matrix_' + time_name + '.png'
    plt.savefig(f'static/Classifier/bayes_img/' + pic_name, dpi=100, bbox_inches='tight')
    plt.close()
    print('d')
    return pic_name


def confusion_matrix(y_test, y_pre):
    labels = [1, 0]
    # 分别求得cut_jieba和cut_snownlp的混淆矩阵
    con_matrix = metrics.confusion_matrix(y_test, y_pre, labels=labels)
    TP = con_matrix[0][0]
    FP = con_matrix[0][1]
    FN = con_matrix[1][0]
    TN = con_matrix[1][1]
    # 分别计算精准率和召回率
    accu_rate = TP / (TP + FP)
    recall_rate = TP / (FN + TP)

    accurate_rate = "%.2f%%" % (accu_rate * 100)
    recall_rate = "%.2f%%" % (recall_rate * 100)

    class_names = ['1', '0']
    pic_name = plot_confusion_matrix(cm=con_matrix, classes=class_names)
    print('e')
    return accurate_rate, recall_rate, pic_name


def ROC(pipe, X_test, y_test):
    # 模型在测试数据集上的预测
    # y_score = bayes.predict_proba(vect.fit_transform(X1_test.cut_jieba).toarray())[:, 1]
    y_score = pipe.fit(X_test.cut_jieba, y_test).predict(X_test.cut_jieba)
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    # ROC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    pic_name = 'bayes_roc_' + time_name + '.png'
    plt.savefig(f'static/Classifier/bayes_img/' + pic_name, dpi=100, bbox_inches='tight')
    plt.close()
    roc_auc = '%.2f' % roc_auc
    print('f')
    return y_score, roc_auc, pic_name


def pr(y_test, y_score):
    # PR图
    plt.figure(1)  # 创建图表1
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    plt.figure(1)  # 创建图表1
    plt.plot(precision, recall)
    plt.plot(thresholds, precision[:-1])
    plt.plot(thresholds, recall[:-1])
    # plt.plot(precision[:-1],recall[:-1])
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    pic_name = 'bayes_pr_' + time_name + '.png'
    plt.savefig(f'static/Classifier/bayes_img/' + pic_name, dpi=100, bbox_inches='tight')
    plt.close()
    print('g')
    return pic_name
