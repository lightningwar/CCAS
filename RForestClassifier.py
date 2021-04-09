import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# 解决 RuntimeError: main thread is not in main loop
plt.switch_backend('agg')


def cal_best_N(vect, X_train, y_train):
    k_range = range(1, 31)
    cv_scores = []
    for n in k_range:
        rf = RandomForestClassifier(n)
        # 利用pipeline管道进行特征向量化
        pipe = make_pipeline(vect, rf)
        score = cross_val_score(pipe, X_train.cut_jieba, y_train, cv=5, scoring='accuracy')
        cv_scores.append(score.mean())
    print(cv_scores.index(max(cv_scores)))
    best_n = cv_scores.index(max(cv_scores)) + 1
    best_cross_score = max(cv_scores)
    best_cross_score = "%.2f%%" % (best_cross_score * 100)
    plt.plot(k_range, cv_scores)
    plt.xlabel('N Value in RANDOM FOREST')
    plt.ylabel('Cross-Validation Mean Accuracy')
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    pic_name = 'rf_bestn_' + time_name + '.png'
    plt.savefig(f'static/Classifier/rf_img/' + pic_name, dpi=100, bbox_inches='tight')
    plt.close()
    return best_n, best_cross_score, pic_name


def model_eva(vect, best_n):
    rf = RandomForestClassifier(best_n)
    pipe = make_pipeline(vect, rf)
    return pipe


def cal_pre_score(pipe, X_train, X_test, y_train, y_test):
    # 训练出模型
    pipe1 = pipe.fit(X_train.cut_jieba, y_train)
    # 用训练的模型分别预测cut_jieba和cut_snownlp的测试集
    y_pre = pipe1.predict(X_test.cut_jieba)
    # 评价模型预测结果
    score = metrics.accuracy_score(y_test, y_pre)
    score = "%.2f%%" % (score * 100)
    # print("metrics.accuracy_score")
    print('c')
    return score, y_pre


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
    pic_name = 'rf_matrix_' + time_name + '.png'
    plt.savefig(f'static/Classifier/rf_img/' + pic_name, dpi=100, bbox_inches='tight')
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
    return TP, FP, FN, TN, accurate_rate, recall_rate, pic_name


def ROC(pipe, X_test, y_test):
    # 模型拟合
    pipe0 = pipe.fit(X_test.cut_jieba, y_test)
    # 模型在测试数据集上的预测
    # 通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    # y_score = rf.predict_proba(vect.fit_transform(X1_test.cut_jieba).toarray())[:, 1]
    y_score = pipe0.predict(X_test.cut_jieba)
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
    pic_name = 'rf_roc_' + time_name + '.png'
    plt.savefig(f'static/Classifier/rf_img/' + pic_name, dpi=100, bbox_inches='tight')
    plt.close()
    roc_auc = "%.2f%%" % (roc_auc * 100)
    print('f')
    return y_score, roc_auc, pic_name


# y_pred_snow = X_test.cut_jieba.apply(lambda x: SnowNLP(x).sentiments)
# y_pred_snow = np.where(y_pred_snow > 0.5, 1, 0)
# metrics.accuracy_score(y_test, y_pred_snow)
# metrics.confusion_matrix(y_test, y_pred_snow)


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
    pic_name = 'rf_pr_' + time_name + '.png'
    plt.savefig(f'static/Classifier/rf_img/' + pic_name, dpi=100, bbox_inches='tight')
    plt.close()
    print('g')
    return pic_name
