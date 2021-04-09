import os
import sys

import pandas as pd


def main():
    sys.getdefaultencoding()  # 查看当前编码格式
    import importlib
    importlib.reload(sys)
    stoplist = 'words/stopwords/中文停用词表.txt'
    outputfile1 = 'scrapingfile/jieba_cut.txt'
    neg = pd.read_table(outputfile1, encoding='utf-8', header=None)  # 读入数据
    # pos = pd.read_csv(outputfile2, encoding = 'utf-8', header = None)
    # stop = pd.read_csv(stoplist, encoding = 'utf-8', header = None, sep = 'tipdm')
    stop = pd.read_table(stoplist, encoding='utf-8', header=None)
    # sep设置分割词，由于csv默认以半角逗号为分割词，而该词恰好在停用词表中，因此会导致读取出错
    # 所以解决办法是手动设置一个不存在的分割词，如tipdm。
    stop = [' ', ''] + list(stop[0])  # Pandas自动过滤了空格符，这里手动添加

    neg[1] = neg[0].apply(lambda s: s.split(' '))  # 定义一个分割函数，然后用apply广播
    neg[2] = neg[1].apply(lambda x: [i for i in x if i not in stop])  # 逐词判断是否停用词
    # pos[1] = pos[0].apply(lambda s: s.split(' '))
    # pos[2] = pos[1].apply(lambda x: [i for i in x if i not in stop])
    neg.to_csv('scrapingfile/topic.txt', index=False, header=False, encoding='utf-8-sig')  # 保存结果
    print(neg.head())
    return neg
