import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import snownlp
from snownlp import sentiment

stop_words = r'words/stopwords/stopwords.txt'
mark_words = r'words/markwords/markwords.txt'


def read_csv():
    """读取商品评论数据文件"""
    comment_data = pd.read_table('scrapingfile/TmallContent.txt', encoding='utf-8-sig', sep='\n', index_col=None,
                                 header=None)
    return comment_data


def clean_data(data):
    """数据清洗"""
    # 消除缺失数据df
    df = data.dropna()
    # df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    # df.columns = df.columns.str.replace('&hellip;', '')
    df.iloc[:, 0] = df.iloc[:, 0].str.replace('&hellip;', '')
    # print(df.iloc[0].values)
    # df=data.drop_duplicates(inplace=True)

    # df = data.drop(df.iloc[:, 0].values == '这衣服的做工真的很精致，不多说了，上图。光线不好，拍的不理想')
    # print(str(df.iloc[0].values))

    # for i in range(len(df)):
    #     df = data.drop(df.iloc[i].values == ['这衣服的做工真的很精致，不多说了，上图。光线不好，拍的不理想'], inplace = True)
    # df = pd.DataFrame(df.iloc[:, 0].unique())
    return df


def remove_markwords(data):
    with open(mark_words, 'r', encoding='utf-8') as f:
        markwords = [line.strip() for line in f.readlines()]
    for i in range(len(markwords)):
        data.iloc[:, 0] = data.iloc[:, 0].str.replace(markwords[i], ' ')
        print(markwords[i])
    return data


def clean_stop_words(data):
    with open(stop_words, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    for i in range(len(stopwords)):
        # data.columns = data.columns.str.replace(stopwords[i], '')
        data.iloc[:, 0] = data.iloc[:, 0].str.replace(stopwords[i], ' ')
        print(stopwords[i])
    return data


def clean_repeat_word(raw_str, reverse=False):
    """去除评论中的重复使用的词汇"""
    if reverse:
        raw_str = raw_str[::-1]
    res_str = ''
    for i in raw_str:
        if i not in res_str:
            res_str += i
    if reverse:
        res_str = res_str[::-1]
    return res_str


def processed_data(filename):
    """清洗完毕的数据，并保存"""
    df = clean_data(read_csv())
    ds = clean_stop_words(df)
    rmmw = remove_markwords(ds)
    # ser1 = df.iloc[:, 0].apply(clean_repeat_word)
    # df2 = pd.DataFrame(ser1.apply(clean_repeat_word, reverse=True))
    # rmmw.to_csv(f'{filename}.csv', encoding='utf-8-sig', index_label=None, index=False, header=False)
    rmmw.to_csv(f'scrapingfile/good.txt', encoding='utf-8-sig', index_label=None, index=False, header=False)

    # print('数据清洗后：', len(df))


def train():
    """训练正向和负向情感数据集，并保存训练模型"""
    sentiment.train('neg.txt', 'pos.txt')
    sentiment.save('sentiment.marshal')


sentiment_list = []

res_list = []


def test(filename, to_filename):
    """商品评论-情感分析-测试"""
    with open(f'scrapingfile/{filename}.csv', 'r', encoding='utf-8-sig') as fr:
        for line in fr.readlines():
            s = snownlp.SnowNLP(line)

            if s.sentiments >= 0.8:
                res = '超赞'
                res_list.append(1)
            elif 0.6 <= s.sentiments < 0.8:
                res = '喜欢'
                res_list.append(0.5)
            elif 0.2 <= s.sentiments < 0.4:
                res = '还行'
                res_list.append(-0.5)
            elif s.sentiments < 0.2:
                res = '厌恶'
                res_list.append(-1)
            else:
                res = '一般'
                res_list.append(0)
            sent_dict = {
                '情感分析结果': s.sentiments,
                '评价倾向': res,
                '商品评论': line.replace('\n', '')
            }
            sentiment_list.append(sent_dict)
            print(sent_dict)
        df = pd.DataFrame(sentiment_list)
        df.to_csv(f'scrapingfile/{to_filename}.txt', index=None, encoding='utf-8-sig', index_label=None, mode='w')
    fr.close()


def data_virtualization():
    """分析结果可视化，以条形图为测试样例"""
    # font = FontProperties(fname=r"C:\\Windows\\Fonts\\simsunb.ttf", size=14)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['font.serif'] = ['KaiTi']
    superb = len([i for i in res_list if i == 1])
    likes = len([i for i in res_list if i == 0.5])
    common = len([i for i in res_list if i == 0])
    Okay = len([i for i in res_list if i == -0.5])
    unlikes = len([i for i in res_list if i == -0.5])

    plt.bar([1], [superb], label='超赞')
    plt.bar([2], [likes], label='喜欢')
    plt.bar([3], [common], label='一般')
    plt.bar([4], [Okay], label='还行')
    plt.bar([5], [unlikes], label='厌恶')

    plt.legend()
    plt.xlabel('结果')
    plt.ylabel('值')
    plt.title(u'商品评论情感分析结果-条形图')
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    pic_name = 'npl_analysis_' + time_name + '.png'
    plt.savefig(f'static/NlpAnalysis/nlp_img/' + pic_name, dpi=200, bbox_inches='tight')
    # plt.title(u'商品评论情感分析结果-条形图', FontProperties=font)
    return pic_name


def main():
    # 动态修改配置
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['font.serif'] = ['KaiTi']

    processed_data('TmallContent')
    # data_virtualization()    # train()  # 训练

    test('TmallContent', 'result')

    data_virtualization()  # 数据可视化
