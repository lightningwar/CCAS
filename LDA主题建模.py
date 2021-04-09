# 没有安装 gensim ,可以试用 !pip install gensim 进行安装
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora, models
import re
import pku2
import matplotlib.pylab as plt


def main():
    neg = pku2.main()
    # 负面主题分析
    neg_dict = corpora.Dictionary(neg[2])  # 建立词典
    neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]  # 建立语料库
    neg_lda = models.LdaModel(neg_corpus, num_topics=4, id2word=neg_dict)  # LDA模型训练

    pos_theme = neg_lda.show_topics()  # 展示主题

    # print(pos_theme)

    # 匹配中文字符
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    """
    [\u4e00-\u9fa5]  是匹配汉字的正则表达式
    [^\u4e00-\u9fa5] 是匹配非汉字的内容
    """

    # 主题一的特征词
    pattern.findall(pos_theme[0][1])

    # 取得每个主题的特征词
    pos_key_words = []
    for i in range(4):
        pos_key_words.append(pattern.findall(pos_theme[i][1]))

    # 变成 DataFrame 格式
    pos_key_words = pd.DataFrame(data=pos_key_words, index=['主题1', "主题2", "主题3", "主题4"])
    print(pos_key_words)

    data = pyLDAvis.gensim.prepare(neg_lda, neg_corpus, neg_dict)
    # pyLDAvis.save_html(data, 'web/lda.html')
    pyLDAvis.save_html(data, 'templates/web/lda.html')

    # pyLDAvis.enable_notebook()

    # plt.show()
