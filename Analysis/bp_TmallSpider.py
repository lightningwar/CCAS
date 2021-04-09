import re
import time

from flask import render_template, request, jsonify, url_for

import tmall_spider
import sentiment_analysis
from Analysis import Analysis_blue

@Analysis_blue.route('/analysis/tmallscraping/', methods=['post'])
def tmallscraping():
    url_for('Analysis.tmallscraping')
    Page_Num = 99
    Num_PerPage = 20
    url = request.get_data().decode("utf-8")

    tmall_spider.Get_Url(Page_Num, url)
    time = tmall_spider.GetInfo(Num_PerPage)

    return time

@Analysis_blue.route('/analysis/sample/', methods=['post'])
def tmallsample():
    url_for('Analysis.tmallsample')
    sample_total = tmall_spider.valid_comment()

    return str(sample_total)

@Analysis_blue.route('/analysis/nlpsentiment/', methods=['post'])
def nlpsentiment():
    url_for('Analysis.nlpsentiment')
    sentiment_analysis.processed_data('TmallContent')
    sentiment_analysis.test('TmallContent', 'result')
    pic_name = sentiment_analysis.data_virtualization()
    return pic_name