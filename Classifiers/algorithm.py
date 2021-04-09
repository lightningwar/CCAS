import re

from flask import render_template, request, jsonify, url_for

from Classifiers import Classifiers_blue


@Classifiers_blue.route('/upload_img', methods=['POST'])
def upload_img():
    test_path = '../static/img/knn.png'
    return jsonify({'signal': 1, 'img_path': test_path})


@Classifiers_blue.route('/algorithm/upload/', methods=['post'])
def upload():
    global filename
    try:
        file = request.files['file']
        r = re.search(r'(\.\S+)', file.filename)
        fn = ""
        if (r != None):
            fn = r.group()
        file.save('static/upload/' + file.filename)
        filename = file.filename
        return filename
    except:
        return 'fail'


@Classifiers_blue.route('/algorithm/knnClassifier/')
def knnClassifier_index():
    url_for('Classifiers.knnClassifier_index')
    return render_template('web/knnClassifier_index.html')


@Classifiers_blue.route('/algorithm/bayesClassifier/')
def bayesClassifier_index():
    url_for('Classifiers.bayesClassifier_index')
    return render_template('web/bayesClassifier_index.html')


@Classifiers_blue.route('/algorithm/rforestClassifier/')
def rforestClassifier_index():
    url_for('Classifiers.rforestClassifier_index')
    return render_template('web/rforestClassifier_index.html')
