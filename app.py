import jieba

jieba.initialize()  # 手动初始化

from flask import Flask, render_template, url_for

filename = None
app = Flask(__name__)
from Classifiers import Classifiers_blue
from Analysis import Analysis_blue

app = Flask(__name__)
app.register_blueprint(Classifiers_blue)
app.register_blueprint(Analysis_blue)


@app.route('/')
def index():
    return render_template('web/index.html')


@app.route('/algorithm/')
def algorithm_index():
    url_for('algorithm_index')
    return render_template('web/algorithm_index.html')


@app.route('/nlp/')
def nlp_index():
    url_for('nlp_index')
    return render_template('web/nlp_index.html')


@app.route('/lda/')
def lda_index():
    url_for('lda_index')
    return render_template('web/lda.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
