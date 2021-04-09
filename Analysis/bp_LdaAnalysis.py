from flask import url_for

from Analysis import Analysis_blue
import pku
import pku2
import LDA主题建模


@Analysis_blue.route('/analysis/ldaprocessing/', methods=['post'])
def ldaprocessing():
    url_for('Analysis.ldaprocessing')
    pku.main()
    # pku2.main()
    LDA主题建模.main()
    return 'OK'
