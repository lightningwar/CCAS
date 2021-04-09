from flask import Blueprint

Analysis_blue = Blueprint('Analysis', __name__)

from Analysis import bp_TmallSpider
from Analysis import bp_LdaAnalysis
