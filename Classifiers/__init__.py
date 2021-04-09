from flask import Blueprint

Classifiers_blue = Blueprint('Classifiers', __name__)

from Classifiers import algorithm
from Classifiers import bp_prepeocess
from Classifiers import bp_knnClassifier
from Classifiers import bp_bayesClassifier
from Classifiers import bp_rforestClassifier
