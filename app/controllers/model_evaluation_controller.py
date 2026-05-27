from flask import Blueprint, render_template
import json, os

model_evaluation_bp = Blueprint('model_evaluation', __name__)

EVAL_PATH = os.path.join(os.path.dirname(__file__),
                          '../../app/static/models/eval_metrics.json')

@model_evaluation_bp.route('/model-evaluation')
def model_evaluation():
    try:
        with open(EVAL_PATH) as f:
            data = json.load(f)
    except Exception as e:
        data = {'error': str(e)}
    return render_template('model_evaluation.html', data=data)
