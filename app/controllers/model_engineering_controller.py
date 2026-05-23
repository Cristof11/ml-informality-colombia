from flask import Blueprint, render_template
import json, os

model_engineering_bp = Blueprint('model_engineering', __name__)

METRICS_PATH = os.path.join(os.path.dirname(__file__),
                             '../../app/static/models/metrics.json')

@model_engineering_bp.route('/model-engineering')
def model_engineering():
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
    except Exception as e:
        data = {'error': str(e)}
    return render_template('model_engineering.html', data=data)
