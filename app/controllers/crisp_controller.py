from flask import Blueprint, render_template

crisp_bp = Blueprint('crisp', __name__)

@crisp_bp.route('/crisp-ml')
def crisp():
    return render_template('crisp.html')
