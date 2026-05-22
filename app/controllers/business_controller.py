from flask import Blueprint, render_template

business_bp = Blueprint('business', __name__)

@business_bp.route('/business-understanding')
def business():
    return render_template('business.html')
