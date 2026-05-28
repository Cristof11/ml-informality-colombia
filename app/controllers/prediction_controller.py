from flask import Blueprint, render_template, request
import joblib, numpy as np, os, json

prediction_bp = Blueprint('prediction', __name__)

MODELS_PATH = os.path.join(os.path.dirname(__file__), '../../app/static/models')

CIIU_LABELS = {
    1: 'Agriculture / Fishing',
    2: 'Manufacturing',
    3: 'Commerce / Retail',
    4: 'Construction',
    5: 'Transport',
    6: 'Accommodation / Food Services',
    7: 'Information / Communication',
    8: 'Real Estate / Business Services',
    9: 'Other Services',
}

DEPT_LABELS = {
    5:'Antioquia', 8:'Atlántico', 11:'Bogotá D.C.', 13:'Bolívar',
    15:'Boyacá', 17:'Caldas', 19:'Cauca', 20:'Cesar', 23:'Córdoba',
    25:'Cundinamarca', 27:'Chocó', 41:'Huila', 44:'La Guajira',
    47:'Magdalena', 50:'Meta', 52:'Nariño', 54:'Norte de Santander',
    63:'Quindío', 66:'Risaralda', 68:'Santander', 70:'Sucre',
    73:'Tolima', 76:'Valle del Cauca', 81:'Arauca',
}

def load_model():
    rf     = joblib.load(f'{MODELS_PATH}/random_forest.pkl')
    scaler = joblib.load(f'{MODELS_PATH}/scaler.pkl')
    return rf, scaler

def make_prediction(form_data):
    """Process form data and return prediction result dict."""
    try:
        rf, scaler = load_model()

        # ── Parse inputs ───────────────────────────────────────────
        rut_bin      = int(form_data.get('rut', 0))
        cc_bin       = int(form_data.get('camara', 0))
        acc_bin      = int(form_data.get('accounting', 0))
        arl_bin      = int(form_data.get('arl', 0))
        workers      = int(form_data.get('workers', 1))
        months_op    = int(form_data.get('months_operating', 12))
        monthly_sales= float(form_data.get('monthly_sales', 0))
        ciiu_sector  = int(form_data.get('ciiu_sector', 3))

        # ── Validate ranges ────────────────────────────────────────
        workers    = max(1, min(9, workers))
        months_op  = max(1, min(600, months_op))
        monthly_sales = max(0, monthly_sales)

        # ── Feature engineering (same as training) ─────────────────
        log_sales = np.log1p(monthly_sales)
        features  = np.array([[rut_bin, cc_bin, acc_bin, arl_bin,
                                workers, months_op, log_sales, ciiu_sector]])

        features_scaled = scaler.transform(features)

        # ── Predict ────────────────────────────────────────────────
        prediction = int(rf.predict(features_scaled)[0])
        probabilities = rf.predict_proba(features_scaled)[0]
        prob_formal   = round(float(probabilities[0]) * 100, 1)
        prob_informal = round(float(probabilities[1]) * 100, 1)

        # ── Informality Index (same formula as training) ───────────
        inf_index = round(1 - np.mean([rut_bin, cc_bin, acc_bin, arl_bin]), 3)

        # ── Risk level ─────────────────────────────────────────────
        if inf_index >= 0.75:
            risk_level = 'High'
            risk_color = 'danger'
        elif inf_index >= 0.5:
            risk_level = 'Medium'
            risk_color = 'warning'
        else:
            risk_level = 'Low'
            risk_color = 'success'

        # ── Missing formality flags ────────────────────────────────
        missing = []
        if rut_bin == 0:   missing.append('RUT (Tax Registration)')
        if cc_bin  == 0:   missing.append('Cámara de Comercio')
        if acc_bin == 0:   missing.append('Accounting Records')
        if arl_bin == 0:   missing.append('ARL (Occupational Risk Insurance)')

        # ── Formalization recommendations ─────────────────────────
        recommendations = []
        if rut_bin == 0:
            recommendations.append({
                'icon': '📋',
                'title': 'Register with DIAN (RUT)',
                'detail': 'Tax registration is the strongest formalization signal. Visit dian.gov.co or any DIAN office. Free process.'
            })
        if cc_bin == 0:
            recommendations.append({
                'icon': '🏛️',
                'title': 'Register with Cámara de Comercio',
                'detail': 'Chamber of Commerce registration legitimizes your business and enables access to credit and public procurement.'
            })
        if acc_bin == 0:
            recommendations.append({
                'icon': '📊',
                'title': 'Implement Accounting Records',
                'detail': 'Even a simple income/expense log significantly improves your formality profile and supports tax compliance.'
            })
        if arl_bin == 0:
            recommendations.append({
                'icon': '🛡️',
                'title': 'Affiliate to ARL (Occupational Risk)',
                'detail': 'Protects you and employees from work accidents. Required by law for all formal workers in Colombia.'
            })
        if not recommendations:
            recommendations.append({
                'icon': '✅',
                'title': 'Business is fully formalized',
                'detail': 'Your business complies with all four key formalization indicators. Maintain your registrations up to date.'
            })

        return {
            'success': True,
            'prediction': prediction,
            'label': 'INFORMAL' if prediction == 1 else 'FORMAL',
            'prob_formal': prob_formal,
            'prob_informal': prob_informal,
            'inf_index': inf_index,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'missing_flags': missing,
            'recommendations': recommendations,
            'formality_flags': {
                'rut': rut_bin, 'cc': cc_bin,
                'acc': acc_bin, 'arl': arl_bin,
            },
            'input_summary': {
                'rut': 'Yes' if rut_bin else 'No',
                'cc': 'Yes' if cc_bin else 'No',
                'accounting': 'Yes' if acc_bin else 'No',
                'arl': 'Yes' if arl_bin else 'No',
                'workers': workers,
                'months_operating': months_op,
                'monthly_sales': f'COP {monthly_sales:,.0f}',
                'sector': CIIU_LABELS.get(ciiu_sector, 'Other'),
            },
            'ciiu_sector': ciiu_sector,
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }


@prediction_bp.route('/prediction-system', methods=['GET', 'POST'])
def prediction_system():
    result   = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        result = make_prediction(form_data)

    return render_template(
        'prediction.html',
        result=result,
        form_data=form_data,
        ciiu_labels=CIIU_LABELS,
        dept_labels=DEPT_LABELS,
    )
