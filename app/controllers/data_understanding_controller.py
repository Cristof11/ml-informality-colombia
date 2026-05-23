from flask import Blueprint, render_template
import pandas as pd
import os

data_understanding_bp = Blueprint('data_understanding', __name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/emicron_sample.csv')

def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None

@data_understanding_bp.route('/data-understanding')
def data_understanding():
    df = load_data()
    stats = {}
    if df is not None:
        stats['n_records']  = f"{df.shape[0]:,}"
        stats['n_variables'] = df.shape[1]
        stats['columns']    = list(df.columns)

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        stats['missing'] = missing_pct[missing_pct > 0].to_dict()

        # Numeric summary
        num_cols = df.select_dtypes(include='number').columns.tolist()
        display_num = ['WORKERS','MONTHS_OPERATING','MONTHLY_SALES','INFORMALITY_INDEX']
        display_num = [c for c in display_num if c in df.columns]
        if display_num:
            stats['numeric_summary'] = df[display_num].describe().round(2).to_html(
                classes='table table-sm', border=0)

        # Categorical distributions
        cat_stats = {}
        for col in ['RUT','CAMARA_COMERCIO','ACCOUNTING','CIIU_LABEL','DEPT_NAME']:
            if col in df.columns:
                vc = df[col].value_counts(dropna=False).head(8)
                cat_stats[col] = vc.to_dict()
        stats['cat_stats'] = cat_stats

        # Correlation
        corr_cols = [c for c in display_num if c in df.columns]
        if len(corr_cols) >= 2:
            stats['correlation'] = df[corr_cols].corr().round(3).to_html(
                classes='table table-sm corr-table', border=0)

        # Informality summary
        stats['inf_mean']      = round(df['INFORMALITY_INDEX'].mean(), 3) if 'INFORMALITY_INDEX' in df.columns else 'N/A'
        stats['no_rut_pct']    = round((df['RUT']==2).mean()*100, 1) if 'RUT' in df.columns else 'N/A'
        stats['no_cc_pct']     = round((df['CAMARA_COMERCIO']==2).mean()*100, 1) if 'CAMARA_COMERCIO' in df.columns else 'N/A'
        stats['no_acc_pct']    = round((df['ACCOUNTING']==5).mean()*100, 1) if 'ACCOUNTING' in df.columns else 'N/A'
        stats['median_sales']  = f"COP {int(df['MONTHLY_SALES'].median()):,}" if 'MONTHLY_SALES' in df.columns else 'N/A'
    else:
        stats['error'] = 'Dataset not found. Place emicron_sample.csv in /data folder.'

    return render_template('data_understanding.html', stats=stats)
