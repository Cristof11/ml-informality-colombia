from flask import Blueprint, render_template
import pandas as pd
import os
import json

data_understanding_bp = Blueprint('data_understanding', __name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/emicron_sample.csv')

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception:
        return None

@data_understanding_bp.route('/data-understanding')
def data_understanding():
    df = load_data()
    stats = {}

    if df is not None:
        # Basic shape
        stats['n_records'] = df.shape[0]
        stats['n_variables'] = df.shape[1]
        stats['columns'] = list(df.columns)

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        stats['missing'] = missing_pct[missing_pct > 0].to_dict()

        # Numeric summary
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            desc = df[num_cols].describe().round(2)
            stats['numeric_summary'] = desc.to_html(
                classes='table table-sm', border=0)

        # Categorical distributions - key variables
        cat_stats = {}
        key_cats = ['P1633', 'P1055', 'P3091', 'P640', 'DPTO']
        for col in key_cats:
            if col in df.columns:
                vc = df[col].value_counts(dropna=False).head(6)
                cat_stats[col] = vc.to_dict()
        stats['cat_stats'] = cat_stats

        # Correlation matrix (numeric only, top 8 cols)
        if len(num_cols) >= 2:
            corr = df[num_cols[:8]].corr().round(2)
            stats['correlation'] = corr.to_html(
                classes='table table-sm corr-table', border=0)
    else:
        stats['error'] = (
            'Dataset not found. Please place emicron_sample.csv '
            'in the /data folder. Download from: '
            'https://microdatos.dane.gov.co/index.php/catalog/832'
        )

    return render_template('data_understanding.html', stats=stats)
