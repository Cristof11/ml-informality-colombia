from flask import Blueprint, render_template
import pandas as pd
import numpy as np
import os

data_engineering_bp = Blueprint('data_engineering', __name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/emicron_sample.csv')

# ── Variable label map (from EMICRON DDI/XML) ──────────────────────────────
LABEL_MAP = {
    'P1633': 'Has RUT (tax ID)',
    'P1055': 'Chamber of Commerce registration',
    'P3091': 'ARL contribution (occupational risk)',
    'P640':  'Accounting records type',
    'P3034': 'Number of workers',
    'P3035': 'Months in operation',
    'P2991': 'Filed income tax return',
    'DPTO':  'Department (location)',
    'P3053': 'CIIU economic sector',
    'P3088': 'Monthly sales (previous month)',
}

RUT_COL     = 'P1633'   # 1=Yes, 2=No
CC_COL      = 'P1055'   # 1=Yes, 2=No
ARL_COL     = 'P3091'   # 1=Yes, 2=No
ACC_COL     = 'P640'    # 1-4=has records, 5=no records
WORKERS_COL = 'P3034'
MONTHS_COL  = 'P3035'
SALES_COL   = 'P3088'
DEPT_COL    = 'DPTO'
SECTOR_COL  = 'P3053'


def run_pipeline(df: pd.DataFrame):
    log = []
    original_shape = df.shape

    # ── STEP 1: Drop irrelevant / identification columns ───────────────────
    id_cols = [c for c in df.columns if c.startswith('DIRECTORIO') or
               c in ('SECUENCIA_P', 'SECUENCIA_ENCUESTA', 'ORDEN')]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])
    log.append({
        'step': '1 · Drop ID columns',
        'detail': f'Removed {len(id_cols)} identifier columns not needed for modelling.',
        'shape': df.shape
    })

    # ── STEP 2: Keep only relevant feature columns ─────────────────────────
    keep = [c for c in LABEL_MAP.keys() if c in df.columns]
    df = df[keep].copy()
    log.append({
        'step': '2 · Feature selection',
        'detail': f'Kept {len(keep)} key variables: {", ".join(keep)}.',
        'shape': df.shape
    })

    # ── STEP 3: Handle missing values ──────────────────────────────────────
    before_na = df.isnull().sum().sum()
    # Categorical: fill with mode per column
    cat_cols = [RUT_COL, CC_COL, ARL_COL, ACC_COL, DEPT_COL, SECTOR_COL]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    # Numeric: fill with median
    num_cols_present = [c for c in [WORKERS_COL, MONTHS_COL, SALES_COL] if c in df.columns]
    for col in num_cols_present:
        df[col] = df[col].fillna(df[col].median())
    after_na = df.isnull().sum().sum()
    log.append({
        'step': '3 · Null value handling',
        'detail': (f'Filled {before_na} missing values → {after_na} remaining. '
                   'Categorical: mode imputation. Numerical: median imputation.'),
        'shape': df.shape
    })

    # ── STEP 4: Encode binary formality variables → 1=formal, 0=informal ──
    binary_map = {RUT_COL: {1: 1, 2: 0},
                  CC_COL:  {1: 1, 2: 0},
                  ARL_COL: {1: 1, 2: 0}}
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col + '_bin'] = df[col].map(mapping).fillna(0).astype(int)
    log.append({
        'step': '4 · Binary encoding (formality flags)',
        'detail': ('RUT, Chamber of Commerce, and ARL columns recoded: '
                   '1 = formal (has it), 0 = informal (does not have it).'),
        'shape': df.shape
    })

    # ── STEP 5: Encode accounting records → 1=keeps records, 0=none ───────
    if ACC_COL in df.columns:
        df['has_accounting'] = df[ACC_COL].apply(lambda x: 0 if x == 5 else 1)
    log.append({
        'step': '5 · Accounting records encoding',
        'detail': ('P640: values 1-4 → 1 (keeps some form of records); '
                   'value 5 (No records) → 0.'),
        'shape': df.shape
    })

    # ── STEP 6: Feature engineering — Informality Index ───────────────────
    flag_cols = [c for c in [RUT_COL+'_bin', CC_COL+'_bin',
                              ARL_COL+'_bin', 'has_accounting'] if c in df.columns]
    if flag_cols:
        df['informality_index'] = 1 - df[flag_cols].mean(axis=1)
        df['informality_index'] = df['informality_index'].round(3)
    log.append({
        'step': '6 · Feature engineering — Informality Index',
        'detail': ('Composite score = 1 − mean(formality flags). '
                   'Range 0 (fully formal) to 1 (fully informal). '
                   f'Mean index: {df["informality_index"].mean():.3f}.'),
        'shape': df.shape
    })

    # ── STEP 7: Normalize numerical variables ──────────────────────────────
    from sklearn.preprocessing import StandardScaler
    scale_cols = [c for c in num_cols_present if c in df.columns]
    if scale_cols:
        scaler = StandardScaler()
        df[[c + '_scaled' for c in scale_cols]] = scaler.fit_transform(df[scale_cols])
    log.append({
        'step': '7 · Normalization (StandardScaler)',
        'detail': (f'Applied StandardScaler to: {", ".join(scale_cols)}. '
                   'Scaled columns have mean≈0 and std≈1.'),
        'shape': df.shape
    })

    # ── STEP 8: One-hot encode CIIU sector ─────────────────────────────────
    if SECTOR_COL in df.columns:
        dummies = pd.get_dummies(df[SECTOR_COL], prefix='sector', drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    log.append({
        'step': '8 · One-Hot Encoding — CIIU sector',
        'detail': (f'pd.get_dummies applied to {SECTOR_COL} (CIIU economic sector). '
                   f'Created {dummies.shape[1] if SECTOR_COL in df.columns else 0} dummy columns.'),
        'shape': df.shape
    })

    # Summary stats after engineering
    summary = {
        'original_records': original_shape[0],
        'original_cols': original_shape[1],
        'final_records': df.shape[0],
        'final_cols': df.shape[1],
        'informality_mean': round(df['informality_index'].mean(), 3) if 'informality_index' in df.columns else 'N/A',
        'informality_std': round(df['informality_index'].std(), 3) if 'informality_index' in df.columns else 'N/A',
        'fully_informal_pct': round((df['informality_index'] == 1.0).mean() * 100, 1) if 'informality_index' in df.columns else 'N/A',
    }

    return df, log, summary


@data_engineering_bp.route('/data-engineering')
def data_engineering():
    result = {}
    try:
        df_raw = pd.read_csv(DATA_PATH)
        df_clean, pipeline_log, summary = run_pipeline(df_raw)
        result['log'] = pipeline_log
        result['summary'] = summary
        result['sample'] = df_clean.head(5).to_html(
            classes='table table-sm', border=0, index=False)
        result['label_map'] = LABEL_MAP
    except FileNotFoundError:
        result['error'] = (
            'Dataset not found. Place emicron_sample.csv in /data. '
            'Download from: https://microdatos.dane.gov.co/index.php/catalog/832'
        )
    except Exception as e:
        result['error'] = f'Pipeline error: {str(e)}'

    return render_template('data_engineering.html', result=result)
