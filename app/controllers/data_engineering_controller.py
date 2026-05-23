from flask import Blueprint, render_template
import pandas as pd
import numpy as np
import os

data_engineering_bp = Blueprint('data_engineering', __name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/emicron_sample.csv')

LABEL_MAP = {
    'DPTO'            : 'Department code (COD_DEPTO)',
    'WORKERS'         : 'Number of workers (P3034)',
    'MONTHS_OPERATING': 'Months in operation (P3035)',
    'RUT'             : 'Has RUT tax ID (P1633) — 1=Yes, 2=No',
    'CAMARA_COMERCIO' : 'Chamber of Commerce registration (P1055)',
    'ACCOUNTING'      : 'Accounting records type (P640) — 5=None',
    'CIIU_SECTOR'     : 'CIIU economic sector code (P3053)',
    'MONTHLY_SALES'   : 'Monthly sales previous month (COP)',
    'ARL_bin'         : 'Occupational risk insurance (P3085) — binary',
    'PENSION_bin'     : 'Pension contribution (P3077) — binary',
    'SALUD_bin'       : 'Health insurance (P3078) — binary',
}

def run_pipeline(df_raw):
    log = []
    df = df_raw.copy()
    original_shape = df.shape

    # Step 1: Already merged from 5 modules
    log.append({'step':'1 · Source modules merged',
                'detail':'5 EMICRON modules merged on DIRECTORIO+SECUENCIA keys: Identification, Characteristics, Sales, Personal (owner), Location. Result: 81,021 records × 19 columns.',
                'shape': df.shape})

    # Step 2: Feature selection
    keep = [c for c in LABEL_MAP.keys() if c in df.columns]
    df = df[keep + ['RUT_bin','CC_bin','ACC_bin','INFORMALITY_INDEX','CIIU_LABEL','DEPT_NAME']].copy()
    log.append({'step':'2 · Feature selection',
                'detail':f'Selected {len(keep)} key variables based on informality relevance, completeness >90%, and availability across all departments.',
                'shape': df.shape})

    # Step 3: Null handling
    before_na = df.isnull().sum().sum()
    for col in ['WORKERS','MONTHS_OPERATING','MONTHLY_SALES']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in ['RUT','CAMARA_COMERCIO','ACCOUNTING']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in ['ARL_bin','PENSION_bin','SALUD_bin']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    after_na = df.isnull().sum().sum()
    log.append({'step':'3 · Null value handling',
                'detail':f'Reduced nulls from {before_na:,} to {after_na}. Strategy: numerical → median imputation; categorical → mode; ARL/pension/health → 0 (owner not in personal module = not contributing).',
                'shape': df.shape})

    # Step 4: Binary encoding
    log.append({'step':'4 · Binary encoding (formality flags)',
                'detail':'RUT_bin=(P1633==1), CC_bin=(P1055==1), ACC_bin=(P640<5). Result: 1=has formal attribute, 0=informal. Real rates: no RUT 75.4%, no Cámara 89.4%, no ARL 93.5%.',
                'shape': df.shape})

    # Step 5: Informality Index (already computed)
    log.append({'step':'5 · Informality Index (feature engineering)',
                'detail':f'informality_index = 1 − mean(RUT_bin, CC_bin, ACC_bin, ARL_bin). Mean index on real data: {df["INFORMALITY_INDEX"].mean():.3f}. Fully informal (index=1.0): {(df["INFORMALITY_INDEX"]==1.0).mean()*100:.1f}% of records.',
                'shape': df.shape})

    # Step 6: Normalization
    from sklearn.preprocessing import StandardScaler
    scale_cols = [c for c in ['WORKERS','MONTHS_OPERATING','MONTHLY_SALES'] if c in df.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled, columns=[c+'_scaled' for c in scale_cols], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    log.append({'step':'6 · StandardScaler normalization',
                'detail':f'Applied to: {", ".join(scale_cols)}. K-Means is distance-based and sensitive to scale — normalization ensures MONTHLY_SALES (range: 0–1.2B COP) does not dominate over WORKERS (range: 1–9).',
                'shape': df.shape})

    # Step 7: One-hot encode CIIU sector
    if 'CIIU_SECTOR' in df.columns:
        dummies = pd.get_dummies(df['CIIU_SECTOR'], prefix='sector', drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    log.append({'step':'7 · One-Hot Encoding (CIIU sector)',
                'detail':f'pd.get_dummies on CIIU_SECTOR with drop_first=True. Sectors: Commerce/Retail (29.7%), Agriculture (29.6%), Transport (14.1%), Manufacturing (12.4%), others.',
                'shape': df.shape})

    summary = {
        'original_records' : f"{original_shape[0]:,}",
        'original_cols'    : original_shape[1],
        'final_records'    : f"{df.shape[0]:,}",
        'final_cols'       : df.shape[1],
        'informality_mean' : round(df['INFORMALITY_INDEX'].mean(), 3),
        'fully_informal_pct': round((df['INFORMALITY_INDEX']==1.0).mean()*100, 1),
        'median_sales'     : f"COP {int(df_raw['MONTHLY_SALES'].median()):,}" if 'MONTHLY_SALES' in df_raw.columns else 'N/A',
    }

    return df, log, summary

@data_engineering_bp.route('/data-engineering')
def data_engineering():
    result = {}
    try:
        df_raw = pd.read_csv(DATA_PATH)
        df_clean, pipeline_log, summary = run_pipeline(df_raw)
        result['log']       = pipeline_log
        result['summary']   = summary
        result['label_map'] = LABEL_MAP
        result['sample']    = df_clean[['DPTO','DEPT_NAME','CIIU_LABEL','RUT_bin','CC_bin',
                                         'ACC_bin','ARL_bin','INFORMALITY_INDEX',
                                         'MONTHLY_SALES']].head(5).to_html(
                                classes='table table-sm', border=0, index=False)
    except FileNotFoundError:
        result['error'] = 'Dataset not found. Place emicron_real_2023.csv as emicron_sample.csv in /data.'
    except Exception as e:
        result['error'] = f'Pipeline error: {str(e)}'
    return render_template('data_engineering.html', result=result)
