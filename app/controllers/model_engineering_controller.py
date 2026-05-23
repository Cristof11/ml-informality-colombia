from flask import Blueprint, render_template
import pandas as pd
import numpy as np
import json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             silhouette_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

model_engineering_bp = Blueprint('model_engineering', __name__)

DATA_PATH    = os.path.join(os.path.dirname(__file__), '../../data/emicron_sample.csv')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '../../app/static/metrics.json')

FEATURES = ['RUT_bin','CC_bin','ACC_bin','ARL_bin',
            'WORKERS','MONTHS_OPERATING','MONTHLY_SALES','CIIU_SECTOR']

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    X['MONTHLY_SALES'] = np.log1p(X['MONTHLY_SALES'])
    X = X.fillna(X.median())
    y = (df['INFORMALITY_INDEX'] >= 0.75).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_scaled, X_train, X_test, y_train, y_test, scaler, df

def run_models():
    X_scaled, X_train, X_test, y_train, y_test, scaler, df = prepare_data()

    # ── K-Means ────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled[:5000], cluster_labels[:5000])
    df['CLUSTER'] = cluster_labels
    cluster_profile = df.groupby('CLUSTER').agg(
        records=('INFORMALITY_INDEX','count'),
        inf_mean=('INFORMALITY_INDEX','mean'),
        no_rut=('RUT_bin', lambda x: round((x==0).mean()*100,1)),
        no_cc=('CC_bin',   lambda x: round((x==0).mean()*100,1)),
        no_arl=('ARL_bin', lambda x: round((x==0).mean()*100,1)),
        med_sales=('MONTHLY_SALES','median')
    ).round(3).reset_index()

    cluster_names = {0:'Deeply Informal', 1:'Near-Formal',
                     2:'Fully Informal',  3:'Borderline Informal'}
    cluster_profile['name'] = cluster_profile['CLUSTER'].map(cluster_names)

    # ── Logistic Regression ────────────────────────────────────────────
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42,
                            class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr  = round(accuracy_score(y_test, y_pred_lr)*100, 2)
    f1_lr   = round(f1_score(y_test, y_pred_lr, average='weighted')*100, 2)
    report_lr = classification_report(y_test, y_pred_lr,
                                       target_names=['Formal','Informal'],
                                       output_dict=True)

    # ── Random Forest ─────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=42, class_weight='balanced',
                                 n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf  = round(accuracy_score(y_test, y_pred_rf)*100, 2)
    f1_rf   = round(f1_score(y_test, y_pred_rf, average='weighted')*100, 2)
    report_rf = classification_report(y_test, y_pred_rf,
                                       target_names=['Formal','Informal'],
                                       output_dict=True)
    fi = sorted(zip(FEATURES, rf.feature_importances_),
                key=lambda x: x[1], reverse=True)

    # ── Prediction examples ────────────────────────────────────────────
    ex_formal   = scaler.transform([[1,1,1,1,3,48,np.log1p(5000000),3]])
    ex_informal = scaler.transform([[0,0,0,0,1,6,np.log1p(500000),1]])
    ex_gateway  = scaler.transform([[1,0,0,0,2,24,np.log1p(1500000),3]])

    preds = {
        'formal':   {'lr': int(lr.predict(ex_formal)[0]),
                     'rf': int(rf.predict(ex_formal)[0]),
                     'lr_prob': round(lr.predict_proba(ex_formal)[0][1]*100,1),
                     'rf_prob': round(rf.predict_proba(ex_formal)[0][1]*100,1)},
        'informal': {'lr': int(lr.predict(ex_informal)[0]),
                     'rf': int(rf.predict(ex_informal)[0]),
                     'lr_prob': round(lr.predict_proba(ex_informal)[0][1]*100,1),
                     'rf_prob': round(rf.predict_proba(ex_informal)[0][1]*100,1)},
        'gateway':  {'lr': int(lr.predict(ex_gateway)[0]),
                     'rf': int(rf.predict(ex_gateway)[0]),
                     'lr_prob': round(lr.predict_proba(ex_gateway)[0][1]*100,1),
                     'rf_prob': round(rf.predict_proba(ex_gateway)[0][1]*100,1)},
    }

    return {
        'kmeans': {
            'k': 4, 'silhouette': round(sil,3),
            'profiles': cluster_profile.to_dict('records'),
            'n_records': len(df),
        },
        'logistic': {
            'accuracy': acc_lr, 'f1': f1_lr,
            'report': report_lr,
            'train_size': len(X_train), 'test_size': len(X_test),
            'params': {'C': 1.0, 'max_iter': 500, 'solver':'lbfgs',
                       'class_weight':'balanced'},
        },
        'random_forest': {
            'accuracy': acc_rf, 'f1': f1_rf,
            'report': report_rf,
            'train_size': len(X_train), 'test_size': len(X_test),
            'params': {'n_estimators':100,'max_depth':8,
                       'class_weight':'balanced'},
            'feature_importance': fi,
        },
        'predictions': preds,
        'dataset': {
            'total': len(df),
            'train': len(X_train),
            'test':  len(X_test),
            'features': FEATURES,
            'target': 'INFORMALITY_INDEX >= 0.75',
            'informal_pct': round((df['INFORMALITY_INDEX']>=0.75).mean()*100,1),
        }
    }

@model_engineering_bp.route('/model-engineering')
def model_engineering():
    try:
        data = run_models()
    except Exception as e:
        data = {'error': str(e)}
    return render_template('model_engineering.html', data=data)
