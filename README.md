# Intelligent System for Detecting Commercial Informality Patterns in Colombian Cities

**Machine Learning Project · CRISP-ML(Q) Methodology**  
**Dataset:** EMICRON 2023 — Encuesta de Micronegocios (DANE)

---

## 📋 Project Overview

This project applies **K-Means clustering** to the DANE EMICRON 2023 dataset to identify and characterize commercial informality patterns across Colombian cities. It follows the **CRISP-ML(Q)** methodology and is deployed as a Flask web application.

---

## 🗂️ Project Structure

```
ml-informality/
├── app.py                          # Flask entry point
├── requirements.txt                # Python dependencies
├── README.md
├── .gitignore
├── data/
│   └── emicron_sample.csv          # EMICRON 2023 sample dataset
├── notebooks/
│   └── eda_emicron.ipynb           # Exploratory Data Analysis
└── app/
    ├── controllers/                # Flask blueprints (one per menu)
    │   ├── home_controller.py
    │   ├── crisp_controller.py
    │   ├── business_controller.py
    │   ├── data_understanding_controller.py
    │   └── data_engineering_controller.py
    ├── templates/                  # Jinja2 HTML templates
    │   ├── base.html
    │   ├── home.html
    │   ├── crisp.html
    │   ├── business.html
    │   ├── data_understanding.html
    │   └── data_engineering.html
    └── static/
        ├── css/style.css
        └── js/main.js
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-informality-colombia.git
cd ml-informality-colombia

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
# Open: http://127.0.0.1:5000
```

---

## 🌐 Live Demo

Deployed on **Render.com**: `https://ml-informality-colombia.onrender.com`

---

## 📦 Dataset

- **Source:** DANE — Encuesta de Micronegocios (EMICRON) 2023
- **URL:** https://microdatos.dane.gov.co/index.php/catalog/832
- **Records:** 81,018 micro-businesses across 24 departments + Bogotá D.C.
- **Key Variables:** RUT ownership, Chamber of Commerce registration, ARL contributions, accounting records, monthly sales, CIIU sector, location type

---

## 🔬 Methodology

**CRISP-ML(Q)** — Cross-Industry Standard Process for Machine Learning with Quality Assurance:
1. Business Understanding
2. Data Understanding
3. Data Engineering
4. Model Engineering *(future phase)*
5. Model Quality Assurance *(future phase)*
6. Deployment *(future phase)*

---

## 📚 References

- Schmid, Wurst & Wirth (2020). *CRISP-ML(Q)*
- DANE (2023). *EMICRON — Encuesta de Micronegocios*
- DANE (2024). *GEIH — Gran Encuesta Integrada de Hogares Q2 2024*
