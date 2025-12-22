# Dynamic Agriculture Price Prediction

## Project Description

This project predicts the **modal price** of agricultural commodities using machine learning. It leverages historical market data, commodity attributes, and arrival dates to generate price forecasts and interactive visualizations for agricultural stakeholders.

## Technologies Used

- Python
- scikit-learn, XGBoost (modeling)
- pandas, NumPy (data processing)
- Plotly, Matplotlib (visualization)
- joblib (model persistence)

## Dataset

- Source: Nashik district, Maharashtra (AGMARKNET / data.gov.in)
- Features: `Market`, `Commodity`, `Variety`, `Grade`, `Arrival_Date`, `Commodity_Code`
- Target: `Modal_Price`
- Preprocessing: categorical encoding, date transformation, and feature engineering

## Features

- End‑to‑end data preprocessing pipeline
- ML models: Linear Regression and XGBoost
- Saved `.joblib` models for fast loading
- Streamlit UI with:
  - Price Prediction
  - Market Trends
  - Commodity Comparison

---

## Installation and Usage (Windows)

### 1. Clone the repository

git clone https://github.com/AnirbanBoral/dynamic-agriculture-price-prediction.git
cd dynamic-agriculture-price-prediction

### 2. Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate

You should now see `(.venv)` at the beginning of your terminal prompt.
Then run:

pip install -r requirements

### 3. Run the Streamlit app

python -m streamlit run app.py

### 4. Stop and rerun later

- Stop the app: press `Ctrl + C` in the terminal.
- Deactivate the virtual environment:


Next time you want to use the app:

cd dynamic-agriculture-price-prediction
.venv\Scripts\activate
python -m streamlit run app.py