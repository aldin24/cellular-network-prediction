📡 Predicting Cellular Network Signal Instability Using Machine Learning & Deep Learning

This repository contains the full implementation of a research study comparing traditional machine learning models and deep learning sequence models for predicting LTE downlink throughput stability using a publicly available Kaggle dataset.

The goal is to evaluate whether deep learning (LSTM, GRU, CNN, CNN–LSTM) can outperform classical ML methods (Linear Regression, Decision Trees, Random Forests, XGBoost) when applied to small real-world cellular measurement datasets.

🚀 1. Overview

Cellular networks generate large volumes of radio measurements such as RSRP, RSRQ, RSSI, CQI, SNR, and throughput. Predicting downlink throughput instability helps operators:

1. optimize radio resources

2. detect bad coverage areas

3. improve Quality of Service (QoS)

4. support self-optimizing networks

This project evaluates multiple ML & DL models to determine which architecture best predicts LTE downlink Mbps.

🔧 2. Installation
1. Clone the repository
git clone https://github.com/<your-username>/cellular-network-prediction.git
cd cellular-network-prediction

2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Install TensorFlow for macOS (Apple Silicon)
pip install tensorflow-macos tensorflow-metal

📦 3. Dataset

The project uses the public Kaggle dataset:

📊 aeryss/lte-dataset
(LTE cellular drive-test measurements)

Downloaded automatically through:

from data_loader import load_kaggle_dataset

🤖 4. Models Implemented
Machine Learning Models
Model	File	Notes
Linear Regression	linear_regression.py	Baseline
Polynomial Regression	integrated	Adds nonlinearity
Decision Tree	decision_tree.py	Captures nonlinear thresholds
Random Forest	random_forest.py	Reduces overfitting
XGBoost	xgboost_model.py	⭐ Best performer
Deep Learning Models
Model	File	Notes
LSTM	lstm.py	Temporal modeling
GRU	gru.py	Lighter version of LSTM
CNN	cnn.py	Extracts spatial features
CNN–LSTM hybrid	cnn_lstm.py	Spatial + Temporal

🧠 5. Running Experiments
Run classical ML experiments
python main.py

Run deep learning experiments
python main_deep.py


R²

Graph showing prediction vs actual throughput
