# Airline Passenger Forecasting — GRU | Deep Learning 📈✈️

A deep learning time series forecasting model using a GRU (Gated Recurrent Unit) 
network to predict monthly airline passenger numbers. Trained on the classic 
Box-Jenkins airline dataset with MinMax normalization and a 12-month lookback window.

---

## 🧩 Overview

This project uses a **Gated Recurrent Unit (GRU)** neural network to learn temporal 
patterns in historical airline passenger data and forecast future passenger counts. 
GRUs are a lightweight alternative to LSTMs that handle sequential data efficiently 
by using update and reset gates.

---

## 📦 Dataset

- **Source:** [Jason Brownlee's Datasets (GitHub)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)
- **Description:** Monthly airline passenger counts from 1949 to 1960
- **Total records:** 144 months
- Loaded directly from GitHub — no local file needed

---

## 🔬 Methodology

### 1. Data Preparation
- Normalized using **MinMaxScaler** to range [0, 1]
- Sequences created with a **lookback window of 12 months** (past 12 months → predict next month)
- **Train/Test split:** 80% training, 20% testing

### 2. GRU Model Architecture
```
Input (12 timesteps, 1 feature)
    ↓
GRU(50 units, activation='relu')
    ↓
Dense(1)  ← regression output
```

### 3. Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Validation | X_test / y_test |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **MSE** | 3442.28 |
| **RMSE** | 58.67 |
| **MAE** | 46.86 |

Predictions closely follow the actual passenger trend, capturing the seasonal 
patterns in the data.

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.10+
- TensorFlow / Keras
- scikit-learn, pandas, numpy, matplotlib

### Install Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Run
```bash
jupyter notebook Airline_Passenger_Forecasting_GRU.ipynb
```

Run all cells in order. The dataset is fetched automatically from GitHub — no 
local files required.

---

## 📁 File Structure

```
Airline-Passenger-Forecasting-GRU/
├── Airline_Passenger_Forecasting_GRU.ipynb   # Main notebook
└── README.md
```

---

## 📝 License

MIT License — for academic use.
