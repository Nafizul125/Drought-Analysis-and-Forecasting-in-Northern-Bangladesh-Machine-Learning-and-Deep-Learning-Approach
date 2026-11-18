# 🌦️ Drought Analysis and Long-Range Forecasting in Northern Bangladesh

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![GeoPandas](https://img.shields.io/badge/GIS-GeoPandas%20%7C%20IDW-2E7D32)](https://geopandas.org/)
[![Status](https://img.shields.io/badge/Research-Complete%20Pipeline-brightgreen)]()

An end-to-end **Time-Series, Machine Learning, Deep Learning, and GIS Spatial Analysis** framework for assessing historical meteorological droughts (1998–2023) and forecasting long-range drought dynamics (2024–2050) across **six weather stations in Northern Bangladesh** (Rajshahi, Ishwardi, Bogura, Syedpur, Rangpur, and Dinajpur).

> **Technical Contribution Note:** All computational pipelines in this repository—including data preprocessing, multi-scale drought index computation (SPI/SPEI/ZSI), wavelet-ADF stationarity analysis, ARIMA vs. SARIMA forecasting, 14-model ML/DL benchmarking, and GIS IDW spatial hazard mapping—were independently designed and implemented by **Md. Nafizul Sadik**.

---

## 📌 Project Overview

Northern Bangladesh—particularly the Barind Tract—faces recurrent slow-onset meteorological droughts driven by lower annual rainfall, high evapotranspiration, and monsoon variability. This project analyzes **26 years of monthly meteorological records (1,872 station-months)** across 7 climate parameters:
* **Parameters:** Rainfall (mm), Max Temperature (°C), Min Temperature (°C), Wind Speed (m/s), Sunshine Duration (hours), Relative Humidity (%), and Cloud Coverage (octas).
* **Accumulation Scales:** 1, 3, 6, 12, and 24 months.
* **Drought Indices:** Standardized Precipitation Index (**SPI**), Standardized Precipitation Evapotranspiration Index (**SPEI** using Hargreaves PET & 3-parameter Log-Logistic PWM), and Z-Score Index (**ZSI**).

---

## 🔄 Methodology Pipeline

1. **Data Quality Control & Imputation:** Fixed string/comma formatting errors and imputed multi-year gaps (e.g., Syedpur 1998–2002) using station-specific monthly climatology to preserve monsoon seasonality.
2. **Multi-Scale Drought Indexing:** Computed SPI (Gamma distribution), SPEI (Hargreaves PET + Log-Logistic PWM), and ZSI across 5 timescales and categorized severity into 5 drought classes (No Drought to Extreme Drought).
3. **Wavelet Decomposition & Stationarity Testing:** Applied 3-level **Daubechies-4 (`db4`) Discrete Wavelet Transform** ($A_3, D_3, D_2, D_1$) followed by **510 Augmented Dickey-Fuller (ADF)** tests.
4. **Long-Range Stochastic Forecasting (2024–2050):** Benchmarked non-seasonal **ARIMA** vs. **Seasonal ARIMA (SARIMA, $s=12$)** across all 42 station-variable pairs via stepwise AIC minimization, projecting 324 months forward.
5. **14-Model ML & DL Benchmarking (SPI-3 Prediction):** Engineered 1-, 2-, and 3-month lagged SPI-3 features (`SPI_3_lag1..3`) and evaluated **10 Machine Learning** models and **4 Deep Learning** sequence architectures (12-month look-back) using Accuracy, F1, MAE, RMSE, $R^2$, NSE, KGE, Kruskal-Wallis test, and **Taylor Diagrams**.
6. **GIS Spatial Hazard Mapping:** Generated 40 regional drought hazard maps clipped to **Rangpur & Rajshahi Divisions** (UTM Zone 45N / `EPSG:32645`) using **Inverse Distance Weighting (IDW)** interpolation.

---

## 📊 Key Results & Findings

### 1. Wavelet Stationarity (510 ADF Tests, $p < 0.05$)
* **Original Raw Series:** `68.6%` stationary
* **Approximation Trend ($A_3$):** `37.3%` stationary *(captures underlying non-stationarity)*
* **Detail Components ($D_1, D_2, D_3$):** `100.0%` stationary

### 2. ARIMA vs. SARIMA Forecasting (42 Station-Variable Pairs)
* **SARIMA outperformed ARIMA in `38 out of 42` pairs (`90.5%`)** based on AIC.
* Capturing monsoon seasonality shifted 2024–2050 forecasted drought frequencies by an average of **38.77 percentage points** compared to non-seasonal ARIMA.

### 3. Machine Learning & Deep Learning Benchmark (SPI-3 Testing Phase — Mean Across 6 Stations)

| Rank (by $R^2$) | Model | Category | Accuracy | F1-Score | MAE | RMSE | $R^2$ / NSE | Taylor Distance ($d$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Stacking Ensemble** *(XGB+MLP+SVR+RF)* | ML | 0.6532 | 0.6333 | 0.4505 | **0.5888** | **0.6466** | 0.3005 |
| **2** | **Multi-Layer Perceptron (MLP)** | ML | 0.6640 | 0.6506 | **0.4503** | 0.6130 | 0.6147 | **0.2202** *(Best)* |
| **3** | **Support Vector Regression (SVM)** | ML | **0.6720** | 0.6462 | 0.4641 | 0.6267 | 0.6007 | 0.3924 |
| **4** | **Linear Regression (LR)** | ML | 0.6613 | **0.6531** | 0.4699 | 0.6283 | 0.5953 | 0.3063 |
| **5** | **SGD Regressor** | ML | 0.6478 | 0.6356 | 0.4743 | 0.6309 | 0.5936 | 0.3273 |
| **6** | **Gradient Boosting (GB)** | ML | 0.6586 | 0.6410 | 0.4941 | 0.6525 | 0.5672 | 0.3457 |
| **7** | **XGBoost (XGB)** | ML | 0.6425 | 0.6289 | 0.5150 | 0.6731 | 0.5412 | 0.3708 |
| **8** | **Random Forest (RF)** | ML | 0.6344 | 0.5900 | 0.5330 | 0.6957 | 0.5082 | 0.5016 |
| **9** | **K-Nearest Neighbors (KNN)** | ML | 0.6156 | 0.5877 | 0.5624 | 0.7424 | 0.4390 | 0.5235 |
| **10** | **Bi-LSTM** *(Best DL by $R^2$)* | DL | 0.5706 | 0.5599 | 0.6308 | 0.8007 | 0.3433 | 0.4111 |
| **11** | **CNN-LSTM** | DL | 0.5763 | 0.5676 | 0.6421 | 0.8127 | 0.3210 | 0.4819 |
| **12** | **LSTM** *(Best DL by Accuracy)* | DL | 0.6017 | 0.5874 | 0.6279 | 0.8309 | 0.2952 | 0.4112 |
| **13** | **GRU** | DL | 0.5904 | 0.5829 | 0.6610 | 0.8590 | 0.2488 | 0.4655 |
| **14** | **Decision Tree (DT)** | ML | 0.5565 | 0.5484 | 0.7599 | 1.0043 | -0.0302 | 0.5171 |

### 4. Historical Hotspots (1998–2023) & 2024–2050 SARIMA Outlook (SPI-12)

| Station | Historical Drought Exposure (All Indices/Scales) | Recent Historical SPI-12 (2016–2023) | Forecasted SPI-12 (2024–2050) | Projected Trend |
| :--- | :---: | :---: | :---: | :--- |
| **Dinajpur** | **54.65%** *(Rank 1 — Most Prone)* | 61.46% | 47.22% | Moderate Decline |
| **Rangpur** | 53.03% *(Rank 2)* | 50.00% | **97.84%** | Perennial Drought Risk |
| **Ishwardi** | 52.96% *(Rank 3)* | 47.92% | **98.15%** | Perennial Drought Risk |
| **Bogura** | 51.75% *(Rank 4)* | 65.62% | 4.01% | Improvement |
| **Rajshahi** | 51.32% *(Rank 5)* | 50.00% | **98.77%** | Perennial Drought Risk |
| **Syedpur** | 47.75% *(Rank 6 — Least Prone)* | 52.08% | 32.10% | Moderate Decline |

---

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_Drought_Analysis_SARIMA_ML_DL.ipynb      # Data QC, SPI/SPEI/ZSI, Wavelet-ADF, SARIMA (to 2050), 10 ML + 4 DL models
│   └── 02_GIS_Spatial_Drought_Hazard_Mapping.ipynb # GADM shapefile clipping, IDW spatial interpolation & 40 hazard maps
├── requirements.txt                                 # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack & Setup

* **Core & Data Science:** Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn
* **Time-Series & Signal Processing:** `statsmodels` (ARIMA/SARIMAX, ADF), `pmdarima`, `PyWavelets`
* **Machine Learning & Deep Learning:** `scikit-learn`, `XGBoost`, `TensorFlow` / `Keras`
* **Geospatial / GIS:** `GeoPandas`, `Rasterio`, `Shapely`, `mapclassify`, GADM v4.1 Shapefiles

### Quick Start
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order (recommended on **Google Colab** with GPU enabled for the deep learning models):
   * First run `notebooks/01_Drought_Analysis_SARIMA_ML_DL.ipynb` with the meteorological CSV dataset to generate cleaned data, indices, forecasts, and GIS summary tables.
   * Then run `notebooks/02_GIS_Spatial_Drought_Hazard_Mapping.ipynb` using the exported GIS CSV files (`GIS_1_Station_Hazard_Overview.csv`, `GIS_2_Period_Detailed_Data_Long.csv`, `GIS_3_MapReady_SPI12_SPEI12_ByPeriod.csv`) to generate high-resolution spatial hazard maps.

---

## 👥 Research Collaborators

This work is part of the collaborative research study: *"Drought Analysis and Forecasting in Northern Bangladesh: An Ensemble Machine Learning and Deep Learning Approach with SARIMA-Based Long-Range Forecasting"*.

* **Md. Nafizul Sadik** — *Lead Technical Implementation (ML/DL Modeling, SARIMA Forecasting, Wavelet-ADF Analysis & GIS Spatial Mapping)*, Department of CSE, BRAC University
* **Readanul Hasan Remon** — Department of Mechanical Engineering, IUBAT
* **Md. Mohaimanul Rabbi** — Department of Mechanical Engineering, IUBAT
* **Md. Alomgir Hossain** — Department of CSE, IUBAT
