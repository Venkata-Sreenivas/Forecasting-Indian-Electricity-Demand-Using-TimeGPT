# ⚡ Indian Electricity Demand Forecasting using TimeGPT and LightGBM

This project aims to **forecast daily electricity demand for Indian states** using two approaches —  
1. **Nixtla's TimeGPT**, a foundation model for time series forecasting, and  
2. **LightGBM (LGBM)**, a feature-based machine learning model.  

The objective is to evaluate the predictive performance of these models on a real-world dataset and provide insights for energy demand planning and optimization in India.

---

## 🌍 Project Background

Electricity demand forecasting plays a crucial role in energy management and policy planning. Accurate forecasts help:
- Reduce energy wastage
- Manage grid loads effectively
- Plan maintenance schedules
- Support renewable integration and power trading decisions

Traditional models like ARIMA or SARIMA often struggle with large-scale, multi-entity forecasting. This project explores **modern AI-based models (TimeGPT)** and compares them with a classical ML-based baseline (**LightGBM**) to identify the best-performing approach.

---

## 🧾 Dataset Description

- **Source:** State-wise electricity consumption data (long-format CSV file)  
- **Original File:** `long_data_.csv`  
- **Processed File:** `indian_electricity_demand.csv`  
- **Key Columns:**
  | Column | Description |
  |:-------|:-------------|
  | `States` | Name of the Indian state |
  | `Dates` | Timestamp of measurement |
  | `Usage` | Recorded electricity usage (in MWh) |
  | `unique_id` | Renamed column for state ID |
  | `ds` | Reformatted date column |
  | `y` | Target variable (electricity demand) |

### 🧹 Preprocessing Steps
- Converted `'Dates'` to datetime format  
- Renamed columns (`States → unique_id`, `Usage → y`)  
- Sorted and cleaned data by state and date  
- Removed duplicates and handled missing values using **linear interpolation**  
- Exported clean dataset for further analysis  

---

## 🧠 Methodology Overview

### Step 1️⃣: Data Preparation
- Data loaded and preprocessed in **Google Colab**
- Missing timestamps filled using `asfreq('D')` (daily frequency)
- Missing `y` values interpolated linearly

### Step 2️⃣: Anomaly Detection
- Leveraged **Nixtla’s TimeGPT** anomaly detection API
- Identified outliers in daily electricity consumption trends

### Step 3️⃣: Forecasting Approaches
#### **A. TimeGPT Forecasting**
- Pre-trained model by **Nixtla**
- API call-based forecasting for 30-day horizon (`h=30`)
- Fine-tuned with:
  - `finetune_steps = 60`
  - `finetune_loss = "mae"`
  - Model used: `timegpt-1-long-horizon`

#### **B. LightGBM Forecasting**
- Traditional ML model trained using lag-based features
- Engineered time-based features:
  - Day, Month, Day of Week
  - Lag values (1, 2, 3, 7 days)
  - Rolling mean of 7 days
- Trained per state to capture regional patterns

### Step 4️⃣: Model Evaluation
Metrics used:
| Metric | Description | Goal |
|:-------|:-------------|:----|
| MAE | Mean Absolute Error | ↓ Lower |
| RMSE | Root Mean Squared Error | ↓ Lower |
| SMAPE | Symmetric Mean Absolute Percentage Error | ↓ Lower |

### Step 5️⃣: Visualization
Compared actual vs predicted demand for individual states (e.g., Andhra Pradesh).

---

## 🔧 Technologies & Dependencies

### Core Libraries
- `pandas`, `numpy` → Data manipulation  
- `matplotlib` → Visualization  
- `lightgbm`, `scikit-learn` → Machine learning  
- `nixtla`, `mlforecast`, `utilsforecast` → TimeGPT forecasting  
- `google.colab` → File upload/download utilities  

### Installation
```bash
%pip install --upgrade pip
%pip install nixtla utilsforecast mlforecast lightgbm scikit-learn pandas matplotlib
