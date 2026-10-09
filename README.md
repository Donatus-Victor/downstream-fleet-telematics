# 🛢️ Downstream Oil & Gas Intelligent Fleet Telematics Platform

An enterprise-grade End-to-End Machine Learning and Business Intelligence solution designed to optimize downstream distribution haulage, predict mechanical failures, and mitigate structural fuel theft across regional logistics corridors.

## 🚀 The Business Context & Core Sector Challenges Solved
Downstream logistics networks lose millions annually due to lack of visibility along distribution paths. This project solves three critical challenges faced by heavy haulage operators:
* **Refined Product Theft (Fuel Siphoning):** Identifies anomalous fuel drainage using real-time velocity and pressure sensor snapshots.
* **Unscheduled Roadside Breakdowns:** Employs an XGBoost Classifier trained on telemetry strain inputs to forecast roadside failure thresholds before they happen.
* **Inflexible Asset Allocation:** Implements automated routing bounds based on run-hour maintenance counters to shield worn hardware from long, punishing corridors.

## 🛠️ System Architecture & Folder Layout
```text
OIL&GAS-virtual-pipeline-ml/
├── data/
│   ├── raw_telematics.csv          # Generated raw dataset
│   └── processed_features.csv      # Cleaned and engineered features
├── notebooks/
│   ├── 01_data_generation_&_eda.ipynb
│   └── 02_model_training_&_evaluation.ipynb
├── src/
│   ├── data_pipeline.py            # Secure database connection & ETL logic
│   └── inference.py                # Serialized Machine Learning calculation engines
├── app.py                          # Streamlit data application frontend
├── requirements.txt                # Infrastructure dependencies
└── README.md                       # Documentation
```

## ⚙️ Quick Installation & Setup
1. **Clone the repository and mount dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your secure database keys (`.env`):**
   Create a hidden `.env` file in the root directory:
   ```text
   DB_USER=postgres
   DB_PASSWORD=your_secure_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=downstream_logistics_intel
   ```
3. **Initialize the real-time Streamlit analytics panel:**
   ```bash
   streamlit run app.py
   ```
