# customer_churn_analysis-009
# 📊 Customer Churn Analysis Project

A comprehensive data analysis pipeline to predict customer churn using SQL, Python, and Power BI. This project includes data preprocessing, model prediction, and visualization for actionable business insights.

---

## 📁 Files Overview

| File Name | Description |
|---------|-------------|
| `Customer_Data.csv` | Raw customer dataset used as input |
| `vw_JoinData.csv` | Cleaned and joined dataset after SQL transformation |
| `vw_ChurnData.csv` | Subset of data focused on churn-related features |
| `churn_analysis.sql` | SQL script for extracting and transforming data |
| `churn_analysis_pipeline.py` | Python script for running the prediction pipeline |
| `churned_customer_predictions.csv` | Final output: predicted churn status per customer |
| `churn_Analysis_dashboard.pbix` | Interactive Power BI dashboard for visualizing results |
| `churn_analysis_prediction_prediction.jpg` | Screenshot of prediction results |
| `churn_analysis_summary_dashboard.jpg` | Screenshot of summary dashboard |

---

## 🔧 Workflow Summary

1. **Data Extraction**:  
   Use `churn_analysis.sql` to extract and clean data from source systems → outputs `vw_JoinData.csv` and `vw_ChurnData.csv`.

2. **Prediction Pipeline**:  
   Run `churn_analysis_pipeline.py` (Python) to process data and generate predictions → outputs `churned_customer_predictions.csv`.

3. **Visualization**:  
   Open `churn_Analysis_dashboard.pbix` in Power BI to explore:
   - Churn rate trends
   - High-risk customer segments
   - Model performance metrics

4. **Reporting**:  
   Use screenshots (`*.jpg`) to share key insights with stakeholders.

---

## 🚀 How to Use

- **For Analysts**:  
  Open `churn_Analysis_dashboard.pbix` in Power BI Desktop to interact with visuals.

- **For Developers/Modelers**:  
  Review `churn_analysis_pipeline.py` to understand logic and extend the model.

- **For Business Users**:  
  Refer to `churned_customer_predictions.csv` for list of at-risk customers.

---

## 🛠️ Future Enhancements

- Add automated scheduling for daily updates
- Integrate model with CRM system
- Include feature importance analysis
- Add documentation for each step

---

💡 *This project demonstrates end-to-end data analysis: from raw data to business-ready insights.*
