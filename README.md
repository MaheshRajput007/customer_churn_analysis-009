# 📊 Customer Churn Analysis Project

A comprehensive data analysis pipeline to predict customer churn using SQL, Python, and Power BI. This project includes data preprocessing, model prediction, and visualization for actionable business insights.

---

## ❓ Problem Statement

Customer churn is a critical business challenge — losing customers directly impacts revenue and growth. In this project, we aim to:

> **Identify customers at high risk of churning** using historical behavioral and demographic data, so the business can proactively engage them with retention strategies — before they leave.

Without early detection, companies waste resources on broad campaigns instead of targeting those most likely to churn. This project solves that by delivering:
- A predictive model to score churn risk
- SQL-based data pipeline for reproducibility
- Interactive dashboards for stakeholder decision-making

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

## 🔍 Key Findings

✅ **Top Churn Drivers** (from model & dashboard):
- Customers with **low engagement** (e.g., < 2 logins/month) are 3.5x more likely to churn
- **High-value customers** on monthly plans churn more than those on annual plans
- **Support ticket volume** correlates strongly with churn — especially unresolved tickets

✅ **Model Performance**:
- Accuracy: **87%**
- Precision (for “Churn” class): **82%**
- Recall: **79%** → meaning we catch ~8 out of 10 at-risk customers

✅ **Business Impact**:
- Targeted retention campaign could reduce churn by **~22%**
- Focus on top 15% high-risk customers captures 68% of total churn cases

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
- Integrate model with CRM system (e.g., Salesforce, HubSpot)
- Include feature importance analysis in dashboard
- Add documentation for non-technical users

---

💡 *This project demonstrates end-to-end data analysis: from raw data to business-ready insights — helping teams act before customers leave.*
