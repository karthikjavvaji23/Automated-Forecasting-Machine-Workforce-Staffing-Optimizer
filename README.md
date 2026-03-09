# Automated Workforce Demand Forecasting System

Python | Time Series Forecasting | Machine Learning | Streamlit

This project implements an automated forecasting pipeline designed to analyze historical operational demand data, compare multiple forecasting models, generate future demand predictions, and translate those forecasts into workforce staffing requirements.

The system demonstrates how time-series analytics and machine learning can be used to support operational planning in service environments such as call centers, customer support operations, and public service systems.

---

## Project Overview

Accurate demand forecasting is essential for workforce planning. Incorrect staffing forecasts can lead to overstaffing, operational inefficiencies, or understaffing that increases customer wait times and employee workload.

This project builds an automated forecasting system that:

• analyzes historical service demand data  
• benchmarks multiple forecasting models  
• generates a six-month demand forecast  
• converts predicted demand into staffing requirements  

The project was implemented using Python and deployed through an interactive Streamlit application.

---

## Forecasting Pipeline

The automated workflow follows a structured data science pipeline:
Dataset Upload
↓
Data Inspection
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Model Benchmarking
↓
Demand Forecast Generation
↓
Workforce Staffing Optimization


---

## Forecasting Models

The system benchmarks multiple statistical and machine learning forecasting models:

• Seasonal Naïve  
• Moving Average  
• Exponential Smoothing  
• Linear Regression  
• XGBoost  
• Prophet  

The best-performing model is automatically selected based on forecasting accuracy metrics.

---

## Model Evaluation Metrics

Model performance is evaluated using standard forecasting accuracy metrics:

• **wMAPE** — Weighted Mean Absolute Percentage Error  
• **MAE** — Mean Absolute Error  
• **RMSE** — Root Mean Squared Error  

These metrics measure how closely model predictions match the observed demand.

---

## Workforce Staffing Calculation

Forecasted demand is converted into staffing requirements using workforce management parameters:

• Average Handle Time (AHT)  
• Occupancy Rate  
• Shrinkage  

The staffing formula used in the system:

---

## Forecasting Models

The system benchmarks multiple statistical and machine learning forecasting models:

• Seasonal Naïve  
• Moving Average  
• Exponential Smoothing  
• Linear Regression  
• XGBoost  
• Prophet  

The best-performing model is automatically selected based on forecasting accuracy metrics.

---

## Model Evaluation Metrics

Model performance is evaluated using standard forecasting accuracy metrics:

• **wMAPE** — Weighted Mean Absolute Percentage Error  
• **MAE** — Mean Absolute Error  
• **RMSE** — Root Mean Squared Error  

These metrics measure how closely model predictions match the observed demand.

---

## Workforce Staffing Calculation

Forecasted demand is converted into staffing requirements using workforce management parameters:

• Average Handle Time (AHT)  
• Occupancy Rate  
• Shrinkage  

The staffing formula used in the system:
Agents = (Forecasted Demand × AHT)
/ (3600 × Occupancy × (1 − Shrinkage))


This allows operational leaders to translate demand forecasts into staffing needs.

---

## Dataset

The primary dataset used in this project is:

**NYC 311 Service Requests Dataset**

This dataset contains time-stamped service requests submitted by residents for non-emergency city services. The request creation timestamp is used to transform event-level records into hourly demand counts suitable for forecasting.

Dataset source:  
https://opendata.cityofnewyork.us

For system validation, the pipeline was also tested using:

**Montgomery County 911 Emergency Calls Dataset**

https://data.montgomerycountymd.gov

<p align="center">
  <img src="Picture1.gif" width="900">
</p>


---

## Running the Application

Install dependencies:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run app.py


The application allows users to:

• upload operational datasets  
• analyze demand patterns  
• compare forecasting models  
• generate demand forecasts  
• calculate workforce staffing requirements  

---

## Key Technologies

Python  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
Prophet  
Statsmodels  
Matplotlib  
Streamlit  

---

## Project Outcome

This project demonstrates how automated forecasting pipelines can support operational decision-making. By integrating time-series analytics, machine learning models, and workforce planning logic, the system provides a scalable approach for predicting demand and optimizing staffing levels.

The forecasting pipeline can be applied across different service environments including call centers, emergency response systems, and customer support operations.

---
