# Red Bull F1 Analysis

## Overview
This project provides an in-depth analysis of Formula 1 racing performance with a focus on Red Bull Racing. Utilizing a data warehouse approach, ETL pipelines, regression analysis, and forecasting models, the project aims to understand factors influencing driver race positions and performance metrics.

## Data Model
The project's data model is structured around a star schema for efficient querying and analysis, with a central fact table that captures race performance metrics. The model comprises various dimension tables that include details about drivers, teams, race events, venues, and seasons.

### Fact Table
- `Race Performance`: Stores quantitative data from each race, with relationships to all dimension tables.

### Dimension Tables
- `Driver Dimension`: Contains detailed information about each driver.
- `Team Dimension`: Provides information about each F1 team.
- `Race Event Dimension`: Holds data for each race event, including weather conditions and schedules.
- `Venue Dimension`: Details the race venues, including location and physical characteristics.
- `Season Dimension`: Captures season-related information, such as timelines and descriptions.
  
![DataModelRD](https://github.com/roniasatourian/Red-bull-F1-Case/assets/36686617/22e70320-59d1-4fd1-94cc-a890c901f9ff)

## ETL Pipeline
The ETL pipeline is implemented to process daily CSV files through a sequence of transformations and finally load the data into Google BigQuery for analysis. The current pipeline utilizes Python and the Apache Airflow framework, with a proposed production pipeline including AWS services for scaling.

![Current_pipeline](https://github.com/roniasatourian/Red-bull-F1-Case/assets/36686617/906d6896-2f6f-4eb2-b6a0-9c5c1b3d3d3e)

## Regression Analysis
A regression analysis is conducted to evaluate the significance of various factors on the drivers' final race positions not for prediction! Feature selection was performed using Lasso to penalize less important variables and reduce multicollinearity using the Variance Inflation Factor (VIF).

![Regression Analysis](https://github.com/roniasatourian/Red-bull-F1-Case/assets/36686617/bc0ea681-237c-4057-9418-ed9d2f7a7d19)

## Forecasting
The ARIMA model was utilized for forecasting future performance by automatically selecting p,d, and q hyperparameters with feature engineering conducted in Python and the model built using the R programming language. 

![Forecasting](https://github.com/roniasatourian/Red-bull-F1-Case/assets/36686617/4e94ef33-39ec-4a02-b1b0-6e0df2a841c6)

## Current Technologies Used
- Python
- Apache Airflow
- Google BigQuery
- R
- Tableau

## Overall Proposed Corporate Pipeline

![Production_pipeline](https://github.com/roniasatourian/Red-bull-F1-Case/assets/36686617/d328d56c-c049-4b2d-a79e-17f50c3a519a)

## Proposed Technologies for Corporate
- Python
- Apache Spark, Hadoop
- Apache Airflow
- AWS S3
- AWS Glue
- AWS Redshift
- AWS Sagemaker
- AWS EC2
- PostgreSQL
- Azure OpenAI LLM
- HTML/CSS/JavaScript
- FastAPI





