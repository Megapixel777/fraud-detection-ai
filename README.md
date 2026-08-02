# Fraud Detection AI

End-to-end Fraud Detection System using **Databricks**, **PySpark**, **Machine Learning**, **FastAPI** and **AI Agents**.

## Project Overview

This project aims to build a complete fraud detection platform capable of:

* Ingesting financial transaction data.
* Processing data using PySpark.
* Building a Medallion Architecture (Bronze, Silver and Gold layers).
* Training Machine Learning models to detect fraudulent transactions.
* Serving predictions through a REST API.
* Explaining fraud decisions using an AI Agent.
* Visualizing fraud metrics in Power BI.

The project follows software engineering best practices, including version control with Git, modular code organization and reproducible environments.

---

## Project Structure

```text
fraud-detection-ai/
│
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   ├── transformation/
│   ├── features/
│   ├── ml/
│   ├── api/
│   ├── ai_agent/
│   └── utils/
│
├── tests/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technology Stack

* Python
* PySpark
* Databricks
* Pandas
* Scikit-learn
* XGBoost
* MLflow
* FastAPI
* Power BI
* Git & GitHub

---

## Dataset

The project uses a public credit card fraud dataset.

Due to its size, the dataset is **not included** in this repository.

Place the downloaded dataset inside:

```text
data/raw/
```

---

## Roadmap

* [ ] Project initialization
* [ ] Data ingestion
* [ ] Bronze layer
* [ ] Silver layer
* [ ] Gold layer
* [ ] Feature engineering
* [ ] Machine Learning model
* [ ] Model evaluation
* [ ] FastAPI service
* [ ] AI Agent
* [ ] Power BI dashboard
* [ ] Documentation

---

## Author

**Tomás Pérez Ede**
