# Fraud Detection AI

End-to-end fraud detection platform built with **PySpark, scikit-learn, FastAPI, Docker, Google Cloud Run and GitHub Actions CI/CD**.

The project covers the complete lifecycle of a Machine Learning application: data ingestion and preparation, feature selection, model training, evaluation, batch prediction, model serving, containerization and cloud deployment. It also includes an **AI Agent layer** that adds investigation context and natural-language explanations on top of the deterministic fraud model.

---

## Project Overview

The project uses a public credit card fraud dataset containing **284,807 transactions** and addresses the highly imbalanced nature of fraud detection.

The main objectives are:

- Ingest and process transaction data using PySpark.
- Implement a Medallion-style data architecture.
- Perform feature selection and feature importance analysis.
- Train a Random Forest fraud detection model.
- Optimize the classification threshold using a validation dataset.
- Evaluate the model on previously unseen test data.
- Generate batch fraud predictions.
- Compare PySpark ML and scikit-learn implementations.
- Expose the model through a REST API using FastAPI.
- Run a Fraud Detection Agent on top of the model to produce structured, explained investigation results.
- Containerize the application with Docker.
- Deploy the API to Google Cloud Run.
- Store container images in Google Artifact Registry.
- Manage secrets (OpenAI API key) using Google Secret Manager.
- Implement automated testing and CI/CD using GitHub Actions.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │   Credit Card Data   │
                         │     creditcard.csv   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                               ┌───────────┐
                               │    Raw    │
                               └─────┬─────┘
                                     │
                                   PySpark
                                     │
                                     ▼
                               ┌───────────┐
                               │  Bronze   │
                               └─────┬─────┘
                                     │
                                   PySpark
                                     │
                                     ▼
                               ┌───────────┐
                               │  Silver   │
                               └─────┬─────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         ▼                       ▼
                  PySpark ML              scikit-learn
                  Random Forest            Random Forest
                         │                       │
                         └───────────┬───────────┘
                                     │
                              Model Evaluation
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         ▼                       ▼
                 Batch Predictions         FastAPI
                         │                       │
                         ▼                       ▼
                      Gold              Fraud Prediction
                                                 │
                                                 ▼
                                           AI Agent
                                                 │
                                      ┌──────────┴──────────┐
                                      │                     │
                                      ▼                     ▼
                                  Local Tools          OpenAI API
                                      │
                                      ▼
                              Structured Investigation
                                      │
                                      ▼
                                    Docker
                                      │
                                      ▼
                              Artifact Registry
                                      │
                                      ▼
                                  Cloud Run
```

---

## Data Pipeline

The data pipeline follows a Medallion-style architecture:

`Raw → Bronze → Silver → Machine Learning → Gold`

### Raw

The original dataset is stored locally as:

```
data/raw/creditcard.csv
```

The original data is kept unchanged at this stage.

### Bronze

The Raw dataset is ingested using PySpark and stored in the Bronze layer:

```
data/bronze/creditcard/
```

The Bronze layer preserves the original transaction structure:

- Time
- V1 – V28
- Amount
- Class

### Silver

The Bronze data is cleaned and prepared for Machine Learning:

```
data/silver/transactions_clean/
```

The Silver layer contains:

- Time
- V1 – V28
- Amount
- Class

Invalid rows are removed and the required columns are converted to appropriate numeric types.

Amount is retained in Silver for experimentation, although it is not part of the selected Top 10 feature model.

### Gold

The Gold layer contains the fraud predictions generated by the trained model:

```
data/gold/fraud_predictions/
```

The prediction output contains:

- fraud_probability
- final_prediction
- prediction_label

Because Spark writes distributed output, the Gold dataset may contain multiple `part-*` files.

---

## Machine Learning

The main Machine Learning algorithm is a Random Forest classifier.

The selected Top 10 features are:

| Rank | Feature |
|------|---------|
| 1 | V12 |
| 2 | V17 |
| 3 | V16 |
| 4 | V10 |
| 5 | V7 |
| 6 | V14 |
| 7 | V4 |
| 8 | V9 |
| 9 | V11 |
| 10 | V3 |

The Spark ML model is stored under:

```
models/fraud_random_forest_top10/
```

A scikit-learn version is also persisted as:

```
models/fraud_random_forest_top10_sklearn.joblib
```

### Feature Importance

The Random Forest model identified the following feature importance ranking:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | V12 | 0.252261 |
| 2 | V17 | 0.186887 |
| 3 | V16 | 0.115661 |
| 4 | V10 | 0.108945 |
| 5 | V7 | 0.087023 |
| 6 | V14 | 0.079335 |
| 7 | V4 | 0.068400 |
| 8 | V9 | 0.063141 |
| 9 | V11 | 0.019964 |
| 10 | V3 | 0.018384 |

The model is primarily driven by V12, V17, V16 and V10.

### Feature Selection Experiment

An experiment was performed to determine whether adding Amount improved the selected Top 10 model.

Two configurations were compared:

- Top 10 features
- Top 10 features + Amount

Results:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|----|--------|---------|
| Top 10 | 0.8770 | 0.7825 | 0.8271 | 0.7779 | 0.9723 |
| Top 10 + Amount | 0.8699 | 0.7744 | 0.8194 | 0.8126 | 0.9725 |

Adding Amount improved PR-AUC and slightly improved ROC-AUC, but reduced Precision, Recall and F1 at the selected operating point.

Because the primary optimization criterion was the F1 score of the FRAUD class, the Top 10 model was retained.

Amount remains available in the Silver layer for experimentation.

---

## Train / Validation / Test Strategy

Because fraud detection is highly imbalanced, the evaluation process uses a stratified Train / Validation / Test methodology.

The data is divided approximately as follows:

- 80% Train
- 10% Validation
- 10% Test

The split preserves the class distribution between:

- Class = 0 → Normal
- Class = 1 → Fraud

The workflow is:

```text
                         Silver
                           │
                           ▼
                    Stratified Split
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
           Train       Validation       Test
            80%            10%           10%
             │              │
             ▼              ▼
        Random Forest   Threshold
                       Optimization
                           │
                           ▼
                       Best Threshold
                           │
                           ▼
                     Final Test
                     Evaluation
```

The Test dataset is not used during threshold selection.

This prevents the Test set from influencing the final operating threshold.

---

## Threshold Optimization

The classification threshold is optimized using the F1 score of the FRAUD class on the Validation dataset.

The selected threshold for the deployed scikit-learn implementation is:

**0.35**

Classification logic:

```text
fraud_probability >= 0.35
        ↓
      FRAUD


fraud_probability < 0.35
        ↓
      NORMAL
```

The threshold is determined during validation and remains fixed during inference.

---

## Final Model Comparison

Two implementations of the Random Forest model were evaluated:

- PySpark ML
- scikit-learn

The comparison was performed using the same feature set and evaluation methodology.

### Results

| Metric | PySpark ML | scikit-learn |
|--------|-----------|---------------|
| Threshold | 0.30 | 0.35 |
| Precision | 0.8298 | 0.8478 |
| Recall | 0.7800 | 0.7800 |
| F1 | 0.8041 | 0.8125 |
| PR-AUC | 0.6848 | 0.7694 |
| ROC-AUC | 0.9350 | 0.9295 |

The two implementations achieved:

**99.9895% prediction agreement**

This showed that the scikit-learn implementation produced almost identical classification decisions while being significantly lighter for online inference.

---

## Model Serving Optimization

The original API used a PySpark-based inference environment.

The resulting Docker image was approximately **1.57 GB**.

A second implementation was created using scikit-learn and the persisted `.joblib` model.

The resulting image was approximately **689 MB**.

This represents a reduction of approximately **56%**.

The comparison demonstrated that Spark is useful for distributed data processing and Machine Learning workflows, while scikit-learn provides a more lightweight option for serving individual predictions through an API.

---

## Batch Prediction

The Spark batch prediction pipeline loads the trained model and generates predictions from the Silver dataset.

```text
Silver
  │
  ▼
Random Forest
  │
  ▼
fraud_probability
  │
  ▼
Classification Threshold
  │
  ▼
final_prediction
  │
  ▼
prediction_label
  │
  ▼
Gold
```

The prediction output contains:

- fraud_probability
- final_prediction
- prediction_label

Example:

| fraud_probability | final_prediction | prediction_label |
|--------------------|-------------------|--------------------|
| 0.00689 | 0 | NORMAL |
| 0.86773 | 1 | FRAUD |

---

## FastAPI

The scikit-learn model is exposed through a REST API built with FastAPI.

Main endpoints:

- `GET /health`
- `POST /predict`
- `POST /investigate`

### `/health`

Used to verify that the API is running.

Example response:

```json
{
  "status": "ok"
}
```

### `/predict`

Receives the Top 10 model features as JSON and returns the fraud probability and final classification.

Example request:

```json
{
  "V12": 0.0,
  "V17": 0.0,
  "V16": 0.0,
  "V7": 0.0,
  "V10": 0.0,
  "V14": 0.0,
  "V11": 0.0,
  "V4": 0.0,
  "V9": 0.0,
  "V3": 0.0
}
```

Example response:

```json
{
  "fraud_probability": 0.00027097637208884463,
  "final_prediction": 0,
  "prediction_label": "NORMAL"
}
```

### `/investigate`

Runs the Fraud Detection Agent for a transaction.

The endpoint combines the scikit-learn prediction, configured threshold, risk classification, key model features and an OpenAI-generated explanation into a structured response.

Example response:

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.5641676689595301,
  "threshold": 0.35,
  "risk_level": "HIGH",
  "recommendation": "MANUAL_REVIEW"
}
```

The API also exposes interactive Swagger documentation through:

```
/docs
```

---

## Fraud Detection Agent

The project includes an AI Agent layer on top of the scikit-learn fraud prediction API.

The Agent combines the deterministic output of the Random Forest model with investigation logic and an OpenAI-powered explanation layer.

### Agent responsibilities

The Agent receives the same Top 10 transaction features used by the deployed scikit-learn model and produces a structured investigation result containing:

- `prediction`
- `fraud_probability`
- `threshold`
- `risk_level`
- `recommendation`
- `key_features`
- `explanation`

The fraud decision remains deterministic and is based on the configured Random Forest threshold:

```text
fraud_probability >= 0.35
        ↓
      FRAUD
        ↓
      HIGH
        ↓
MANUAL_REVIEW
```

The OpenAI component is used to generate the investigation explanation. The core fraud prediction is not delegated to the LLM.

### Agent tools

The Agent uses application-side tools to access the fraud model and supporting information. This keeps model inference separate from the natural-language investigation layer.

### OpenAI client and testability

The OpenAI client is created through a dedicated `get_openai_client()` function rather than as a module-level client.

This allows the external dependency to be replaced during unit tests:

```text
Production
investigate()
    ↓
get_openai_client()
    ↓
OpenAI API

Tests
investigate()
    ↓
get_openai_client()
    ↓
MockOpenAIClient
```

As a result, Agent tests do not require an OpenAI API key, network access or real API calls.

The real OpenAI integration is validated separately through Docker and Cloud Run end-to-end testing.

### Structured Agent tests

The project includes dedicated tests for:

- Normal transactions
- Fraudulent transactions
- Structured Agent output

The OpenAI response is mocked while the real scikit-learn fraud model remains active. This validates the Agent logic without coupling the test suite to an external service.

---

## Docker

The project contains separate Docker configurations for the Spark and scikit-learn implementations.

- `Dockerfile` → PySpark-based API
- `Dockerfile.sklearn` → lightweight scikit-learn API and Agent

The scikit-learn image contains:

- Python
- FastAPI
- scikit-learn
- joblib
- OpenAI client
- Fraud Detection Agent
- Persisted Random Forest model

The resulting scikit-learn image is approximately **689 MB**.

The OpenAI API key is not included in the Docker image. It is injected at runtime.

Docker images are stored in Google Artifact Registry.

---

## Google Cloud Deployment

The scikit-learn API and Fraud Detection Agent were deployed and validated using Google Cloud Run.

Deployment architecture:

```text
GitHub Actions
      │
      ▼
Docker Build
      │
      ▼
Artifact Registry
      │
      ▼
Google Cloud Run
      │
      ├── FastAPI
      ├── scikit-learn Model
      └── Fraud Detection Agent
               │
               ▼
          Secret Manager
               │
               ▼
           OpenAI API
```

The deployed API was tested successfully through:

- `/health`
- `/predict`
- `/investigate`
- Swagger `/docs`

The `/investigate` endpoint was successfully executed after deployment and returned the expected structured fraud investigation result.

The Cloud Run service was removed after validation to avoid leaving an unnecessary running service.

The Docker image remains available in Artifact Registry for future deployments.

### Secret management

The OpenAI API key is stored in Google Secret Manager and exposed to Cloud Run as the `OPENAI_API_KEY` environment variable.

The secret is never stored in:

- Git
- GitHub Actions workflow files
- Dockerfiles
- Docker images

Cloud Run uses a runtime service account with the `roles/secretmanager.secretAccessor` permission for the OpenAI secret.

### Deployment strategy

The scikit-learn Cloud Run deployment is intentionally manual.

The deployment workflow is triggered with GitHub Actions `workflow_dispatch`, allowing a deployment to be started from the GitHub Actions interface instead of deploying on every push.

The deployment performs:

```text
Run workflow
     │
     ▼
Docker Build
     │
     ▼
Artifact Registry :sklearn
     │
     ▼
Cloud Run
     │
     ▼
Secret Manager
     │
     ▼
OpenAI
```

---

## CI/CD

The project uses GitHub Actions for Continuous Integration and Continuous Deployment.

The repository contains separate workflows for:

- `tests.yml`
- `deploy.yml`
- `deploy-sklearn.yml`

### Continuous Integration

The CI workflow executes the automated test suite using:

```bash
python -m pytest -v
```

The test suite validates the project automatically on pushes and pull requests to `main`.

The current test suite contains:

- API health validation
- Normal transaction prediction
- Fraud transaction prediction
- Invalid input validation
- Missing feature validation
- Null feature validation
- Normal Agent investigation
- Fraud Agent investigation
- Structured Agent output

The Agent tests mock the OpenAI client, so CI does not require an OpenAI API key.

### Continuous Deployment

The scikit-learn deployment is manually triggered from GitHub Actions.

The workflow performs:

```text
GitHub Actions
      │
      ▼
Docker Build
      │
      ▼
Artifact Registry :sklearn
      │
      ▼
Cloud Run
      │
      ▼
Secret Manager → OPENAI_API_KEY
```

This separates automatic validation from deployment:

```text
git push
   │
   ▼
Automatic tests
   │
   └── pytest

Manual GitHub Actions trigger
   │
   ▼
Docker build + push
   │
   ▼
Cloud Run deployment
```

The scikit-learn deployment was successfully executed through GitHub Actions and the resulting Cloud Run service was validated end-to-end.

---

## Git and Version Control

The project uses Git and GitHub for version control.

The repository includes:

- Source code
- Configuration files
- Docker configuration
- CI/CD workflows
- Model artifacts required for deployment
- Tests
- Documentation

A specific `.gitignore` strategy is used to prevent unwanted model artifacts from being committed while allowing the model required by the deployment pipeline to be versioned.

---

## Automated Testing

Automated tests are executed using pytest.

The CI pipeline validates the project automatically through GitHub Actions.

The current suite contains 9 tests covering:

- FastAPI health endpoint
- Normal fraud prediction
- Fraud prediction
- Invalid transaction input
- Missing features
- Null features
- Normal Agent investigation
- Fraud Agent investigation
- Structured Agent output

The OpenAI integration is mocked in Agent unit tests using `MockOpenAIClient`.

This keeps the tests independent from:

- OpenAI availability
- Network access
- API credentials
- External API costs

The real OpenAI integration is tested separately during Docker and Cloud Run end-to-end validation.

---

## Project Structure

```text
fraud-detection-ai/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   ├── bronze/
│   │   └── creditcard/
│   ├── silver/
│   │   └── transactions_clean/
│   └── gold/
│       └── fraud_predictions/
│
├── models/
│   ├── fraud_random_forest_top10/
│   └── fraud_random_forest_top10_sklearn.joblib
│
├── src/
│   ├── agent/
│   │   ├── fraud_agent.py
│   │   └── tools.py
│   ├── api/
│   │   ├── main.py
│   │   └── main_sklearn.py
│   └── ml/
│       ├── predictor_sklearn.py
│       └── ...
│
├── tests/
│   ├── test_api.py
│   └── test_agent.py
│
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── deploy.yml
│       └── deploy-sklearn.yml
│
├── Dockerfile
├── Dockerfile.sklearn
├── requirements.txt
├── requirements-api-sklearn.txt
├── README.md
└── .gitignore
```

---

## Technology Stack

### Data Engineering
- Python
- PySpark
- Apache Spark
- SQL
- Medallion Architecture

### Machine Learning
- scikit-learn
- PySpark ML
- Random Forest
- Feature importance
- Threshold optimization
- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

### API & AI Agent
- FastAPI
- REST
- Swagger / OpenAPI
- OpenAI API
- AI Agent architecture
- Structured Agent output
- Dependency injection / mocking

### DevOps & Cloud
- Docker
- Git
- GitHub
- GitHub Actions
- Google Artifact Registry
- Google Cloud Run
- Google Secret Manager

---

## Dataset

The project uses a public credit card fraud detection dataset.

The dataset is not included in the repository because of its size.

Expected location:

```
data/raw/creditcard.csv
```

The dataset contains:

- Time
- V1 – V28
- Amount
- Class

The target variable is:

- 0 = Normal
- 1 = Fraud

---

## Project Status

The project is considered complete as an end-to-end portfolio project.

Implemented components:

- [x] Project initialization
- [x] Raw data ingestion
- [x] Bronze layer
- [x] Silver layer
- [x] Gold layer
- [x] Medallion-style architecture
- [x] Feature selection
- [x] Feature importance analysis
- [x] Random Forest model
- [x] Model persistence
- [x] Model loading for inference
- [x] Fraud probability calculation
- [x] Amount feature experiment
- [x] Stratified Train / Validation / Test split
- [x] Threshold optimization
- [x] Final Test evaluation
- [x] Confusion matrix
- [x] Precision / Recall / F1 evaluation
- [x] PR-AUC / ROC-AUC evaluation
- [x] Batch fraud prediction
- [x] Spark ML implementation
- [x] scikit-learn implementation
- [x] Spark vs scikit-learn comparison
- [x] Prediction agreement analysis
- [x] FastAPI service
- [x] Swagger documentation
- [x] Fraud Detection AI Agent
- [x] OpenAI integration
- [x] Mocked Agent testing
- [x] Automated testing
- [x] Docker containerization
- [x] Artifact Registry
- [x] Google Secret Manager
- [x] Google Cloud Run deployment
- [x] Manual Cloud Run deployment through GitHub Actions
- [x] GitHub Actions CI/CD
- [x] End-to-end cloud deployment validation

---

## Deliberately Out of Scope

The following features were considered but were not implemented because they are not required for the current production-oriented portfolio version:

- Power BI dashboard
- Alternative Machine Learning algorithms
- Formal k-fold cross-validation
- Production monitoring
- Model monitoring

These are potential extensions rather than incomplete components of the current end-to-end pipeline.

---

## Key Engineering Decisions

### Why Random Forest?

Random Forest provides a strong baseline for tabular classification and works well with the numerical feature representation of this dataset.

### Why F1 instead of Accuracy?

Fraud is a highly imbalanced classification problem. Accuracy can therefore be misleading.

The project prioritizes:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

with particular emphasis on the FRAUD class.

### Why optimize the threshold?

The default threshold of 0.50 is not necessarily appropriate for fraud detection.

The project explicitly evaluates different thresholds and selects the operating point using the validation dataset.

### Why scikit-learn for the API?

The Spark and scikit-learn implementations achieved 99.9895% prediction agreement while the scikit-learn container was substantially smaller:

| Implementation | Docker image |
|-----------------|---------------|
| Spark | ~1.57 GB |
| scikit-learn | ~689 MB |

Therefore, scikit-learn was selected as the preferred implementation for lightweight online inference.

---

## Key Agent Engineering Decisions

### Why keep the fraud decision outside the LLM?

The Random Forest remains responsible for the numerical fraud probability and final classification.

The Agent adds investigation context and an explanation layer around the deterministic model output.

This avoids delegating the core fraud classification to a generative model.

### Why mock OpenAI in unit tests?

OpenAI is an external dependency. Unit tests should remain deterministic and independent of network availability and external credentials.

The Agent therefore obtains the OpenAI client through `get_openai_client()`, which can be replaced by a `MockOpenAIClient` during tests.

The real OpenAI integration is validated separately as part of the Docker and Cloud Run end-to-end tests.

### Why use Secret Manager?

The OpenAI API key is required at runtime but should never be stored in source control or inside a container image.

Google Secret Manager provides the runtime secret while keeping the Docker image reusable and free of credentials.

---

## Key Results

| Metric | Result |
|--------|--------|
| Dataset size | 284,807 transactions |
| Selected features | 10 |
| Spark F1 | 0.8041 |
| scikit-learn F1 | 0.8125 |
| scikit-learn PR-AUC | 0.7694 |
| scikit-learn ROC-AUC | 0.9295 |
| Prediction agreement | 99.9895% |
| Spark Docker image | ~1.57 GB |
| scikit-learn Docker image | ~689 MB |
| Docker image reduction | ~56% |

---

## End-to-End Workflow

```text
Credit Card Dataset
        │
        ▼
      PySpark
        │
        ▼
   Bronze / Silver
        │
        ▼
  Feature Selection
        │
        ▼
   Random Forest
        │
   ┌────┴───────────────┐
   ▼                    ▼
Spark ML           scikit-learn
   │                    │
   └────────┬───────────┘
            ▼
      Model Evaluation
            │
            ▼
    Threshold Optimization
            │
            ▼
      Batch Predictions
            │
            ▼
           Gold
            │
            ▼
         FastAPI
            │
            ▼
      Fraud Prediction
            │
            ▼
        AI Agent
        │      │
        │      └──────────► OpenAI
        │
        ▼
   Structured Result
            │
            ▼
          Docker
            │
            ▼
     Artifact Registry
            │
            ▼
        Cloud Run
            │
            ▼
 REST / Swagger / Investigate
```

---

## Author

Tomás Pérez Ede