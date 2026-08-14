from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# ==========================
# Test /health
# ==========================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


# ==========================
# Test normal transaction
# ==========================

def test_normal_transaction():

    transaction = {
        "V12": 0,
        "V17": 0,
        "V16": 0,
        "V7": 0,
        "V10": 0,
        "V14": 0,
        "V11": 0,
        "V4": 0,
        "V9": 0,
        "V3": 0
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "final_prediction" in result
    assert "prediction_label" in result

    assert result["final_prediction"] == 0
    assert result["prediction_label"] == "NORMAL"


# ==========================
# Test fraud transaction
# ==========================

def test_fraud_transaction():

    transaction = {
        "V12": -2.89990738849473,
        "V17": -2.83005567450437,
        "V16": -1.14074717980657,
        "V7": -2.53738730624579,
        "V10": -2.77227214465915,
        "V14": -4.28925378244217,
        "V11": 3.20203320709635,
        "V4": 3.9979055875468,
        "V9": -2.77008927719433,
        "V3": -1.60985073229769
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "final_prediction" in result
    assert "prediction_label" in result

    assert result["final_prediction"] == 1
    assert result["prediction_label"] == "FRAUD"

    assert result["fraud_probability"] >= 0.20


# ==========================
# Test invalid transaction
# ==========================

def test_invalid_transaction():

    transaction = {
        "V12": "invalid",
        "V17": 0,
        "V16": 0,
        "V7": 0,
        "V10": 0,
        "V14": 0,
        "V11": 0,
        "V4": 0,
        "V9": 0,
        "V3": 0
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422

    # ==========================
# Test missing feature
# ==========================

def test_missing_feature():

    transaction = {
        "V17": 0,
        "V16": 0,
        "V7": 0,
        "V10": 0,
        "V14": 0,
        "V11": 0,
        "V4": 0,
        "V9": 0,
        "V3": 0
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422


# ==========================
# Test null feature
# ==========================

def test_null_feature():

    transaction = {
        "V12": None,
        "V17": 0,
        "V16": 0,
        "V7": 0,
        "V10": 0,
        "V14": 0,
        "V11": 0,
        "V4": 0,
        "V9": 0,
        "V3": 0
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 422