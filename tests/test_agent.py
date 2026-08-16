import json

from src.agent.fraud_agent import investigate


class MockOpenAIClient:

    class Responses:

        @staticmethod
        def create(*args, **kwargs):
            return MockResponse()

    responses = Responses()


NORMAL_TRANSACTION = {
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


FRAUD_TRANSACTION = {
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


# ==========================
# Mock OpenAI response
# ==========================

class MockResponse:

    output_text = json.dumps({
        "explanation": (
            "The transaction was classified as fraudulent because "
            "the fraud probability exceeds the configured threshold. "
            "Manual review is recommended."
        )
    })


# ==========================
# Test NORMAL investigation
# ==========================

def test_agent_normal(monkeypatch):

    monkeypatch.setattr(
        "src.agent.fraud_agent.get_openai_client",
        lambda: MockOpenAIClient()
    )

    result = investigate(NORMAL_TRANSACTION)

    assert result.prediction == "NORMAL"
    assert result.fraud_probability < result.threshold
    assert result.risk_level == "LOW"
    assert result.recommendation == "STANDARD_PROCESSING"
    assert len(result.key_features) == 6
    assert result.explanation


# ==========================
# Test FRAUD investigation
# ==========================

def test_agent_fraud(monkeypatch):

    monkeypatch.setattr(
        "src.agent.fraud_agent.get_openai_client",
        lambda: MockOpenAIClient()
    )

    result = investigate(FRAUD_TRANSACTION)

    assert result.prediction == "FRAUD"
    assert result.fraud_probability >= result.threshold
    assert result.risk_level == "HIGH"
    assert result.recommendation == "MANUAL_REVIEW"
    assert len(result.key_features) == 6
    assert result.explanation


# ==========================
# Test structured output
# ==========================

def test_agent_returns_structured_result(monkeypatch):

    monkeypatch.setattr(
        "src.agent.fraud_agent.get_openai_client",
        lambda: MockOpenAIClient()
    )

    result = investigate(FRAUD_TRANSACTION)

    assert isinstance(result.prediction, str)
    assert isinstance(result.fraud_probability, float)
    assert isinstance(result.threshold, float)
    assert isinstance(result.risk_level, str)
    assert isinstance(result.recommendation, str)
    assert isinstance(result.key_features, list)
    assert isinstance(result.explanation, str)