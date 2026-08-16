import json
from src.agent.schemas import InvestigationResult

from openai import OpenAI

from src.agent.tools import (
    predict_fraud,
    get_feature_importance,
    get_model_configuration,
)


# ==========================
# OpenAI Client
# ==========================

def get_openai_client():
    return OpenAI()


# ==========================
# Agent Instructions
# ==========================

SYSTEM_PROMPT = """
You are a Fraud Investigation Agent.

Your job is to investigate credit card transactions using the
available fraud detection tools.

You must rely on the tools for factual information about the
machine learning model and the transaction.

Do not invent model results, probabilities, features, thresholds,
or metrics.

When investigating a transaction:

1. Use predict_fraud() to obtain the actual model prediction.
2. Use get_feature_importance() when an explanation of the
   important model features is useful.
3. Use get_model_configuration() when information about the
   model or classification threshold is required.

Explain the result clearly to a fraud analyst.

Always distinguish between:
- the prediction produced by the Machine Learning model
- your natural-language explanation of that prediction.

If the model predicts FRAUD, recommend MANUAL_REVIEW.
If the model predicts NORMAL, explain that the model did not
identify the transaction as fraudulent at the configured threshold.
"""


# ==========================
# Tool Definitions
# ==========================

TOOLS = [
    {
        "type": "function",
        "name": "predict_fraud",
        "description": (
            "Predict whether a credit card transaction is fraudulent "
            "using the deployed scikit-learn Random Forest model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "V12": {"type": "number"},
                "V17": {"type": "number"},
                "V16": {"type": "number"},
                "V7": {"type": "number"},
                "V10": {"type": "number"},
                "V14": {"type": "number"},
                "V11": {"type": "number"},
                "V4": {"type": "number"},
                "V9": {"type": "number"},
                "V3": {"type": "number"},
            },
            "required": [
                "V12",
                "V17",
                "V16",
                "V7",
                "V10",
                "V14",
                "V11",
                "V4",
                "V9",
                "V3",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_feature_importance",
        "description": (
            "Return the feature importance values from the "
            "Random Forest fraud detection model."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_model_configuration",
        "description": (
            "Return the fraud detection model type, classification "
            "threshold and features used by the model."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# ==========================
# Tool Execution
# ==========================

def execute_tool(name, arguments):
    """
    Execute a tool requested by the model.
    """

    if name == "predict_fraud":
        return predict_fraud(arguments)

    if name == "get_feature_importance":
        return get_feature_importance()

    if name == "get_model_configuration":
        return get_model_configuration()

    raise ValueError(f"Unknown tool: {name}")


# ==========================
# Fraud Investigation Agent
# ==========================

def investigate(transaction: dict) -> InvestigationResult:
    """
    Run the Fraud Investigation Agent and return
    a structured investigation result.
    """

    # Get the real ML prediction directly from the model.
    prediction = predict_fraud(transaction)

    # Get real model information.
    feature_importance = get_feature_importance()
    model_configuration = get_model_configuration()

    fraud_probability = prediction["fraud_probability"]
    prediction_label = prediction["prediction_label"]
    threshold = model_configuration["threshold"]

    # Select the most important features for the explanation.
    top_features = feature_importance[:6]

    risk_level = (
        "HIGH"
        if prediction_label == "FRAUD"
        else "LOW"
    )

    recommendation = (
        "MANUAL_REVIEW"
        if prediction_label == "FRAUD"
        else "STANDARD_PROCESSING"
    )

    context = {
        "prediction": prediction_label,
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "model": model_configuration["model"],
        "feature_importance": top_features,
        "recommendation": recommendation,
    }

    user_message = f"""
Investigate this fraud detection result for a fraud analyst.

The following information comes directly from the Machine Learning
model and must NOT be changed:

{json.dumps(context, indent=2)}

Provide a concise analyst explanation.

Important:
- Do not invent model results.
- Do not change the prediction.
- Do not change the fraud probability.
- Do not change the threshold.
- Do not claim that global feature importance proves why this
  particular transaction was classified as fraud.
- Explain that feature importance represents global model behavior.
- If the prediction is FRAUD, explain why manual review is recommended.
- If the prediction is NORMAL, explain why standard processing is appropriate.
"""

    client = get_openai_client()
    response = client.responses.create(
        model="gpt-5.6",
        instructions=SYSTEM_PROMPT,
        input=user_message,
        text={
            "format": {
                "type": "json_schema",
                "name": "fraud_investigation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "explanation"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    explanation_data = json.loads(
        response.output_text
    )

    return InvestigationResult(
        prediction=prediction_label,
        fraud_probability=fraud_probability,
        threshold=threshold,
        risk_level=risk_level,
        recommendation=recommendation,
        key_features=top_features,
        explanation=explanation_data["explanation"],
    )