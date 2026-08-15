from typing import Literal

from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class InvestigationResult(BaseModel):
    prediction: Literal["FRAUD", "NORMAL"]
    fraud_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: Literal["STANDARD_PROCESSING", "MANUAL_REVIEW"]
    key_features: list[FeatureImportance]
    explanation: str