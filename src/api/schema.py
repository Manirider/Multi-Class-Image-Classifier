from pydantic import BaseModel, Field
from typing import Optional
class PredictionRequest(BaseModel):
    pass
class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "cat",
                "confidence": 0.95
            }
        }
class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    class Config:
        json_schema_extra = {"example": {"status": "ok"}}
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
