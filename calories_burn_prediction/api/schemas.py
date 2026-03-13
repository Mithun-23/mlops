from pydantic import BaseModel, Field

class CaloriesPredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age in years")
    gender: str = Field(..., pattern="^(male|female)$", description="Gender: male or female")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=30, le=200, description="Weight in kg")
    duration: int = Field(..., ge=1, le=300, description="Duration of exercise in minutes")
    heart_rate: int = Field(..., ge=40, le=220, description="Average heart rate in bpm")
    body_temp: float = Field(..., ge=35.0, le=42.0, description="Body temperature in Celsius")

class CaloriesPredictionResponse(BaseModel):
    predicted_calories: float
