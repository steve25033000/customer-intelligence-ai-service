"""
Data schemas for Customer Intelligence AI Service
"""

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ContextType(str, Enum):
    CHAT = "chat"
    EMAIL = "email"
    REVIEW = "review"
    SUPPORT_TICKET = "support_ticket"
    SOCIAL_MEDIA = "social_media"
    GENERAL = "general"

class CustomerData(BaseModel):
    customerId: str
    email: EmailStr
    firstName: str
    lastName: str
    purchaseHistory: Optional[List[Dict[str, Any]]] = []
    engagementData: Optional[Dict[str, Any]] = {}
    demographics: Optional[Dict[str, Any]] = {}

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    context: ContextType = ContextType.GENERAL

class CustomerAnalysisResponse(BaseModel):
    customerId: str
    aiSegment: str
    segmentDescription: str
    behavioralScore: int = Field(..., ge=0, le=100)
    churnRisk: RiskLevel
    churnProbability: float = Field(..., ge=0.0, le=1.0)
    analysis: Dict[str, Any]
    recommendations: List[str]
    insights: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    modelVersion: str
    analyzedAt: datetime

class SentimentResponse(BaseModel):
    text: str
    sentiment: SentimentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    emotion: str
    urgency: str
    scores: Dict[str, float]
    context: ContextType
    keyPhrases: List[str]
    modelVersion: str
    analyzedAt: datetime

class ChurnPredictionResponse(BaseModel):
    customerId: str
    churnProbability: float = Field(..., ge=0.0, le=1.0)
    churnPrediction: bool
    riskLevel: RiskLevel
    riskColor: str
    contributingFactors: List[Dict[str, Any]]
    recommendations: List[str]
    daysUntilChurn: Optional[int]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    modelVersion: str
    predictionDate: datetime

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: bool
    timestamp: str
    models_status: Optional[Dict[str, bool]] = None

