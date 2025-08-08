from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum

class ContextType(str, Enum):
    GENERAL = "general"
    EMAIL = "email"
    CHAT = "chat"
    REVIEW = "review"
    SUPPORT_TICKET = "support_ticket"
    SOCIAL_MEDIA = "social_media"

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium" 
    HIGH = "High"
    CRITICAL = "Critical"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

# Input Models
class CustomerData(BaseModel):
    customerId: str = Field(..., description="Unique customer identifier")
    email: str = Field(..., description="Customer email address")
    firstName: str = Field(..., description="Customer first name")
    lastName: str = Field(..., description="Customer last name")
    purchaseHistory: List[Dict[str, Any]] = Field(default=[], description="List of customer purchases")
    engagementData: Dict[str, Any] = Field(default={}, description="Customer engagement metrics")
    demographics: Optional[Dict[str, Any]] = Field(default={}, description="Customer demographic information")
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('purchaseHistory')
    def validate_purchases(cls, v):
        for purchase in v:
            if 'amount' in purchase and not isinstance(purchase['amount'], (int, float)):
                raise ValueError('Purchase amount must be numeric')
        return v

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    context: ContextType = Field(default=ContextType.GENERAL, description="Context of the text")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class BatchAnalysisRequest(BaseModel):
    customers: List[CustomerData] = Field(..., max_items=100, description="List of customers to analyze")

# Output Models
class CustomerSegment(BaseModel):
    name: str = Field(..., description="Segment name")
    description: str = Field(..., description="Segment description")
    characteristics: Dict[str, Any] = Field(..., description="Segment characteristics")
    color: str = Field(..., description="UI color for segment")

class CustomerAnalysisResponse(BaseModel):
    customerId: str
    aiSegment: str
    segmentDescription: str
    behavioralScore: int = Field(..., ge=0, le=100)
    churnRisk: RiskLevel
    churnProbability: float = Field(..., ge=0.0, le=1.0)
    
    analysis: Dict[str, Any] = Field(..., description="Detailed analysis data")
    recommendations: List[str] = Field(..., description="AI-generated recommendations")
    insights: List[str] = Field(default=[], description="Key insights about the customer")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence score")
    modelVersion: str = Field(..., description="AI model version used")
    analyzedAt: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SentimentResponse(BaseModel):
    text: str
    sentiment: SentimentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    emotion: str
    urgency: str
    
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    context: ContextType
    keyPhrases: List[str] = Field(default=[], description="Key phrases extracted")
    
    analyzedAt: datetime = Field(default_factory=datetime.utcnow)
    modelVersion: str = Field(..., description="Model version used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChurnPredictionResponse(BaseModel):
    customerId: str
    churnProbability: float = Field(..., ge=0.0, le=1.0)
    churnPrediction: bool
    riskLevel: RiskLevel
    riskColor: str
    
    contributingFactors: List[Dict[str, Any]] = Field(..., description="Factors contributing to churn risk")
    recommendations: List[str] = Field(..., description="Retention recommendations")
    daysUntilChurn: Optional[int] = Field(None, description="Estimated days until churn")
    
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    modelVersion: str = Field(..., description="Model version used")
    predictionDate: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class InsightsResponse(BaseModel):
    customerId: str
    insights: List[Dict[str, Any]] = Field(..., description="Generated insights with categories")
    summary: str = Field(..., description="Executive summary of insights")
    actionItems: List[str] = Field(..., description="Recommended action items")
    
    priority: str = Field(..., description="Overall priority level")
    generatedAt: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RecommendationsResponse(BaseModel):
    customerId: str
    recommendations: List[Dict[str, Any]] = Field(..., description="Detailed recommendations with priorities")
    strategicActions: List[str] = Field(..., description="High-level strategic actions")
    tacticalActions: List[str] = Field(..., description="Immediate tactical actions")
    
    expectedOutcome: str = Field(..., description="Expected outcome of recommendations")
    timeframe: str = Field(..., description="Recommended timeframe for implementation")
    
    generatedAt: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
