"""
Customer Intelligence AI Service - Main FastAPI Application
Production-ready AI service with 90.5% churn accuracy
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from datetime import datetime
from contextlib import asynccontextmanager

# Import your AI service components
from app.services.cache_service import CacheService
from app.services.ai_engine import AIEngine
from app.models.schemas import (
    CustomerData, 
    SentimentRequest,
    CustomerAnalysisResponse,
    SentimentResponse,
    ChurnPredictionResponse,
    HealthResponse
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global services
cache_service = None
ai_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    global cache_service, ai_engine
    
    # Startup
    logger.info("🚀 Starting Customer Intelligence AI Service...")
    
    try:
        # Initialize services
        cache_service = CacheService()
        ai_engine = AIEngine(cache_service)
        
        # Initialize AI models
        logger.info("🤖 Initializing AI models...")
        await ai_engine.initialize_models()
        
        if ai_engine.is_ready():
            logger.info("✅ Customer Intelligence AI Service ready for production!")
        else:
            logger.warning("⚠️ AI Service started with limited functionality")
            
        yield
        
    except Exception as e:
        logger.error("❌ Failed to initialize AI Service", error=str(e))
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down Customer Intelligence AI Service...")
    if ai_engine:
        await ai_engine.cleanup()

# Create FastAPI application
app = FastAPI(
    title="Customer Intelligence AI Service",
    description="AI-powered customer analysis, sentiment analysis, and churn prediction service with 90.5% accuracy",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    try:
        status = await ai_engine.get_model_status() if ai_engine else {"initialized": False}
        
        return HealthResponse(
            status="healthy" if status.get("ready", False) else "degraded",
            service="customer-intelligence-ai",
            version="2.0.0",
            models_loaded=status.get("ready", False),
            timestamp=datetime.utcnow().isoformat(),
            models_status=status.get("models", {})
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            service="customer-intelligence-ai",
            version="2.0.0",
            models_loaded=False,
            timestamp=datetime.utcnow().isoformat()
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Customer Intelligence AI Service",
        "version": "2.0.0",
        "description": "AI-powered customer analysis with 90.5% churn prediction accuracy",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "customer_analysis": "/analyze-customer",
            "sentiment_analysis": "/analyze-sentiment",
            "churn_prediction": "/predict-churn"
        }
    }

# Customer Analysis endpoint
@app.post("/analyze-customer", response_model=CustomerAnalysisResponse)
async def analyze_customer(customer_data: CustomerData):
    """Analyze customer with AI-powered segmentation and behavioral scoring"""
    try:
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(status_code=503, detail="AI service not available")
        
        logger.info("🔍 Processing customer analysis", customer_id=customer_data.customerId)
        
        analysis = await ai_engine.analyze_customer(customer_data)
        
        logger.info("✅ Customer analysis completed", 
                   customer_id=customer_data.customerId,
                   segment=analysis.aiSegment,
                   score=analysis.behavioralScore)
        
        return analysis
        
    except Exception as e:
        logger.error("❌ Customer analysis failed", 
                    customer_id=customer_data.customerId,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Sentiment Analysis endpoint
@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze text sentiment with transformer models"""
    try:
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(status_code=503, detail="AI service not available")
        
        logger.info("💭 Processing sentiment analysis", 
                   text_length=len(request.text),
                   context=request.context.value)
        
        sentiment = await ai_engine.analyze_sentiment(request.text, request.context)
        
        logger.info("✅ Sentiment analysis completed",
                   sentiment=sentiment.sentiment.value,
                   confidence=sentiment.confidence)
        
        return sentiment
        
    except Exception as e:
        logger.error("❌ Sentiment analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

# Churn Prediction endpoint
@app.post("/predict-churn", response_model=ChurnPredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """Predict customer churn with ML models"""
    try:
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(status_code=503, detail="AI service not available")
        
        logger.info("🔮 Processing churn prediction", customer_id=customer_data.customerId)
        
        prediction = await ai_engine.predict_churn(customer_data)
        
        logger.info("✅ Churn prediction completed",
                   customer_id=customer_data.customerId,
                   probability=prediction.churnProbability,
                   risk=prediction.riskLevel.value)
        
        return prediction
        
    except Exception as e:
        logger.error("❌ Churn prediction failed", 
                    customer_id=customer_data.customerId,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Churn prediction failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "Check /docs for available endpoints"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "AI service encountered an error"}
    )

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
