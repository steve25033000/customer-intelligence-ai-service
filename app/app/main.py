from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import json
import logging
import structlog
from datetime import datetime, timedelta
import asyncio
import os
from contextlib import asynccontextmanager

# Import our custom services (we'll create these next)
from app.services.ai_engine import AIEngine
from app.services.cache_service import CacheService
from app.models.schemas import (
    CustomerData, 
    SentimentRequest, 
    BatchAnalysisRequest,
    CustomerAnalysisResponse,
    SentimentResponse,
    ChurnPredictionResponse,
    InsightsResponse,
    RecommendationsResponse
)

# Configure structured logging
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

# Global AI engine instance
ai_engine = None
cache_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global ai_engine, cache_service
    
    logger.info("🚀 Starting Customer Intelligence AI Service...")
    
    try:
        # Initialize services
        cache_service = CacheService()
        ai_engine = AIEngine(cache_service)
        
        # Load AI models
        await ai_engine.initialize_models()
        
        logger.info("✅ AI Service startup complete")
        yield
        
    except Exception as e:
        logger.error("❌ Failed to start AI Service", error=str(e))
        raise
    finally:
        logger.info("🛑 Shutting down AI Service...")
        if ai_engine:
            await ai_engine.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Customer Intelligence AI Service",
    description="Advanced AI-powered customer analysis and insights",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure properly for production
)

# Health check endpoint
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Railway and monitoring"""
    global ai_engine
    
    health_data = {
        "status": "healthy",
        "service": "customer-intelligence-ai",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": ai_engine.is_ready() if ai_engine else False,
        "cache_connected": cache_service.is_connected() if cache_service else False,
        "endpoints": [
            "/analyze-customer",
            "/analyze-sentiment", 
            "/predict-churn",
            "/generate-insights",
            "/recommend-actions",
            "/customer-segments",
            "/batch-analyze"
        ]
    }
    
    logger.info("Health check requested", **health_data)
    return health_data

@app.get("/models/status", tags=["Health"])
async def model_status():
    """Get detailed model loading status"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI Engine not initialized")
    
    return await ai_engine.get_model_status()

# Customer Analysis endpoint
@app.post("/analyze-customer", response_model=CustomerAnalysisResponse, tags=["Customer Analysis"])
async def analyze_customer(
    customer_data: CustomerData,
    background_tasks: BackgroundTasks
):
    """Comprehensive AI-powered customer analysis"""
    try:
        logger.info("Starting customer analysis", customer_id=customer_data.customerId)
        
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503, 
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        # Check cache first
        cache_key = f"customer_analysis:{customer_data.customerId}"
        cached_result = await cache_service.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached analysis", customer_id=customer_data.customerId)
            return CustomerAnalysisResponse(**cached_result)
        
        # Perform AI analysis
        analysis_result = await ai_engine.analyze_customer(customer_data)
        
        # Cache result for 1 hour
        background_tasks.add_task(
            cache_service.set,
            cache_key,
            analysis_result.dict(),
            expire_seconds=3600
        )
        
        logger.info("Customer analysis completed", customer_id=customer_data.customerId)
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Customer analysis failed", customer_id=customer_data.customerId, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Customer analysis failed: {str(e)}"
        )

# Sentiment Analysis endpoint
@app.post("/analyze-sentiment", response_model=SentimentResponse, tags=["Sentiment Analysis"])
async def analyze_sentiment(request: SentimentRequest):
    """Advanced sentiment analysis using transformer models"""
    try:
        logger.info("Starting sentiment analysis", text_length=len(request.text))
        
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        result = await ai_engine.analyze_sentiment(request.text, request.context)
        
        logger.info("Sentiment analysis completed", sentiment=result.sentiment)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Sentiment analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )

# Churn Prediction endpoint
@app.post("/predict-churn", response_model=ChurnPredictionResponse, tags=["Churn Prediction"])
async def predict_churn(customer_data: CustomerData):
    """ML-powered churn probability prediction"""
    try:
        logger.info("Starting churn prediction", customer_id=customer_data.customerId)
        
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        result = await ai_engine.predict_churn(customer_data)
        
        logger.info("Churn prediction completed", 
                   customer_id=customer_data.customerId,
                   churn_probability=result.churnProbability)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Churn prediction failed", 
                    customer_id=customer_data.customerId, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Churn prediction failed: {str(e)}"
        )

# Insights Generation endpoint
@app.post("/generate-insights", response_model=InsightsResponse, tags=["Insights"])
async def generate_insights(customer_data: CustomerData):
    """Generate natural language insights about customer"""
    try:
        logger.info("Starting insights generation", customer_id=customer_data.customerId)
        
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        result = await ai_engine.generate_insights(customer_data)
        
        logger.info("Insights generation completed", customer_id=customer_data.customerId)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Insights generation failed", 
                    customer_id=customer_data.customerId, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Insights generation failed: {str(e)}"
        )

# Recommendations endpoint
@app.post("/recommend-actions", response_model=RecommendationsResponse, tags=["Recommendations"])
async def recommend_actions(customer_data: CustomerData):
    """Generate personalized action recommendations"""
    try:
        logger.info("Starting recommendations generation", customer_id=customer_data.customerId)
        
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        result = await ai_engine.recommend_actions(customer_data)
        
        logger.info("Recommendations generated", customer_id=customer_data.customerId)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Recommendations generation failed", 
                    customer_id=customer_data.customerId, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Recommendations generation failed: {str(e)}"
        )

# Customer Segmentation endpoint
@app.get("/customer-segments", tags=["Segmentation"])
async def get_customer_segments():
    """Get available customer segments from AI analysis"""
    try:
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        segments = await ai_engine.get_available_segments()
        return {
            "segments": segments,
            "total_segments": len(segments),
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get customer segments", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get customer segments: {str(e)}"
        )

# Batch Analysis endpoint
@app.post("/batch-analyze", tags=["Batch Processing"])
async def batch_analyze(request: BatchAnalysisRequest):
    """Process multiple customers in batch"""
    try:
        customer_count = len(request.customers)
        logger.info("Starting batch analysis", customer_count=customer_count)
        
        if not ai_engine or not ai_engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="AI models are still loading. Please try again in a few moments."
            )
        
        if customer_count > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 100 customers"
            )
        
        results = await ai_engine.batch_analyze(request.customers)
        
        logger.info("Batch analysis completed", 
                   processed_count=len(results),
                   total_submitted=customer_count)
        
        return {
            "results": results,
            "processed": len(results),
            "total_submitted": customer_count,
            "processing_time": datetime.utcnow().isoformat(),
            "success_rate": len(results) / customer_count if customer_count > 0 else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(500)
async def internal_server_error(request, exc):
    logger.error("Internal server error", error=str(exc))
    return {"error": "Internal server error", "detail": "Please try again later"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True if os.getenv("ENVIRONMENT") == "development" else False,
        log_level="info"
    )
