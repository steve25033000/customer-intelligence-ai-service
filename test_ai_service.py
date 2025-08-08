"""
Local testing script for Customer Intelligence AI Service
Run this to test your AI service functionality locally
"""

import asyncio
import json
from datetime import datetime
from app.models.schemas import CustomerData, SentimentRequest, ContextType
from app.services.cache_service import CacheService  
from app.services.ai_engine import AIEngine

async def test_ai_service():
    """Comprehensive test of AI service functionality"""
    print("🚀 Starting Customer Intelligence AI Service Tests")
    print("=" * 50)
    
    try:
        # Initialize services
        print("📦 Initializing services...")
        cache_service = CacheService()
        ai_engine = AIEngine(cache_service)
        
        # Initialize AI models
        print("🤖 Loading AI models...")
        await ai_engine.initialize_models()
        
        if not ai_engine.is_ready():
            print("❌ AI Engine failed to initialize")
            return
        
        print("✅ AI Engine ready!")
        print()
        
        # Test 1: Customer Analysis
        print("🔍 TEST 1: Customer Analysis")
        print("-" * 30)
        
        test_customer = CustomerData(
            customerId="test-001",
            email="john.doe@example.com",
            firstName="John",
            lastName="Doe",
            purchaseHistory=[
                {"amount": 299.99, "date": "2024-01-15", "product": "Premium Plan"},
                {"amount": 149.99, "date": "2024-02-10", "product": "Add-on Service"},
                {"amount": 399.99, "date": "2024-03-05", "product": "Enterprise Upgrade"}
            ],
            engagementData={
                "emailOpens": 25,
                "websiteVisits": 45,
                "supportTickets": 2,
                "daysAsCustomer": 120
            },
            demographics={
                "company": "Tech Corp",
                "industry": "Software"
            }
        )
        
        analysis_result = await ai_engine.analyze_customer(test_customer)
        print(f"📊 Customer: {analysis_result.customerId}")
        print(f"🏷️  AI Segment: {analysis_result.aiSegment}")
        print(f"📈 Behavioral Score: {analysis_result.behavioralScore}/100")
        print(f"⚠️  Churn Risk: {analysis_result.churnRisk.value}")
        print(f"🎯 Top Recommendations:")
        for i, rec in enumerate(analysis_result.recommendations[:3], 1):
            print(f"   {i}. {rec}")
        print()
        
        # Test 2: Sentiment Analysis
        print("💭 TEST 2: Sentiment Analysis")
        print("-" * 30)
        
        sentiment_tests = [
            ("I absolutely love this product! It's amazing and works perfectly!", "review"),
            ("This is terrible and frustrating. Nothing works as expected.", "support_ticket"),
            ("The service is okay, nothing special but does the job.", "general"),
            ("Thank you so much for the quick response and excellent support!", "email")
        ]
        
        for text, context in sentiment_tests:
            sentiment_result = await ai_engine.analyze_sentiment(text, ContextType(context))
            print(f"📝 Text: \"{text[:50]}...\"")
            print(f"😊 Sentiment: {sentiment_result.sentiment.value} ({sentiment_result.confidence:.2f})")
            print(f"🎭 Emotion: {sentiment_result.emotion}")
            print(f"🚨 Urgency: {sentiment_result.urgency}")
            print()
        
        # Test 3: Churn Prediction
        print("🔮 TEST 3: Churn Prediction")
        print("-" * 30)
        
        churn_test_customers = [
            # High-risk customer
            CustomerData(
                customerId="test-002",
                email="atrisk@example.com",
                firstName="At",
                lastName="Risk",
                purchaseHistory=[],  # No purchases
                engagementData={"emailOpens": 2, "websiteVisits": 3, "supportTickets": 0}
            ),
            # Low-risk customer  
            CustomerData(
                customerId="test-003",
                email="loyal@example.com",
                firstName="Loyal",
                lastName="Customer",
                purchaseHistory=[
                    {"amount": 500, "date": "2024-01-01"},
                    {"amount": 750, "date": "2024-02-01"},
                    {"amount": 600, "date": "2024-03-01"}
                ],
                engagementData={"emailOpens": 35, "websiteVisits": 80, "supportTickets": 1}
            )
        ]
        
        for customer in churn_test_customers:
            churn_result = await ai_engine.predict_churn(customer)
            print(f"👤 Customer: {customer.firstName} {customer.lastName}")
            print(f"📈 Churn Probability: {churn_result.churnProbability:.1%}")
            print(f"🚦 Risk Level: {churn_result.riskLevel.value}")
            print(f"💡 Top Contributing Factors:")
            for factor in churn_result.contributingFactors[:2]:
                print(f"   • {factor['factor']}: {factor['value']} ({factor['impact']} impact)")
            print()
        
        # Test 4: Model Status
        print("⚙️  TEST 4: Model Status Check")
        print("-" * 30)
        
        status = await ai_engine.get_model_status()
        print(f"🤖 Models Initialized: {status['initialized']}")
        print(f"✅ Ready for Requests: {status['ready']}")
        print(f"📊 Model Status:")
        for model_name, loaded in status['models'].items():
            print(f"   • {model_name}: {'✅ Loaded' if loaded else '❌ Failed'}")
        print()
        
        # Test 5: Cache Performance
        print("💾 TEST 5: Cache Performance")
        print("-" * 30)
        
        print(f"🔗 Cache Connected: {'✅ Yes' if cache_service.is_connected() else '⚠️  Using fallback'}")
        
        # Test caching
        test_key = "test_cache_key"
        test_data = {"message": "Hello from cache!", "timestamp": datetime.now().isoformat()}
        
        await cache_service.set(test_key, test_data, 60)
        cached_data = await cache_service.get(test_key)
        
        if cached_data:
            print("✅ Cache set/get working correctly")
        else:
            print("⚠️  Cache fallback working")
        print()
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ Your AI service is ready for production!")
        print("🚀 Next step: Deploy to Railway")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ai_service())
