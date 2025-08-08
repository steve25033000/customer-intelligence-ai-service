import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog
import warnings

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from app.models.schemas import (
    CustomerData,
    CustomerAnalysisResponse,
    SentimentResponse,
    ChurnPredictionResponse,
    InsightsResponse,
    RecommendationsResponse,
    RiskLevel,
    SentimentType,
    ContextType
)

logger = structlog.get_logger()

class AIEngine:
    """
    Advanced AI engine for Customer Intelligence SaaS
    Handles customer analysis, sentiment analysis, and churn prediction
    """
    
    def __init__(self, cache_service):
        self.cache_service = cache_service
        self.is_initialized = False
        self.models = {}
        self.model_status = {
            "sentiment_analyzer": False,
            "embedding_model": False,
            "churn_model": False,
            "text_generator": False
        }
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_analyses = 0
        self.initialization_time = None
    
    async def initialize_models(self):
        """Initialize all AI models with comprehensive error handling"""
        start_time = datetime.utcnow()
        logger.info("🤖 Starting AI model initialization...")
        
        try:
            # Load models in sequence with detailed tracking
            model_results = {}
            
            logger.info("📊 Loading sentiment analysis model...")
            model_results['sentiment'] = await self._load_sentiment_model()
            
            logger.info("🔗 Loading embedding model...")
            model_results['embedding'] = await self._load_embedding_model()
            
            logger.info("🔮 Initializing churn prediction model...")
            model_results['churn'] = await self._initialize_churn_model()
            
            # Count successful models
            successful_models = sum(model_results.values())
            total_models = len(model_results)
            
            logger.info(f"📊 Model loading results: {successful_models}/{total_models} loaded")
            
            # Determine initialization success (need at least 2/3 models)
            if successful_models >= 2:
                self.is_initialized = True
                self.initialization_time = datetime.utcnow() - start_time
                logger.info(f"✅ AI Engine initialized successfully ({successful_models}/{total_models} models)")
                logger.info(f"⏱️ Initialization completed in {self.initialization_time.total_seconds():.2f} seconds")
            else:
                self.is_initialized = False
                raise Exception(f"Insufficient models loaded: only {successful_models}/{total_models}")
                
        except Exception as e:
            logger.error(f"❌ AI Engine initialization failed: {str(e)}")
            self.is_initialized = False
            raise
    
    async def _load_sentiment_model(self):
        """Load transformer-based sentiment analysis model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            # Load tokenizer and model with safe tensor loading
            self.models['sentiment_tokenizer'] = AutoTokenizer.from_pretrained(model_name)
            self.models['sentiment_model'] = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                use_safetensors=True
            )
            
            # Create pipeline with updated parameters
            self.models['sentiment_pipeline'] = pipeline(
                "sentiment-analysis",
                model=self.models['sentiment_model'],
                tokenizer=self.models['sentiment_tokenizer'],
                device=-1,  # Force CPU usage
                top_k=None  # Replace deprecated return_all_scores=True
            )
            
            self.model_status["sentiment_analyzer"] = True
            logger.info("✅ Sentiment model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {str(e)}")
            logger.info("⚠️ Will use rule-based sentiment analysis as fallback")
            self.model_status["sentiment_analyzer"] = False
            return False
    
    async def _load_embedding_model(self):
        """Load sentence transformer embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load lightweight but effective embedding model
            self.models['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_status["embedding_model"] = True
            
            logger.info("✅ Embedding model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {str(e)}")
            self.model_status["embedding_model"] = False
            return False
    
    async def _initialize_churn_model(self):
        """Initialize machine learning churn prediction model"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Initialize the model
            self.models['churn_model'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            # Generate synthetic training data
            X_train, y_train = self._generate_synthetic_churn_data()
            
            # Split for validation
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Train the model
            self.models['churn_model'].fit(X_train_split, y_train_split)
            
            # Validate model performance
            y_pred = self.models['churn_model'].predict(X_test_split)
            accuracy = accuracy_score(y_test_split, y_pred)
            logger.info(f"📈 Churn model accuracy: {accuracy:.2%}")
            
            self.model_status["churn_model"] = True
            logger.info("✅ Churn model initialized and validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize churn model: {str(e)}")
            self.model_status["churn_model"] = False
            return False
    
    def _generate_synthetic_churn_data(self):
        """Generate realistic synthetic training data for churn prediction"""
        np.random.seed(42)
        n_samples = 2000
        
        # Feature engineering: 7 features that correlate with churn
        X = np.random.rand(n_samples, 7)
        
        # Scale features to realistic business ranges
        X[:, 0] = X[:, 0] * 365      # days_since_last_purchase (0-365 days)
        X[:, 1] = X[:, 1] * 50       # total_purchases (0-50 purchases)
        X[:, 2] = X[:, 2] * 15000    # total_spent ($0-$15,000)
        X[:, 3] = X[:, 3] * 100      # email_opens (0-100 opens)
        X[:, 4] = X[:, 4] * 200      # website_visits (0-200 visits)
        X[:, 5] = X[:, 5] * 25       # support_tickets (0-25 tickets)
        X[:, 6] = X[:, 6] * 100      # engagement_score (0-100)
        
        # Generate realistic churn labels based on business logic
        churn_probability = (
            (X[:, 0] > 90) * 0.3 +      # Long time since purchase
            (X[:, 1] < 2) * 0.4 +       # Few purchases
            (X[:, 2] < 100) * 0.3 +     # Low spending
            (X[:, 3] < 5) * 0.2 +       # Low email engagement
            (X[:, 4] < 10) * 0.2 +      # Low website activity
            (X[:, 5] > 10) * 0.3 +      # Many support tickets
            (X[:, 6] < 30) * 0.4        # Low engagement score
        )
        
        # Convert to binary labels with some noise
        y = (churn_probability > 0.5).astype(int)
        
        # Add some realistic noise (10% label flip)
        noise_indices = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
        y[noise_indices] = 1 - y[noise_indices]
        
        return X, y
    
    def is_ready(self) -> bool:
        """Check if AI engine is ready to process requests"""
        return self.is_initialized and any(self.model_status.values())
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status information"""
        return {
            "initialized": self.is_initialized,
            "models": self.model_status,
            "ready": self.is_ready(),
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate": self.successful_analyses / max(self.total_analyses, 1),
            "initialization_time": str(self.initialization_time) if self.initialization_time else None,
            "last_check": datetime.utcnow().isoformat()
        }
    
    async def analyze_customer(self, customer_data: CustomerData) -> CustomerAnalysisResponse:
        """Comprehensive AI-powered customer analysis"""
        try:
            self.total_analyses += 1
            logger.info("🔍 Starting customer analysis", customer_id=customer_data.customerId)
            
            # Extract comprehensive customer features
            features = self._extract_customer_features(customer_data)
            
            # Generate customer embedding if embedding model is available
            customer_embedding = None
            if self.model_status.get("embedding_model"):
                customer_description = self._create_customer_description(customer_data, features)
                customer_embedding = self.models['embedding_model'].encode([customer_description])[0]
            
            # AI-powered customer segmentation
            segment_info = self._determine_ai_segment(features, customer_embedding)
            
            # Calculate behavioral score using ML techniques
            behavioral_score = self._calculate_ai_behavioral_score(features)
            
            # Predict churn using trained model
            churn_analysis = await self._predict_churn_advanced(customer_data, features)
            
            # Generate AI insights
            insights = self._generate_ai_insights(features, segment_info, churn_analysis)
            
            # Create personalized recommendations
            recommendations = self._generate_ai_recommendations(features, segment_info, churn_analysis)
            
            # Build comprehensive response
            response = CustomerAnalysisResponse(
                customerId=customer_data.customerId,
                aiSegment=segment_info["name"],
                segmentDescription=segment_info["description"],
                behavioralScore=behavioral_score,
                churnRisk=churn_analysis["risk_level"],
                churnProbability=churn_analysis["probability"],
                analysis={
                    "totalSpent": features["total_spent"],
                    "purchaseCount": features["total_purchases"],
                    "avgOrderValue": features["avg_order_value"],
                    "engagementLevel": features["engagement_level"],
                    "loyaltyScore": features["loyalty_score"],
                    "customerValue": features["customer_value"],
                    "riskFactors": churn_analysis.get("risk_factors", []),
                    "growthPotential": features.get("growth_potential", "Medium")
                },
                recommendations=recommendations,
                insights=insights,
                confidence=0.92 if self.model_status["churn_model"] else 0.78,
                modelVersion="customer-intelligence-ai-v2.1",
                analyzedAt=datetime.utcnow()
            )
            
            self.successful_analyses += 1
            logger.info("✅ Customer analysis completed", 
                       customer_id=customer_data.customerId,
                       segment=segment_info["name"],
                       score=behavioral_score,
                       churn_risk=churn_analysis["risk_level"].value)
            
            return response
            
        except Exception as e:
            logger.error("❌ Customer analysis failed", 
                        customer_id=customer_data.customerId,
                        error=str(e))
            raise
    
    async def analyze_sentiment(self, text: str, context: ContextType) -> SentimentResponse:
        """Advanced sentiment analysis with transformer models and rule-based fallback"""
        try:
            logger.info("💭 Analyzing sentiment", text_length=len(text), context=context.value)
            
            # Try transformer model first if available
            if self.model_status.get("sentiment_analyzer") and 'sentiment_pipeline' in self.models:
                return await self._analyze_sentiment_transformer(text, context)
            else:
                # Use enhanced rule-based analysis
                return await self._analyze_sentiment_enhanced_rules(text, context)
                
        except Exception as e:
            logger.error("❌ Sentiment analysis failed", error=str(e))
            # Fallback to basic rules
            return await self._analyze_sentiment_basic_rules(text, context)
    
    async def _analyze_sentiment_transformer(self, text: str, context: ContextType) -> SentimentResponse:
        """Transformer-based sentiment analysis"""
        try:
            # Truncate text to model limits
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
                logger.info("📝 Text truncated to fit model limits")
            
            # Get sentiment scores from transformer pipeline
            results = self.models['sentiment_pipeline'](text)
            
            # Process transformer results
            if isinstance(results[0], list):
                sentiment_scores = {result['label'].lower(): result['score'] for result in results[0]}
            else:
                sentiment_scores = {results[0]['label'].lower(): results[0]['score']}
            
            # Map transformer labels to our sentiment types
            label_mapping = {
                'label_0': 'negative',
                'label_1': 'neutral', 
                'label_2': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'positive': 'positive'
            }
            
            # Find primary sentiment
            primary_label = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
            mapped_sentiment = label_mapping.get(primary_label, primary_label)
            confidence = sentiment_scores[primary_label]
            
            # Convert to our enum types
            sentiment_type = {
                'positive': SentimentType.POSITIVE,
                'negative': SentimentType.NEGATIVE,
                'neutral': SentimentType.NEUTRAL
            }.get(mapped_sentiment, SentimentType.NEUTRAL)
            
            # Determine emotion and urgency
            emotion = self._determine_emotion_advanced(sentiment_scores, context, confidence)
            urgency = self._determine_urgency_advanced(sentiment_type, context, confidence)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases_advanced(text)
            
            response = SentimentResponse(
                text=text,
                sentiment=sentiment_type,
                confidence=confidence,
                emotion=emotion,
                urgency=urgency,
                scores=sentiment_scores,
                context=context,
                keyPhrases=key_phrases,
                modelVersion="roberta-sentiment-v2.0",
                analyzedAt=datetime.utcnow()
            )
            
            logger.info("✅ Transformer sentiment analysis completed", 
                       sentiment=sentiment_type.value,
                       confidence=confidence,
                       emotion=emotion)
            
            return response
            
        except Exception as e:
            logger.warning("⚠️ Transformer sentiment failed, using enhanced rules", error=str(e))
            return await self._analyze_sentiment_enhanced_rules(text, context)
    
    async def _analyze_sentiment_enhanced_rules(self, text: str, context: ContextType) -> SentimentResponse:
        """Enhanced rule-based sentiment analysis with weighted words"""
        try:
            # Comprehensive sentiment dictionaries with weights
            positive_words = {
                # Strong positive
                "love": 0.9, "amazing": 0.9, "excellent": 0.8, "fantastic": 0.8,
                "outstanding": 0.8, "brilliant": 0.8, "perfect": 0.9, "wonderful": 0.8,
                # Medium positive  
                "great": 0.7, "good": 0.6, "nice": 0.5, "happy": 0.7,
                "satisfied": 0.7, "pleased": 0.6, "enjoy": 0.6, "like": 0.5,
                # Mild positive
                "ok": 0.3, "fine": 0.4, "decent": 0.4, "adequate": 0.3,
                # Gratitude
                "thank": 0.6, "thanks": 0.6, "appreciate": 0.7, "grateful": 0.7
            }
            
            negative_words = {
                # Strong negative
                "hate": 0.9, "terrible": 0.9, "awful": 0.8, "horrible": 0.9,
                "disgusting": 0.9, "furious": 0.9, "outraged": 0.8,
                # Medium negative
                "bad": 0.6, "poor": 0.6, "disappointed": 0.7, "frustrated": 0.7,
                "annoyed": 0.6, "upset": 0.6, "angry": 0.8, "mad": 0.7,
                # Problem indicators
                "problem": 0.5, "issue": 0.4, "trouble": 0.6, "difficult": 0.5,
                "broken": 0.7, "failed": 0.6, "error": 0.5, "bug": 0.4,
                # Mild negative
                "meh": 0.3, "boring": 0.4, "slow": 0.4, "confusing": 0.5
            }
            
            # Context-specific adjustments
            context_multipliers = {
                ContextType.SUPPORT_TICKET: 1.2,  # Amplify negative sentiment
                ContextType.REVIEW: 1.1,
                ContextType.CHAT: 1.0,
                ContextType.EMAIL: 0.9,
                ContextType.SOCIAL_MEDIA: 1.1,
                ContextType.GENERAL: 1.0
            }
            
            text_lower = text.lower()
            multiplier = context_multipliers.get(context, 1.0)
            
            # Calculate weighted scores
            positive_score = sum(weight * multiplier for word, weight in positive_words.items() 
                               if word in text_lower)
            negative_score = sum(weight * multiplier for word, weight in negative_words.items() 
                               if word in text_lower)
            
            # Determine primary sentiment
            if positive_score > negative_score and positive_score > 0:
                sentiment = SentimentType.POSITIVE
                confidence = min(0.6 + (positive_score * 0.1), 0.95)
                dominant_score = positive_score
            elif negative_score > positive_score and negative_score > 0:
                sentiment = SentimentType.NEGATIVE
                confidence = min(0.6 + (negative_score * 0.1), 0.95)
                dominant_score = negative_score
            else:
                sentiment = SentimentType.NEUTRAL
                confidence = 0.6
                dominant_score = 0
            
            # Determine emotion based on intensity
            emotion = self._determine_emotion_from_score(sentiment, dominant_score, context)
            urgency = self._determine_urgency_from_sentiment(sentiment, context, confidence)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases_basic(text)
            
            response = SentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                emotion=emotion,
                urgency=urgency,
                scores={
                    "positive": positive_score,
                    "negative": negative_score,
                    "neutral": max(0, 1.0 - positive_score - negative_score)
                },
                context=context,
                keyPhrases=key_phrases,
                modelVersion="enhanced-rules-v2.1",
                analyzedAt=datetime.utcnow()
            )
            
            logger.info("✅ Enhanced rule-based sentiment completed", 
                       sentiment=sentiment.value,
                       confidence=confidence)
            
            return response
            
        except Exception as e:
            logger.error("❌ Enhanced rule sentiment failed", error=str(e))
            return await self._analyze_sentiment_basic_rules(text, context)
    
    async def _analyze_sentiment_basic_rules(self, text: str, context: ContextType) -> SentimentResponse:
        """Basic fallback sentiment analysis"""
        try:
            positive_words = ["good", "great", "love", "excellent", "happy", "satisfied"]
            negative_words = ["bad", "terrible", "hate", "angry", "frustrated", "disappointed"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = SentimentType.POSITIVE
                confidence = 0.7
            elif negative_count > positive_count:
                sentiment = SentimentType.NEGATIVE
                confidence = 0.7
            else:
                sentiment = SentimentType.NEUTRAL
                confidence = 0.6
            
            return SentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                emotion="neutral",
                urgency="normal",
                scores={"positive": positive_count, "negative": negative_count, "neutral": 0},
                context=context,
                keyPhrases=[],
                modelVersion="basic-rules-v1.0",
                analyzedAt=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("❌ Basic sentiment analysis failed", error=str(e))
            raise
    
    async def predict_churn(self, customer_data: CustomerData) -> ChurnPredictionResponse:
        """Advanced churn prediction using ML models - returns proper response object"""  # Correct!
        try:
            # ... rest of code

            features = self._extract_customer_features(customer_data)
            churn_analysis = await self._predict_churn_advanced(customer_data, features)
            
            # Convert dictionary to proper response object
            response = ChurnPredictionResponse(
                customerId=customer_data.customerId,
                churnProbability=churn_analysis["probability"],
                churnPrediction=churn_analysis["prediction"],
                riskLevel=churn_analysis["risk_level"],
                riskColor=churn_analysis["color"],
                contributingFactors=churn_analysis["risk_factors"],
                recommendations=churn_analysis["recommendations"],
                daysUntilChurn=churn_analysis.get("days_until_churn"),
                confidenceScore=0.88 if self.model_status["churn_model"] else 0.75,
                modelVersion="churn-prediction-v2.1",
                predictionDate=datetime.utcnow()
            )
            
            logger.info("✅ Churn prediction completed", 
                       customer_id=customer_data.customerId,
                       churn_probability=response.churnProbability,
                       risk_level=response.riskLevel.value)
            
            return response
            
        except Exception as e:
            logger.error("❌ Churn prediction failed", 
                        customer_id=customer_data.customerId,
                        error=str(e))
            raise


    
    async def _predict_churn_advanced(self, customer_data: CustomerData, features: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced churn prediction with ML model"""
        try:
            if self.model_status.get("churn_model") and 'churn_model' in self.models:
                # Use trained ML model
                feature_array = np.array([
                    features["days_since_last_purchase"],
                    features["total_purchases"],
                    features["total_spent"],
                    features["email_opens"],
                    features["website_visits"],
                    features["support_tickets"],
                    features["engagement_score"]
                ]).reshape(1, -1)
                
                # Get churn probability from trained model
                churn_probability = self.models['churn_model'].predict_proba(feature_array)[0][1]
                
                # Get feature importance for explanations
                feature_importance = self.models['churn_model'].feature_importances_
                feature_names = [
                    "days_since_last_purchase", "total_purchases", "total_spent",
                    "email_opens", "website_visits", "support_tickets", "engagement_score"
                ]
                
                risk_factors = self._analyze_risk_factors_ml(features, feature_importance, feature_names)
                
            else:
                # Use rule-based prediction
                churn_probability, risk_factors = self._predict_churn_rules(features)
            
            # Determine risk level and color
            if churn_probability >= 0.8:
                risk_level = RiskLevel.CRITICAL
                risk_color = "red"
            elif churn_probability >= 0.6:
                risk_level = RiskLevel.HIGH
                risk_color = "orange"
            elif churn_probability >= 0.4:
                risk_level = RiskLevel.MEDIUM
                risk_color = "yellow"
            else:
                risk_level = RiskLevel.LOW
                risk_color = "green"
            
            # Generate retention recommendations
            recommendations = self._generate_retention_recommendations(features, risk_level, churn_probability)
            
            # Estimate days until churn
            days_until_churn = self._estimate_days_until_churn(churn_probability, features)
            
            return {
                "probability": round(churn_probability, 3),
                "prediction": churn_probability > 0.5,
                "risk_level": risk_level,
                "color": risk_color,
                "risk_factors": risk_factors,
                "recommendations": recommendations,
                "days_until_churn": days_until_churn
            }
            
        except Exception as e:
            logger.error("❌ Advanced churn prediction failed", error=str(e))
            # Fallback to simple rule-based prediction
            return await self._predict_churn_simple_fallback(features)
    
    # Helper methods for customer analysis
    
    def _extract_customer_features(self, customer_data: CustomerData) -> Dict[str, Any]:
        """Extract comprehensive features from customer data"""
        purchase_history = customer_data.purchaseHistory or []
        engagement_data = customer_data.engagementData or {}
        demographics = customer_data.demographics or {}
        
        # Basic metrics
        total_purchases = len(purchase_history)
        total_spent = sum(p.get('amount', 0) for p in purchase_history)
        avg_order_value = total_spent / total_purchases if total_purchases > 0 else 0
        
        # Engagement metrics
        email_opens = engagement_data.get('emailOpens', 0)
        website_visits = engagement_data.get('websiteVisits', 0)
        support_tickets = engagement_data.get('supportTickets', 0)
        days_as_customer = engagement_data.get('daysAsCustomer', 30)
        
        # Calculate derived metrics
        engagement_score = min((email_opens * 0.5 + website_visits * 0.3 + total_purchases * 2), 100)
        loyalty_score = min(total_purchases * 3 + min(total_spent / 100, 50), 100)
        customer_value = total_spent + (engagement_score * 10)
        
        # Recency analysis
        days_since_last_purchase = 365  # Default for new customers
        if purchase_history:
            try:
                last_purchase_date = max(p.get('date', '2023-01-01') for p in purchase_history)
                # Simplified - in production you'd parse dates properly
                days_since_last_purchase = 30  # Assume recent for demo
            except:
                days_since_last_purchase = 90
        
        # Growth potential analysis
        if total_purchases == 0:
            growth_potential = "High" if engagement_score > 50 else "Medium"
        elif total_purchases < 3:
            growth_potential = "High"
        elif avg_order_value < 100:
            growth_potential = "Medium"
        else:
            growth_potential = "Established"
        
        return {
            "total_purchases": total_purchases,
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "email_opens": email_opens,
            "website_visits": website_visits,
            "support_tickets": support_tickets,
            "days_as_customer": days_as_customer,
            "days_since_last_purchase": days_since_last_purchase,
            "engagement_score": engagement_score,
            "loyalty_score": loyalty_score,
            "customer_value": customer_value,
            "growth_potential": growth_potential,
            "engagement_level": "High" if engagement_score > 70 else "Medium" if engagement_score > 30 else "Low",
            "spending_tier": "Premium" if total_spent > 2000 else "Standard" if total_spent > 500 else "Basic"
        }
    
    def _create_customer_description(self, customer_data: CustomerData, features: Dict[str, Any]) -> str:
        """Create natural language description for embedding generation"""
        description = f"""
        Customer {customer_data.firstName} {customer_data.lastName} has been with us for {features['days_as_customer']} days.
        They have made {features['total_purchases']} purchases totaling ${features['total_spent']:.2f}.
        Their engagement includes {features['email_opens']} email opens and {features['website_visits']} website visits.
        Average order value is ${features['avg_order_value']:.2f} with {features['support_tickets']} support tickets.
        Overall engagement level: {features['engagement_level']}, Spending tier: {features['spending_tier']}.
        """
        return description.strip()
    
    def _determine_ai_segment(self, features: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> Dict[str, str]:
        """AI-powered customer segmentation using multiple factors"""
        total_spent = features["total_spent"]
        engagement_score = features["engagement_score"]
        loyalty_score = features["loyalty_score"]
        purchases = features["total_purchases"]
        customer_value = features["customer_value"]
        growth_potential = features["growth_potential"]
        
        # Advanced segmentation logic
        if total_spent > 5000 and engagement_score > 80:
            return {
                "name": "VIP Champions",
                "description": "Highest value customers with exceptional engagement and loyalty"
            }
        elif total_spent > 2000 and loyalty_score > 70:
            return {
                "name": "Premium Advocates", 
                "description": "High-value customers who consistently engage and purchase"
            }
        elif growth_potential == "High" and engagement_score > 60:
            return {
                "name": "Rising Stars",
                "description": "High-potential customers showing strong engagement patterns"
            }
        elif purchases > 0 and engagement_score > 50:
            return {
                "name": "Engaged Buyers",
                "description": "Active customers with good purchase and engagement history"
            }
        elif engagement_score > 70 and purchases == 0:
            return {
                "name": "Hot Prospects",
                "description": "Highly engaged prospects with strong conversion potential"
            }
        elif total_spent > 500:
            return {
                "name": "Steady Customers",
                "description": "Reliable customers with consistent purchase patterns"
            }
        elif features["days_as_customer"] < 30:
            return {
                "name": "New Arrivals",
                "description": "Recently onboarded customers in the discovery phase"
            }
        else:
            return {
                "name": "Opportunity Customers",
                "description": "Customers with untapped potential for growth and engagement"
            }
    
    def _calculate_ai_behavioral_score(self, features: Dict[str, Any]) -> int:
        """Calculate comprehensive behavioral score using AI techniques"""
        score = 0
        
        # Spending component (35% weight)
        spending_normalized = min(features["total_spent"] / 100, 35)
        score += spending_normalized
        
        # Engagement component (30% weight)
        engagement_component = (features["engagement_score"] / 100) * 30
        score += engagement_component
        
        # Loyalty component (20% weight)
        loyalty_component = (features["loyalty_score"] / 100) * 20
        score += loyalty_component
        
        # Recency component (15% weight)
        recency_score = max(0, 15 - (features["days_since_last_purchase"] / 30))
        score += recency_score
        
        # Bonus for high-value indicators
        if features["total_purchases"] > 10:
            score += 5
        if features["avg_order_value"] > 200:
            score += 3
        if features["support_tickets"] == 0 and features["total_purchases"] > 0:
            score += 2  # Self-sufficient customer bonus
        
        return min(int(score), 100)
    
    def _generate_ai_insights(self, features: Dict[str, Any], segment_info: Dict[str, str], churn_analysis: Dict[str, Any]) -> List[str]:
        """Generate AI-powered insights about the customer"""
        insights = []
        
        # Value insights
        if features["customer_value"] > 2000:
            insights.append(f"High-value customer generating ${features['customer_value']:.0f} in total value")
        
        # Engagement insights
        if features["engagement_score"] > 80:
            insights.append("Exceptional engagement indicates strong product-market fit")
        elif features["engagement_score"] < 20:
            insights.append("Low engagement suggests need for targeted re-engagement campaign")
        
        # Purchase pattern insights
        if features["avg_order_value"] > 300:
            insights.append(f"Above-average order value (${features['avg_order_value']:.0f}) indicates premium preference")
        elif features["total_purchases"] > 15:
            insights.append("Frequent purchaser showing strong product adoption")
        
        # Risk insights
        if churn_analysis["probability"] > 0.7:
            insights.append("High churn risk requires immediate retention intervention")
        elif features["days_since_last_purchase"] > 90:
            insights.append("Extended purchase gap may indicate disengagement")
        
        # Growth insights
        if features["growth_potential"] == "High":
            insights.append("Strong growth potential with proper nurturing and upselling")
        
        # Support insights
        if features["support_tickets"] == 0 and features["total_purchases"] > 0:
            insights.append("Self-sufficient customer with smooth product experience")
        elif features["support_tickets"] > 5:
            insights.append("High support activity may indicate product complexity issues")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _generate_ai_recommendations(self, features: Dict[str, Any], segment_info: Dict[str, str], churn_analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized AI recommendations"""
        recommendations = []
        segment_name = segment_info["name"]
        
        # Segment-specific recommendations
        if "VIP" in segment_name or "Premium" in segment_name:
            recommendations.extend([
                "Offer exclusive VIP benefits and early access to new features",
                "Request case study or testimonial participation",
                "Provide dedicated customer success manager"
            ])
        elif "Rising Stars" in segment_name or "Hot Prospects" in segment_name:
            recommendations.extend([
                "Implement targeted nurturing sequence with educational content",
                "Offer personalized product demo or consultation",
                "Provide limited-time upgrade incentive"
            ])
        elif "Engaged Buyers" in segment_name:
            recommendations.extend([
                "Introduce complementary products through cross-selling",
                "Share advanced usage tips and best practices",
                "Invite to user community or beta programs"
            ])
        
        # Risk-based recommendations
        if churn_analysis["probability"] > 0.6:
            recommendations.extend([
                "Launch immediate retention campaign with personalized offer",
                "Schedule customer success call to address concerns",
                "Provide enhanced support and onboarding assistance"
            ])
        elif features["engagement_score"] < 30:
            recommendations.extend([
                "Implement re-engagement campaign with value-focused messaging",
                "Offer product training or consultation session",
                "Simplify user experience and remove friction points"
            ])
        
        # Growth-based recommendations
        if features["avg_order_value"] < 100 and features["total_purchases"] > 3:
            recommendations.append("Present premium upgrade options to increase order value")
        
        if features["total_purchases"] == 0 and features["engagement_score"] > 50:
            recommendations.append("Deploy high-intent conversion campaign with social proof")
        if not recommendations:
            recommendations = [
                "Monitor customer engagement patterns",
                "Maintain current service quality",
                "Consider targeted retention campaigns"
            ]    
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    # Additional helper methods for sentiment analysis
    
    def _determine_emotion_advanced(self, sentiment_scores: Dict[str, float], context: ContextType, confidence: float) -> str:
        """Determine emotion based on sentiment analysis"""
        primary_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
        
        if 'positive' in primary_sentiment.lower() or sentiment_scores.get('positive', 0) > 0.7:
            return "delighted" if confidence > 0.8 else "satisfied"
        elif 'negative' in primary_sentiment.lower() or sentiment_scores.get('negative', 0) > 0.7:
            if context in [ContextType.SUPPORT_TICKET, ContextType.CHAT]:
                return "frustrated" if confidence < 0.8 else "angry"
            else:
                return "disappointed"
        else:
            return "neutral"
    
    def _determine_urgency_advanced(self, sentiment: SentimentType, context: ContextType, confidence: float) -> str:
        """Determine urgency level based on sentiment and context"""
        if sentiment == SentimentType.NEGATIVE and context in [ContextType.SUPPORT_TICKET, ContextType.CHAT] and confidence > 0.7:
            return "high"
        elif sentiment == SentimentType.NEGATIVE and confidence > 0.8:
            return "medium"
        else:
            return "normal"
    
    def _extract_key_phrases_advanced(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation - in production you'd use more sophisticated NLP
        words = text.split()
        key_phrases = []
        
        # Extract longer words and phrases
        for i, word in enumerate(words):
            if len(word) > 6:
                key_phrases.append(word.lower().strip('.,!?'))
            elif i < len(words) - 1 and len(word) > 3:
                phrase = f"{word} {words[i+1]}"
                if len(phrase) > 8:
                    key_phrases.append(phrase.lower().strip('.,!?'))
        
        return key_phrases[:5]
    
    # Simplified helper methods
    
    def _determine_emotion_from_score(self, sentiment: SentimentType, score: float, context: ContextType) -> str:
        if sentiment == SentimentType.POSITIVE:
            return "delighted" if score > 2 else "satisfied"
        elif sentiment == SentimentType.NEGATIVE:
            return "angry" if score > 2 else "frustrated"
        else:
            return "neutral"
    
    def _determine_urgency_from_sentiment(self, sentiment: SentimentType, context: ContextType, confidence: float) -> str:
        if sentiment == SentimentType.NEGATIVE and context in [ContextType.SUPPORT_TICKET, ContextType.CHAT]:
            return "high"
        elif sentiment == SentimentType.NEGATIVE:
            return "medium"
        else:
            return "normal"
    
    def _extract_key_phrases_basic(self, text: str) -> List[str]:
        words = [word.strip('.,!?').lower() for word in text.split() if len(word) > 5]
        return words[:5]
    
    def _predict_churn_rules(self, features: Dict[str, Any]) -> tuple:
        """Rule-based churn prediction fallback"""
        risk_score = 0
        risk_factors = []
        
        # Analyze key risk factors
        if features["total_spent"] < 100:
            risk_score += 0.3
            risk_factors.append({"factor": "Low Total Spending", "impact": "High", "value": f"${features['total_spent']:.2f}"})
        
        if features["engagement_score"] < 30:
            risk_score += 0.4
            risk_factors.append({"factor": "Low Engagement", "impact": "Critical", "value": f"{features['engagement_score']:.1f}/100"})
        
        if features["total_purchases"] == 0:
            risk_score += 0.5
            risk_factors.append({"factor": "No Purchases", "impact": "Critical", "value": "0 purchases"})
        
        if features["days_since_last_purchase"] > 180:
            risk_score += 0.3
            risk_factors.append({"factor": "Long Purchase Gap", "impact": "High", "value": f"{features['days_since_last_purchase']} days"})
        
        churn_probability = min(risk_score, 1.0)
        return churn_probability, risk_factors
    
    def _analyze_risk_factors_ml(self, features: Dict[str, Any], feature_importance: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Analyze risk factors using ML model feature importance"""
        risk_factors = []
        
        # Get top risk factors based on feature importance
        feature_impacts = list(zip(feature_names, feature_importance))
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        
        for feature_name, importance in feature_impacts[:4]:  # Top 4 factors
            if importance > 0.1:  # Only significant factors
                impact_level = "Critical" if importance > 0.3 else "High" if importance > 0.2 else "Medium"
                feature_value = features.get(feature_name.replace("_", ""), 0)
                
                risk_factors.append({
                    "factor": feature_name.replace("_", " ").title(),
                    "impact": impact_level,
                    "value": str(feature_value),
                    "importance": round(importance, 3)
                })
        
        return risk_factors
    
    def _generate_retention_recommendations(self, features: Dict[str, Any], risk_level: RiskLevel, churn_probability: float) -> List[str]:
        """Generate retention-specific recommendations"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Launch immediate emergency retention campaign",
                "Assign dedicated customer success representative",
                "Offer significant value-add incentive or discount"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Initiate proactive outreach within 48 hours",
                "Provide personalized re-engagement content",
                "Schedule product value demonstration call"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Implement nurturing campaign with success stories",
                "Share advanced features and usage tips",
                "Monitor engagement closely for changes"
            ])
        else:
            recommendations.extend([
                "Continue standard engagement practices",
                "Explore upselling and cross-selling opportunities",
                "Request feedback for product improvements"
            ])
        
        return recommendations
    
    def _estimate_days_until_churn(self, churn_probability: float, features: Dict[str, Any]) -> Optional[int]:
        """Estimate days until potential churn"""
        if churn_probability > 0.5:
            # Simple estimation based on engagement patterns
            base_days = int((1 - churn_probability) * 365)
            
            # Adjust based on engagement level
            if features["engagement_score"] < 20:
                base_days = int(base_days * 0.5)  # Faster churn for disengaged
            elif features["engagement_score"] > 60:
                base_days = int(base_days * 1.5)  # Slower churn for engaged
            
            return max(7, base_days)  # Minimum 7 days
        else:
            return None
    
    async def _predict_churn_simple_fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback churn prediction"""
        churn_probability = 0.3  # Default medium risk
        
        if features["total_purchases"] == 0:
            churn_probability = 0.7
        elif features["engagement_score"] < 20:
            churn_probability = 0.6
        elif features["total_spent"] > 1000:
            churn_probability = 0.2
        
        risk_level = RiskLevel.HIGH if churn_probability > 0.6 else RiskLevel.MEDIUM if churn_probability > 0.4 else RiskLevel.LOW
        
        return {
            "probability": churn_probability,
            "prediction": churn_probability > 0.5,
            "risk_level": risk_level,
            "color": "orange" if risk_level == RiskLevel.HIGH else "yellow" if risk_level == RiskLevel.MEDIUM else "green",
            "risk_factors": [{"factor": "Fallback Analysis", "impact": "Unknown", "value": "Limited data"}],
            "recommendations": ["Monitor customer closely", "Implement engagement campaigns"],
            "days_until_churn": 90 if churn_probability > 0.5 else None
        }
    
    async def cleanup(self):
        """Clean up AI engine resources"""
        logger.info("🧹 Cleaning up AI Engine resources...")
        try:
            self.models.clear()
            self.is_initialized = False
            self.total_analyses = 0
            self.successful_analyses = 0
            logger.info("✅ AI Engine cleanup completed")
        except Exception as e:
            logger.error(f"⚠️ Cleanup had issues: {str(e)}")
