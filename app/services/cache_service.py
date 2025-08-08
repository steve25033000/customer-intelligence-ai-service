import redis
import json
import logging
import os
from typing import Any, Optional, Dict
import structlog

logger = structlog.get_logger()

class CacheService:
    """Redis-based caching service for AI results"""
    
    def __init__(self):
        self.redis_client = None
        self.connected = False
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with fallback to in-memory cache"""
        try:
            # Try to connect to Redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis cache connected")
            
        except Exception as e:
            logger.warning("❌ Redis not available, using in-memory cache", error=str(e))
            # Fallback to simple in-memory cache
            self.redis_client = {}
            self.connected = False
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key"""
        try:
            if self.connected and hasattr(self.redis_client, 'get'):
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            elif isinstance(self.redis_client, dict):
                # In-memory fallback
                return self.redis_client.get(key)
            return None
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Dict[str, Any], expire_seconds: int = 3600):
        """Set cached value with expiration"""
        try:
            if self.connected and hasattr(self.redis_client, 'setex'):
                self.redis_client.setex(key, expire_seconds, json.dumps(value))
            elif isinstance(self.redis_client, dict):
                # In-memory fallback (no expiration for simplicity)
                self.redis_client[key] = value
                
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
    
    def is_connected(self) -> bool:
        """Check if cache is connected"""
        return self.connected
