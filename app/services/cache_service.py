import asyncio
import json
import time
from typing import Any, Optional, Dict
import structlog

logger = structlog.get_logger(__name__)

class CacheService:
    def __init__(self):
        """Railway-optimized in-memory cache service"""
        self.cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        self.max_size = 10000  # Railway can handle larger cache
        self.default_ttl = 3600  # 1 hour default TTL
        self.platform = "Railway"
        
        logger.info("🗄️ [Railway] Cache service initialized with in-memory storage")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Railway-optimized cache"""
        self.cache_stats["total_requests"] += 1
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if entry has expired
            if entry["expires_at"] > time.time():
                self.cache_stats["hits"] += 1
                logger.debug(f"[Railway] Cache hit for key: {key}")
                return entry["value"]
            else:
                # Remove expired entry
                del self.cache[key]
                logger.debug(f"[Railway] Cache entry expired for key: {key}")
        
        self.cache_stats["misses"] += 1
        logger.debug(f"[Railway] Cache miss for key: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Railway-optimized cache"""
        try:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Railway memory management - remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest 10% of entries
                oldest_keys = sorted(self.cache.keys(), 
                                   key=lambda k: self.cache[k]["created_at"])[:int(self.max_size * 0.1)]
                for old_key in oldest_keys:
                    del self.cache[old_key]
                logger.info(f"[Railway] Cache cleanup: removed {len(oldest_keys)} entries")
            
            # Store with Railway optimization
            self.cache[key] = {
                "value": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl,
                "platform": "Railway"
            }
            
            logger.debug(f"[Railway] Cache set for key: {key}, TTL: {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"[Railway] Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Railway cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"[Railway] Cache deleted for key: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"[Railway] Cache delete failed for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Railway cache"""
        try:
            cache_size = len(self.cache)
            self.cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}
            logger.info(f"[Railway] Cache cleared: {cache_size} entries removed")
            return True
        except Exception as e:
            logger.error(f"[Railway] Cache clear failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Railway cache statistics"""
        hit_rate = (self.cache_stats["hits"] / max(self.cache_stats["total_requests"], 1)) * 100
        
        return {
            "platform": "Railway",
            "cache_type": "in-memory",
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 2),
            "stats": self.cache_stats,
            "memory_usage": "optimized_for_railway"
        }
