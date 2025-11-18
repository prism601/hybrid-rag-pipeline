"""
Caching layer using Redis
"""
from typing import Optional, Any
import pickle
import hashlib
import logging
from functools import wraps

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
    ):
        """
        Initialize cache manager

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self.enabled = REDIS_AVAILABLE

        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False,
                )
                # Test connection
                self.client.ping()
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.enabled = False
        else:
            logger.warning("Redis not available, caching disabled")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None

        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.enabled:
            return False

        try:
            ttl = ttl or self.ttl
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache"""
        if not self.enabled:
            return False

        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return False

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()


# Global cache instance
cache_manager = CacheManager(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    ttl=settings.cache_ttl,
)


def cached(ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{cache_manager.generate_key(*args, **kwargs)}"

            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache_manager.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator
