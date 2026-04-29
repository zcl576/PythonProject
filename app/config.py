from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类
    
    定义应用的配置项，包括 Estate AI 服务和 LLM 服务的相关配置。
    """
    base_dir: str = "app"  # 应用根目录

    # Estate AI 服务配置
    estate_ai_base_url: str = "http://127.0.0.1:4002"  # Estate AI 服务基础 URL
    estate_ai_timeout_seconds: float = 10.0  # Estate AI 服务请求超时时间（秒）

    # LLM 服务配置
    llm_enabled: bool = False  # 是否启用 LLM 服务
    llm_base_url: str = "https://api.openai.com/v1"  # LLM 服务基础 URL
    llm_api_key: str | None = None  # LLM 服务 API 密钥
    llm_model: str = "gpt-4o-mini"  # 使用的 LLM 模型
    llm_timeout_seconds: float = 20.0  # LLM 服务请求超时时间（秒）

    #redis
    redis_enabled: bool = False
    redis_host: str | None = None
    redis_port: int = 6379
    redis_password: str | None = None
    redis_ssl: bool = False
    redis_ssl_cert_reqs: str = "none"

    #mysql
    mysql_host: str | None = None
    mysql_port: int
    mysql_user: str | None = None
    mysql_password: str | None = None
    mysql_db: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")  # 配置加载设置


@lru_cache
def get_settings() -> Settings:
    """获取应用配置实例
    
    使用 lru_cache 装饰器缓存配置实例，确保应用运行期间只创建一次配置对象。
    
    Returns:
        Settings: 应用配置实例
    """
    return Settings()
