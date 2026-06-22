import os
import yaml
from functools import lru_cache


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    加载并缓存 config.yaml
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Cannot Find Config: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict or {}

global_config = get_config()
