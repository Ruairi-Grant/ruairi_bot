"""Configuration loader for Ruairi Bot"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Config file {self.config_path} not found")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: get('openai.model') returns config['openai']['model']
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            logger.warning(f"Configuration key '{key_path}' not found")
            return None
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    # Convenience properties for commonly used configs
    @property
    def openai_model(self) -> str:
        return self.get('openai.model', 'gpt-4o-mini')
    
    @property
    def embedding_model(self) -> str:
        return self.get('openai.embedding_model', 'text-embedding-3-small')
    
    @property
    def chunk_size(self) -> int:
        return self.get('text_processing.chunk_size', 500)
    
    @property
    def chunk_overlap(self) -> int:
        return self.get('text_processing.chunk_overlap', 50)
    
    @property
    def chunk_separators(self) -> list:
        return self.get('text_processing.separators', ["\n\n", "\n", ".", " ", ""])
    
    @property
    def max_iterations(self) -> int:
        return self.get('chat.max_iterations', 5)
    
    @property
    def system_name(self) -> str:
        return self.get('chat.system_name', 'Ruairi Grant')
    
    @property
    def qna_threshold(self) -> float:
        return self.get('vector_search.qna_threshold', 0.7)
    
    @property
    def thesis_threshold(self) -> float:
        return self.get('vector_search.thesis_threshold', 1.0)
    
    @property
    def qna_n_results(self) -> int:
        return self.get('vector_search.qna_n_results', 1)
    
    @property
    def thesis_n_results(self) -> int:
        return self.get('vector_search.thesis_n_results', 2)

# Global config instance
config = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global config instance"""
    global config
    if config is None:
        config = Config(config_path)
    return config