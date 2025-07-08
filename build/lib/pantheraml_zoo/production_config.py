# PantheraML Zoo - Production Configuration
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = [
    "ProductionConfig",
    "load_config",
    "save_config",
    "validate_config",
    "get_default_config",
]

@dataclass
class ProductionConfig:
    """Production configuration for PantheraML"""
    
    # Model Configuration
    model_name: str = "llama-3.1-8b"
    model_path: Optional[str] = None
    use_gradient_checkpointing: Union[bool, str] = "unsloth"
    use_reentrant: bool = True
    
    # Training Configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Distributed Training
    max_gpus: Optional[int] = None
    enable_tpu: bool = True
    ddp_timeout_minutes: int = 30
    
    # Performance
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "no"
    compile_model: bool = False
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Monitoring
    logging_level: str = "INFO"
    log_file: Optional[str] = None
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    enable_performance_monitoring: bool = True
    
    # Error Handling
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./checkpoints"
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Memory Management
    enable_cpu_offload: bool = False
    gradient_checkpointing_kwargs: Dict[str, Any] = None
    
    # Output
    output_dir: str = "./output"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    def __post_init__(self):
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProductionConfig":
        """Create from dictionary"""
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration"""
        errors = []
        
        # Validate batch size
        if self.per_device_train_batch_size <= 0:
            errors.append("per_device_train_batch_size must be positive")
        
        # Validate learning rate
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Validate epochs
        if self.num_train_epochs <= 0:
            errors.append("num_train_epochs must be positive")
        
        # Validate mixed precision
        if self.mixed_precision not in ["fp16", "bf16", "no"]:
            errors.append("mixed_precision must be 'fp16', 'bf16', or 'no'")
        
        # Validate logging level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging_level.upper() not in valid_levels:
            errors.append(f"logging_level must be one of {valid_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

def load_config(config_path: Union[str, Path]) -> ProductionConfig:
    """Load configuration from file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    config = ProductionConfig.from_dict(config_dict)
    config.validate()
    return config

def save_config(config: ProductionConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    # Determine file format
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def validate_config(config: Union[ProductionConfig, Dict[str, Any]]) -> ProductionConfig:
    """Validate and return configuration"""
    if isinstance(config, dict):
        config = ProductionConfig.from_dict(config)
    
    config.validate()
    return config

def get_default_config() -> ProductionConfig:
    """Get default production configuration"""
    return ProductionConfig()

def load_config_from_env() -> ProductionConfig:
    """Load configuration from environment variables"""
    config = get_default_config()
    
    # Override with environment variables
    env_mapping = {
        'PANTHERA_MODEL_NAME': 'model_name',
        'PANTHERA_BATCH_SIZE': 'per_device_train_batch_size',
        'PANTHERA_LEARNING_RATE': 'learning_rate',
        'PANTHERA_EPOCHS': 'num_train_epochs',
        'PANTHERA_OUTPUT_DIR': 'output_dir',
        'PANTHERA_LOGGING_LEVEL': 'logging_level',
        'PANTHERA_LOG_FILE': 'log_file',
        'PANTHERA_MIXED_PRECISION': 'mixed_precision',
        'PANTHERA_GRADIENT_ACCUMULATION': 'gradient_accumulation_steps',
        'PANTHERA_MAX_GPUS': 'max_gpus',
        'PANTHERA_ENABLE_TPU': 'enable_tpu',
        'PANTHERA_CHECKPOINT_DIR': 'checkpoint_dir',
    }
    
    config_dict = config.to_dict()
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Type conversion
            if config_key in ['per_device_train_batch_size', 'num_train_epochs', 'gradient_accumulation_steps', 'max_gpus']:
                value = int(value)
            elif config_key in ['learning_rate']:
                value = float(value)
            elif config_key in ['enable_tpu']:
                value = value.lower() in ('true', '1', 'yes', 'on')
            
            config_dict[config_key] = value
    
    config = ProductionConfig.from_dict(config_dict)
    config.validate()
    return config
