# PantheraML Zoo - Production Logging System
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

import logging
import os
import sys
from typing import Optional

__all__ = [
    "get_logger",
    "setup_logging",
    "log_device_info",
    "log_training_config",
]

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1
) -> logging.Logger:
    """Setup production-ready logging configuration"""
    
    # Only main process logs to avoid spam
    if rank != 0:
        logging.disable(logging.CRITICAL)
        return logging.getLogger("panthera_ml")
    
    logger = logging.getLogger("panthera_ml")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - PantheraML - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)d/%(world_size)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "panthera_ml") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

def log_device_info(device_manager, logger: Optional[logging.Logger] = None):
    """Log device and distributed training information"""
    if logger is None:
        logger = get_logger()
    
    if device_manager.is_main_process:
        logger.info(f"Device Configuration:")
        logger.info(f"  Device Type: {'TPU' if device_manager.is_tpu else 'CUDA' if 'cuda' in str(device_manager.device) else 'CPU'}")
        logger.info(f"  Device: {device_manager.device}")
        logger.info(f"  World Size: {device_manager.world_size}")
        logger.info(f"  Current Rank: {device_manager.rank}")
        logger.info(f"  Distributed: {device_manager.is_distributed}")
        
        if device_manager.is_tpu:
            logger.info("  TPU-optimized training enabled")
        elif device_manager.world_size > 1:
            logger.info("  Multi-GPU distributed training enabled")

def log_training_config(training_args, model_info: dict, logger: Optional[logging.Logger] = None):
    """Log training configuration"""
    if logger is None:
        logger = get_logger()
    
    logger.info("Training Configuration:")
    logger.info(f"  Model: {model_info.get('name', 'Unknown')}")
    logger.info(f"  Parameters: {model_info.get('parameters', 'Unknown'):,}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch Size per Device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning Rate: {training_args.learning_rate}")
    logger.info(f"  Warmup Steps: {training_args.warmup_steps}")
    logger.info(f"  Weight Decay: {training_args.weight_decay}")
    logger.info(f"  Mixed Precision: {getattr(training_args, 'fp16', False) or getattr(training_args, 'bf16', False)}")
