# PantheraML Zoo - Error Handling and Recovery
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

import torch
import functools
import traceback
import time
from typing import Any, Callable, Optional, Dict
from .production_logging import get_logger

__all__ = [
    "ErrorHandler",
    "retry_on_failure",
    "checkpoint_training",
    "recover_from_checkpoint",
    "safe_device_operation",
]

class ErrorHandler:
    """Production error handling for PantheraML training"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = get_logger("error_handler")
    
    def handle_cuda_oom(self, func: Callable, *args, **kwargs):
        """Handle CUDA out of memory errors"""
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning("CUDA OOM detected, attempting recovery...")
                torch.cuda.empty_cache()
                
                # Try with reduced batch size if possible
                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                    self.logger.info(f"Retrying with reduced batch size: {kwargs['batch_size']}")
                    return func(*args, **kwargs)
            raise
    
    def handle_distributed_errors(self, func: Callable, *args, **kwargs):
        """Handle distributed training errors"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["nccl", "distributed", "timeout"]):
                self.logger.error(f"Distributed training error: {e}")
                self.logger.info("Attempting to reinitialize distributed backend...")
                
                # Reinitialize distributed if possible
                try:
                    if torch.distributed.is_initialized():
                        torch.distributed.destroy_process_group()
                    from .device_utils import setup_distributed
                    setup_distributed()
                    return func(*args, **kwargs)
                except Exception as retry_e:
                    self.logger.error(f"Failed to recover from distributed error: {retry_e}")
            raise

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Decorator for retrying operations on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("retry")
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def checkpoint_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    step: int,
    epoch: int,
    filepath: str
) -> None:
    """Save training checkpoint"""
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'step': step,
            'epoch': epoch,
            'torch_version': torch.__version__,
        }
        torch.save(checkpoint, filepath)
        get_logger("checkpoint").info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        get_logger("checkpoint").error(f"Failed to save checkpoint: {e}")

def recover_from_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    filepath: str
) -> Dict[str, Any]:
    """Recover training from checkpoint"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        get_logger("checkpoint").info(f"Recovered from checkpoint at step {checkpoint['step']}, epoch {checkpoint['epoch']}")
        
        return {
            'step': checkpoint['step'],
            'epoch': checkpoint['epoch'],
            'torch_version': checkpoint.get('torch_version', 'unknown')
        }
    except Exception as e:
        get_logger("checkpoint").error(f"Failed to recover from checkpoint: {e}")
        return {'step': 0, 'epoch': 0}

def safe_device_operation(func: Callable) -> Callable:
    """Decorator for safe device operations with error handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            logger = get_logger("device_ops")
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM in device operation, clearing cache...")
                torch.cuda.empty_cache()
                raise
            elif "device" in str(e).lower():
                logger.error(f"Device error in operation: {e}")
                raise
            else:
                raise
        except Exception as e:
            logger = get_logger("device_ops")
            logger.error(f"Unexpected error in device operation: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    return wrapper
