# PantheraML Zoo - Performance Monitoring
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

import time
import torch
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .production_logging import get_logger

__all__ = [
    "PerformanceMonitor",
    "TrainingMetrics",
    "memory_monitor",
    "gpu_monitor",
    "performance_profiler",
]

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    throughput_samples_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    cpu_memory_used_gb: float = 0.0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

class PerformanceMonitor:
    """Production performance monitoring for training"""
    
    def __init__(self, device_manager, log_interval: int = 100):
        self.device_manager = device_manager
        self.log_interval = log_interval
        self.logger = get_logger("performance")
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _background_monitor(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                self.logger.warning(f"Error in background monitoring: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        if not self.device_manager.is_main_process:
            return
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        cpu_memory_gb = memory.used / (1024**3)
        
        # GPU metrics
        gpu_memory_used_gb = 0.0
        gpu_memory_total_gb = 0.0
        
        if torch.cuda.is_available() and not self.device_manager.is_tpu:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_used_gb = gpu_memory_used
                gpu_memory_total_gb = gpu_memory_total
            except Exception as e:
                self.logger.warning(f"Failed to get GPU memory info: {e}")
        
        # Log if memory usage is high
        if gpu_memory_used_gb / max(gpu_memory_total_gb, 1) > 0.9:
            self.logger.warning(f"High GPU memory usage: {gpu_memory_used_gb:.1f}GB / {gpu_memory_total_gb:.1f}GB")
        
        if cpu_percent > 90:
            self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
    
    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        sequence_length: Optional[int] = None
    ):
        """Log training step metrics"""
        if not self.device_manager.is_main_process:
            return
        
        current_time = time.time()
        
        # Calculate throughput
        elapsed_time = current_time - self.last_log_time
        if elapsed_time > 0:
            samples_per_sec = (batch_size * self.log_interval) / elapsed_time
            tokens_per_sec = samples_per_sec * (sequence_length or 1)
        else:
            samples_per_sec = tokens_per_sec = 0.0
        
        # Collect metrics
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            throughput_samples_per_sec=samples_per_sec,
            throughput_tokens_per_sec=tokens_per_sec,
            timestamp=current_time
        )
        
        # Add system metrics
        if torch.cuda.is_available() and not self.device_manager.is_tpu:
            try:
                metrics.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
        
        memory = psutil.virtual_memory()
        metrics.cpu_memory_used_gb = memory.used / (1024**3)
        metrics.cpu_percent = psutil.cpu_percent()
        
        self.metrics_history.append(metrics)
        
        # Log every log_interval steps
        if step % self.log_interval == 0:
            self._log_performance_summary(metrics)
            self.last_log_time = current_time
    
    def _log_performance_summary(self, metrics: TrainingMetrics):
        """Log performance summary"""
        total_time = time.time() - self.start_time
        
        self.logger.info(
            f"Step {metrics.step} | "
            f"Loss: {metrics.loss:.4f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Throughput: {metrics.throughput_samples_per_sec:.1f} samples/s"
        )
        
        if torch.cuda.is_available() and not self.device_manager.is_tpu:
            self.logger.info(
                f"GPU Memory: {metrics.gpu_memory_used_gb:.1f}GB / {metrics.gpu_memory_total_gb:.1f}GB "
                f"({metrics.gpu_memory_used_gb/max(metrics.gpu_memory_total_gb, 1)*100:.1f}%)"
            )
        
        self.logger.info(
            f"System: CPU {metrics.cpu_percent:.1f}% | "
            f"RAM {metrics.cpu_memory_used_gb:.1f}GB | "
            f"Runtime: {total_time/3600:.1f}h"
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        return {
            "total_steps": len(self.metrics_history),
            "avg_loss": sum(m.loss for m in recent_metrics) / len(recent_metrics),
            "avg_throughput": sum(m.throughput_samples_per_sec for m in recent_metrics) / len(recent_metrics),
            "max_gpu_memory": max(m.gpu_memory_used_gb for m in self.metrics_history),
            "total_runtime_hours": (time.time() - self.start_time) / 3600,
        }

def memory_monitor(func):
    """Decorator to monitor memory usage of a function"""
    def wrapper(*args, **kwargs):
        logger = get_logger("memory_monitor")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            logger.info(
                f"Function {func.__name__}: "
                f"Memory change: {(final_memory - initial_memory) / (1024**3):.2f}GB | "
                f"Peak: {peak_memory / (1024**3):.2f}GB"
            )
        
        return result
    return wrapper

def gpu_monitor():
    """Get current GPU status"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "device_name": torch.cuda.get_device_name(),
    }

def performance_profiler(enabled: bool = True):
    """Decorator for performance profiling"""
    def decorator(func):
        if not enabled:
            return func
        
        def wrapper(*args, **kwargs):
            logger = get_logger("profiler")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Function {func.__name__} executed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator
