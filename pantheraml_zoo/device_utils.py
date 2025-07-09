# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.distributed as dist
import os
from typing import Union, Optional

__all__ = [
    "DeviceManager",
    "get_device_manager",
    "is_tpu_available",
    "is_distributed",
    "get_device",
    "get_world_size", 
    "get_rank",
    "barrier",
    "all_reduce_tensor",
    "setup_distributed",
]

class DeviceManager:
    """Manages device detection and distributed training across CUDA, TPU, and CPU"""
    
    def __init__(self):
        self._device = None
        self._world_size = None
        self._rank = None
        self._is_tpu = None
        self._is_distributed = None
        self._detect_environment()
    
    def _detect_environment(self):
        """Detect the current training environment"""
        # Check for TPU/XLA
        try:
            import torch_xla.core.xla_model as xm
            self._is_tpu = True
            self._device = xm.xla_device()  # This returns the proper XLA device string
            self._world_size = xm.xrt_world_size()
            self._rank = xm.get_ordinal()
            self._is_distributed = self._world_size > 1
        except ImportError:
            self._is_tpu = False
            
            # Check for distributed CUDA
            if dist.is_available() and dist.is_initialized():
                self._is_distributed = True
                self._world_size = dist.get_world_size()
                self._rank = dist.get_rank()
                if torch.cuda.is_available():
                    self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    self._device = torch.device(f"xpu:{torch.xpu.current_device()}")
                else:
                    self._device = torch.device("cpu")
            else:
                self._is_distributed = False
                self._world_size = 1
                self._rank = 0
                if torch.cuda.is_available():
                    self._device = torch.device("cuda:0")
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    self._device = torch.device("xpu:0")
                else:
                    self._device = torch.device("cpu")
    
    @property
    def device(self) -> Union[torch.device, str]:
        """Get the current device"""
        return self._device
    
    @property
    def world_size(self) -> int:
        """Get the world size"""
        return self._world_size
    
    @property
    def rank(self) -> int:
        """Get the current rank"""
        return self._rank
    
    @property
    def is_tpu(self) -> bool:
        """Check if running on TPU"""
        return self._is_tpu
    
    @property
    def is_distributed(self) -> bool:
        """Check if in distributed training mode"""
        return self._is_distributed or self._is_tpu
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self._rank == 0
    
    @property
    def device_type(self) -> str:
        """Get the device type string"""
        if self._is_tpu:
            return "xla"  # TPUs use XLA device type
        elif isinstance(self._device, torch.device):
            return self._device.type
        else:
            return str(self._device).split(":")[0]  # Handle string device representations
    
    def barrier(self):
        """Synchronization barrier"""
        if self._is_tpu:
            import torch_xla.core.xla_model as xm
            xm.rendezvous("barrier")
        elif self._is_distributed:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor across all processes"""
        if self._is_tpu:
            import torch_xla.core.xla_model as xm
            return xm.all_reduce(xm.REDUCE_SUM, tensor) / self._world_size
        elif self._is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor / self._world_size
        else:
            return tensor
    
    def mark_step(self):
        """Mark step for TPU training"""
        if self._is_tpu:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
    
    def to_device(self, tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
        """Move tensor to the appropriate device"""
        if isinstance(tensor, torch.Tensor):
            if self._is_tpu:
                return tensor.to(self._device)
            else:
                return tensor.to(device=self._device, non_blocking=non_blocking)
        return tensor

# Global device manager instance
_device_manager: Optional[DeviceManager] = None

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager

def is_tpu_available() -> bool:
    """Check if TPU is available"""
    return get_device_manager().is_tpu

def is_distributed() -> bool:
    """Check if we're in a distributed training environment"""
    return get_device_manager().is_distributed

def get_device() -> Union[torch.device, str]:
    """Get the appropriate device for training"""
    return get_device_manager().device

def get_world_size() -> int:
    """Get the world size for distributed training"""
    return get_device_manager().world_size

def get_rank() -> int:
    """Get the rank for distributed training"""
    return get_device_manager().rank

def barrier():
    """Synchronization barrier for distributed training"""
    get_device_manager().barrier()

def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor across all processes"""
    return get_device_manager().all_reduce(tensor)

def setup_distributed():
    """Setup distributed training environment"""
    # Initialize distributed training if environment variables are set
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            # Choose appropriate backend based on device type
            from . import DEVICE_TYPE
            if DEVICE_TYPE == "cuda":
                backend = "nccl"
            elif DEVICE_TYPE == "tpu":
                backend = "xla"  # XLA backend for TPU
            else:
                backend = "gloo"  # Fallback for CPU/XPU
            
            dist.init_process_group(backend=backend)
    
    # Refresh device manager to pick up any changes
    global _device_manager
    _device_manager = None
    return get_device_manager()
