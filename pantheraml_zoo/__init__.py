# PantheraML Zoo - Production-ready training utilities
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

__version__ = "2025.6.8"

from importlib.util import find_spec
if find_spec("unsloth") is None:
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
pass
del find_spec

def get_device_type():
    import torch
    
    # Check for TPU first
    try:
        import torch_xla.core.xla_model as xm
        # Just check if XLA is available, don't call xla_device() yet
        return "xla"  # Use "xla" instead of "tpu" to match PyTorch device types
    except ImportError:
        pass
    
    # Check for CUDA
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    else:
        # For development on systems without GPU/TPU (like macOS)
        import os
        if os.getenv("PANTHERAML_ALLOW_CPU", "0") == "1":
            return "cpu"
        else:
            raise NotImplementedError("PantheraML currently only works on NVIDIA GPUs, Intel GPUs, and TPUs. Set PANTHERAML_ALLOW_CPU=1 for CPU-only development.")
pass
DEVICE_TYPE : str = get_device_type()

import os
if not ("UNSLOTH_IS_PRESENT" in os.environ):
    raise ImportError("Please install PantheraML! via `pip install pantheraml` or `pip install git+https://github.com/PantheraAIML/PantheraML.git`!")
pass

try:
    print("🐾 PantheraML: Will optimize your training to be 2x faster with production-grade features.")
except:
    print("PantheraML: Will optimize your training to be 2x faster with production-grade features.")
pass
# Log PantheraML-Zoo Utilities
os.environ["PANTHERAML_ZOO_IS_PRESENT"] = "1"
del os

# Multi-GPU and TPU support
from .device_utils import (
    DeviceManager,
    get_device_manager,
    is_tpu_available,
    is_distributed,
    get_device,
    get_world_size,
    get_rank,
    barrier,
    all_reduce_tensor,
    setup_distributed,
)

# Production modules
from .production_logging import get_logger, setup_logging
from .production_config import ProductionConfig, load_config
from .error_handling import (
    ErrorHandler,
    checkpoint_training,
    retry_on_failure,
    safe_device_operation,
)
from .performance_monitoring import (
    PerformanceMonitor,
    TrainingMetrics,
    memory_monitor,
    performance_profiler,
)
