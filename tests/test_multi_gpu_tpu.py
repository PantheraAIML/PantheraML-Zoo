# Tests for Multi-GPU and TPU Support
import pytest
import torch
import os
from unittest.mock import patch, MagicMock

from unsloth_zoo.device_utils import (
    DeviceManager, get_device_manager, is_tpu_available,
    is_distributed, get_device, get_world_size, get_rank,
    barrier, all_reduce_tensor, setup_distributed
)

class TestDeviceManager:
    
    def test_device_manager_cuda_single_gpu(self):
        """Test device manager with single CUDA GPU"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.distributed.is_available', return_value=False):
            
            dm = DeviceManager()
            assert dm.world_size == 1
            assert dm.rank == 0
            assert not dm.is_tpu
            assert not dm.is_distributed
            assert dm.is_main_process
    
    def test_device_manager_tpu(self):
        """Test device manager with TPU"""
        mock_xm = MagicMock()
        mock_xm.xla_device.return_value = "xla:0"
        mock_xm.xrt_world_size.return_value = 8
        mock_xm.get_ordinal.return_value = 0
        
        with patch.dict('sys.modules', {'torch_xla.core.xla_model': mock_xm}):
            dm = DeviceManager()
            assert dm.world_size == 8
            assert dm.rank == 0
            assert dm.is_tpu
            assert dm.device == "xla:0"
            assert dm.is_main_process
    
    def test_device_manager_distributed_cuda(self):
        """Test device manager with distributed CUDA"""
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=4), \
             patch('torch.distributed.get_rank', return_value=1), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=1):
            
            dm = DeviceManager()
            assert dm.world_size == 4
            assert dm.rank == 1
            assert not dm.is_tpu
            assert dm.is_distributed
            assert not dm.is_main_process
    
    def test_device_manager_cpu_fallback(self):
        """Test device manager with CPU fallback"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.distributed.is_available', return_value=False):
            
            dm = DeviceManager()
            assert dm.world_size == 1
            assert dm.rank == 0
            assert not dm.is_tpu
            assert not dm.is_distributed
            assert dm.device == torch.device("cpu")
    
    def test_barrier_cuda(self):
        """Test barrier for CUDA distributed training"""
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.barrier') as mock_barrier:
            
            dm = DeviceManager()
            dm.barrier()
            mock_barrier.assert_called_once()
    
    def test_barrier_tpu(self):
        """Test barrier for TPU training"""
        mock_xm = MagicMock()
        
        with patch.dict('sys.modules', {'torch_xla.core.xla_model': mock_xm}):
            dm = DeviceManager()
            dm.barrier()
            mock_xm.rendezvous.assert_called_with("barrier")
    
    def test_all_reduce_cuda(self):
        """Test all_reduce for CUDA distributed training"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2), \
             patch('torch.distributed.all_reduce') as mock_all_reduce:
            
            dm = DeviceManager()
            result = dm.all_reduce(tensor)
            mock_all_reduce.assert_called_once()
            assert torch.equal(result, tensor / 2)  # Divided by world_size
    
    def test_all_reduce_tpu(self):
        """Test all_reduce for TPU training"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_xm = MagicMock()
        mock_xm.xrt_world_size.return_value = 8
        mock_xm.all_reduce.return_value = tensor * 8  # Simulated sum across 8 cores
        
        with patch.dict('sys.modules', {'torch_xla.core.xla_model': mock_xm}):
            dm = DeviceManager()
            result = dm.all_reduce(tensor)
            mock_xm.all_reduce.assert_called_with(mock_xm.REDUCE_SUM, tensor)
            assert torch.equal(result, tensor)  # Should be normalized by world_size
    
    def test_to_device_cuda(self):
        """Test moving tensor to CUDA device"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        with patch('torch.cuda.is_available', return_value=True):
            dm = DeviceManager()
            # Mock tensor.to method
            with patch.object(tensor, 'to', return_value=tensor) as mock_to:
                result = dm.to_device(tensor)
                mock_to.assert_called_with(device=dm.device, non_blocking=True)
    
    def test_to_device_tpu(self):
        """Test moving tensor to TPU device"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_xm = MagicMock()
        mock_xm.xla_device.return_value = "xla:0"
        
        with patch.dict('sys.modules', {'torch_xla.core.xla_model': mock_xm}):
            dm = DeviceManager()
            with patch.object(tensor, 'to', return_value=tensor) as mock_to:
                result = dm.to_device(tensor)
                mock_to.assert_called_with("xla:0")
    
    def test_mark_step_tpu(self):
        """Test mark_step for TPU training"""
        mock_xm = MagicMock()
        
        with patch.dict('sys.modules', {'torch_xla.core.xla_model': mock_xm}):
            dm = DeviceManager()
            dm.mark_step()
            mock_xm.mark_step.assert_called_once()
    
    def test_mark_step_cuda(self):
        """Test mark_step for CUDA (should be no-op)"""
        with patch('torch.cuda.is_available', return_value=True):
            dm = DeviceManager()
            # Should not raise any errors
            dm.mark_step()

class TestGlobalFunctions:
    
    def test_is_tpu_available_true(self):
        """Test TPU availability detection when TPU is available"""
        with patch.dict('sys.modules', {'torch_xla.core.xla_model': MagicMock()}):
            assert is_tpu_available() == True
    
    def test_is_tpu_available_false(self):
        """Test TPU availability detection when TPU is not available"""
        # Clear any existing device manager
        import unsloth_zoo.device_utils
        unsloth_zoo.device_utils._device_manager = None
        
        with patch('importlib.import_module', side_effect=ImportError):
            assert is_tpu_available() == False
    
    def test_setup_distributed_with_env_vars(self):
        """Test setup_distributed with environment variables"""
        with patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '4'}), \
             patch('torch.distributed.is_initialized', return_value=False), \
             patch('torch.distributed.init_process_group') as mock_init:
            
            dm = setup_distributed()
            mock_init.assert_called_once()
    
    def test_setup_distributed_already_initialized(self):
        """Test setup_distributed when already initialized"""
        with patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '4'}), \
             patch('torch.distributed.is_initialized', return_value=True):
            
            dm = setup_distributed()
            # Should not raise any errors

if __name__ == "__main__":
    pytest.main([__file__])
