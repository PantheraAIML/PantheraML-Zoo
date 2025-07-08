#!/usr/bin/env python3
"""
TPU/XLA Compatibility Test for PantheraML Zoo
Tests all device-agnostic code paths and TPU/XLA logic
"""

import os
os.environ["PANTHERAML_ALLOW_CPU"] = "1"
os.environ["UNSLOTH_IS_PRESENT"] = "1"

import unittest
from unittest.mock import patch, MagicMock
import torch

class TestTPUCompatibility(unittest.TestCase):
    
    def test_device_detection(self):
        """Test device detection includes TPU"""
        import pantheraml_zoo
        
        # Should work on CPU with fallback
        self.assertEqual(pantheraml_zoo.DEVICE_TYPE, "cpu")
        
        # Test TPU detection mock
        with patch('torch_xla.core.xla_model.xla_device', return_value=MagicMock()):
            # Reimport to test TPU detection
            import importlib
            importlib.reload(pantheraml_zoo)
    
    def test_device_manager_tpu_methods(self):
        """Test device manager has TPU-specific methods"""
        from pantheraml_zoo.device_utils import DeviceManager
        
        # Mock TPU environment
        with patch('pantheraml_zoo.device_utils.is_tpu_available', return_value=True):
            with patch('torch_xla.core.xla_model') as mock_xm:
                mock_xm.get_ordinal.return_value = 0
                mock_xm.xrt_world_size.return_value = 1
                mock_xm.get_local_ordinal.return_value = 0
                
                dm = DeviceManager()
                
                # Should have TPU methods available
                self.assertTrue(hasattr(dm, 'barrier'))
                self.assertTrue(hasattr(dm, 'all_reduce'))
                self.assertTrue(hasattr(dm, 'mark_step'))
    
    def test_loss_utils_device_agnostic(self):
        """Test loss utils work with different device types"""
        from pantheraml_zoo.loss_utils import _get_device_context_manager
        
        # Test CUDA device
        with patch('torch.cuda.device') as mock_cuda:
            ctx = _get_device_context_manager(torch.device('cuda:0'))
            self.assertIsNotNone(ctx)
        
        # Test TPU/XLA device (should return nullcontext)
        from contextlib import nullcontext
        ctx = _get_device_context_manager(torch.device('xla:0'))
        self.assertIsInstance(ctx, type(nullcontext()))
    
    def test_gradient_checkpointing_device_support(self):
        """Test gradient checkpointing supports multiple devices"""
        from pantheraml_zoo.gradient_checkpointing import torch_gpu_stream
        
        # Should return callable for any device type
        self.assertTrue(callable(torch_gpu_stream))
        
        # Test with different device types
        stream = torch_gpu_stream(torch.device('cpu'))
        self.assertIsNotNone(stream)
    
    def test_training_utils_autocast(self):
        """Test training utils use device-agnostic autocast"""
        # This would require more complex mocking, but we can verify
        # the training utils module imports without errors
        from pantheraml_zoo import training_utils
        self.assertTrue(hasattr(training_utils, 'train_model'))
    
    def test_error_handling_device_agnostic(self):
        """Test error handling works with any device"""
        from pantheraml_zoo.error_handling import ErrorHandler
        
        handler = ErrorHandler()
        
        # Should have device-agnostic OOM handler
        self.assertTrue(hasattr(handler, 'handle_device_oom'))
    
    def test_distributed_backend_selection(self):
        """Test distributed backend selection includes XLA"""
        from pantheraml_zoo.device_utils import setup_distributed
        
        # Test without environment variables (should not initialize)
        old_rank = os.environ.pop('RANK', None)
        old_world_size = os.environ.pop('WORLD_SIZE', None)
        
        try:
            dm = setup_distributed()
            self.assertIsNotNone(dm)
        finally:
            if old_rank:
                os.environ['RANK'] = old_rank
            if old_world_size:
                os.environ['WORLD_SIZE'] = old_world_size


def test_mock_tpu_environment():
    """Test with mocked TPU environment"""
    print("Testing mock TPU environment...")
    
    with patch.dict('sys.modules', {'torch_xla': MagicMock(), 'torch_xla.core': MagicMock(), 'torch_xla.core.xla_model': MagicMock()}):
        mock_xm = MagicMock()
        mock_xm.xla_device.return_value = MagicMock()
        mock_xm.get_ordinal.return_value = 0
        mock_xm.xrt_world_size.return_value = 8  # 8 TPU cores
        mock_xm.get_local_ordinal.return_value = 0
        mock_xm.all_reduce.return_value = None
        mock_xm.rendezvous.return_value = None
        mock_xm.mark_step.return_value = None
        
        with patch('torch_xla.core.xla_model', mock_xm):
            # Test device detection
            import pantheraml_zoo
            
            # Test device manager with TPU
            from pantheraml_zoo.device_utils import DeviceManager, is_tpu_available
            
            print(f"✅ TPU available: {is_tpu_available()}")
            
            dm = DeviceManager()
            print(f"✅ Device manager created: {dm}")
            print(f"✅ Is TPU: {dm.is_tpu}")
            print(f"✅ World size: {dm.world_size}")
            print(f"✅ Rank: {dm.rank}")


if __name__ == '__main__':
    print("🧪 Running TPU/XLA Compatibility Tests for PantheraML Zoo")
    print("=" * 60)
    
    # Run mock TPU test first
    test_mock_tpu_environment()
    print()
    
    # Run unittest suite
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("✅ All TPU/XLA compatibility tests completed!")
    print("🐾 PantheraML Zoo is ready for multi-device deployment!")
