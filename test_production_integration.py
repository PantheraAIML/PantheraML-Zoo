#!/usr/bin/env python3
"""
PantheraML Zoo - Production Integration Test
===========================================

This test script validates that all production components work together correctly:
- Device management and distributed training
- Production logging system
- Error handling and recovery
- Performance monitoring
- Configuration management

Run this to ensure your PantheraML Zoo installation is production-ready.
"""

import os
import sys
import tempfile
import shutil
import torch
from pathlib import Path
import json
import time

def test_imports():
    """Test that all production modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Core imports
        from pantheraml_zoo import (
            setup_distributed,
            get_device_manager,
            setup_production_logging,
            get_logger,
            load_config,
            ProductionConfig,
            ErrorHandler,
            get_performance_monitor,
        )
        print("✅ All core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_device_management():
    """Test device management and distributed setup"""
    print("\n🧪 Testing device management...")
    
    try:
        from pantheraml_zoo import setup_distributed, get_device_manager
        
        # Setup distributed (should work even with single device)
        device_manager = setup_distributed()
        
        # Test device manager properties
        assert hasattr(device_manager, 'device')
        assert hasattr(device_manager, 'world_size')
        assert hasattr(device_manager, 'rank')
        assert hasattr(device_manager, 'is_main_process')
        assert hasattr(device_manager, 'is_tpu')
        
        # Test tensor operations
        test_tensor = torch.randn(2, 2)
        device_tensor = device_manager.to_device(test_tensor)
        assert device_tensor.device == device_manager.device
        
        print(f"✅ Device management test passed")
        print(f"   Device: {device_manager.device}")
        print(f"   World size: {device_manager.world_size}")
        print(f"   Rank: {device_manager.rank}")
        print(f"   Is TPU: {device_manager.is_tpu}")
        
        return True
    except Exception as e:
        print(f"❌ Device management test failed: {e}")
        return False

def test_production_logging():
    """Test production logging system"""
    print("\n🧪 Testing production logging...")
    
    try:
        from pantheraml_zoo import setup_production_logging, get_logger
        
        # Test different log formats
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Test text format
            setup_production_logging(
                level="DEBUG",
                format="text",
                log_file=str(log_file)
            )
            
            logger = get_logger("test_logger")
            logger.info("Test message", extra={"test_key": "test_value"})
            
            # Test JSON format
            setup_production_logging(
                level="DEBUG", 
                format="json",
                log_file=str(log_file)
            )
            
            logger = get_logger("test_logger_json")
            logger.info("JSON test message", extra={"json_key": "json_value"})
            
            # Verify log file exists and has content
            assert log_file.exists()
            assert log_file.stat().st_size > 0
            
        print("✅ Production logging test passed")
        return True
    except Exception as e:
        print(f"❌ Production logging test failed: {e}")
        return False

def test_configuration():
    """Test configuration management"""
    print("\n🧪 Testing configuration management...")
    
    try:
        from pantheraml_zoo import ProductionConfig, load_config
        
        # Test default config
        config = ProductionConfig()
        assert hasattr(config, 'max_sequence_length')
        assert hasattr(config, 'enable_checkpointing')
        assert hasattr(config, 'enable_performance_monitoring')
        
        # Test config serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        
        # Test loading from environment
        os.environ["PANTHERAML_MAX_SEQUENCE_LENGTH"] = "2048"
        os.environ["PANTHERAML_ENABLE_CHECKPOINTING"] = "false"
        
        env_config = load_config()
        assert env_config.max_sequence_length == 2048
        assert env_config.enable_checkpointing == False
        
        # Clean up environment
        os.environ.pop("PANTHERAML_MAX_SEQUENCE_LENGTH", None)
        os.environ.pop("PANTHERAML_ENABLE_CHECKPOINTING", None)
        
        print("✅ Configuration management test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery"""
    print("\n🧪 Testing error handling...")
    
    try:
        from pantheraml_zoo import ErrorHandler, get_logger, ProductionConfig
        
        logger = get_logger("test_error_handler")
        config = ProductionConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            error_handler = ErrorHandler(
                logger=logger,
                config=config,
                checkpoint_dir=temp_dir
            )
            
            # Test context manager
            with error_handler.context():
                # Simulate some work
                time.sleep(0.1)
            
            # Test decorator
            @error_handler.with_error_handling
            def test_function():
                return "success"
            
            result = test_function()
            assert result == "success"
            
        print("✅ Error handling test passed")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring"""
    print("\n🧪 Testing performance monitoring...")
    
    try:
        from pantheraml_zoo import get_performance_monitor
        
        monitor = get_performance_monitor()
        
        # Test training context
        with monitor.training_context():
            # Simulate training work
            time.sleep(0.1)
            
            # Track some metrics
            monitor.track_step_metrics({
                'step': 1,
                'loss': 0.5,
                'learning_rate': 1e-4
            })
        
        # Get summary
        summary = monitor.get_summary()
        assert isinstance(summary, dict)
        
        # Test timer functions
        timer_id = monitor.start_timer()
        time.sleep(0.05)
        elapsed = monitor.stop_timer(timer_id)
        assert elapsed > 0.04  # Should be at least 50ms
        
        print("✅ Performance monitoring test passed")
        return True
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def test_integration():
    """Test all components working together"""
    print("\n🧪 Testing full integration...")
    
    try:
        from pantheraml_zoo import (
            setup_distributed,
            setup_production_logging,
            get_logger,
            load_config,
            ErrorHandler,
            get_performance_monitor
        )
        
        # Setup all components
        setup_production_logging(level="INFO", format="text")
        logger = get_logger("integration_test")
        config = load_config()
        device_manager = setup_distributed()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            error_handler = ErrorHandler(
                logger=logger,
                config=config,
                checkpoint_dir=temp_dir
            )
            performance_monitor = get_performance_monitor()
            
            # Test everything working together
            with error_handler.context():
                with performance_monitor.training_context():
                    logger.info("Integration test in progress")
                    
                    # Simulate some work
                    test_tensor = torch.randn(10, 10)
                    device_tensor = device_manager.to_device(test_tensor)
                    
                    # Track metrics
                    performance_monitor.track_step_metrics({
                        'integration_test': True,
                        'tensor_shape': list(device_tensor.shape)
                    })
                    
                    time.sleep(0.1)
            
            metrics = performance_monitor.get_summary()
            logger.info("Integration test completed", extra={"metrics": metrics})
        
        print("✅ Full integration test passed")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🐾 PantheraML Zoo - Production Integration Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_device_management,
        test_production_logging,
        test_configuration,
        test_error_handling,
        test_performance_monitoring,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("🏁 Test Results Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! PantheraML Zoo is production-ready!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
