#!/usr/bin/env python3
"""
PantheraML Zoo - Production Training Example
===========================================

This example demonstrates how to use PantheraML Zoo's production features:
- Multi-GPU/TPU distributed training
- Production logging with structured output
- Error handling with automatic checkpointing
- Performance monitoring and metrics tracking
- Configuration management

Usage:
    # Single GPU
    python production_training_example.py

    # Multi-GPU
    torchrun --nproc_per_node=4 production_training_example.py

    # With custom config
    PANTHERAML_LOG_LEVEL=DEBUG python production_training_example.py
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# PantheraML Zoo imports
from unsloth_zoo import (
    # Device management
    setup_distributed,
    get_device_manager,
    
    # Production modules
    setup_production_logging,
    get_logger,
    load_config,
    ErrorHandler,
    get_performance_monitor,
    
    # Training utilities
    unsloth_train,
    prepare_model_for_training,
)

def create_dummy_dataset(tokenizer, num_samples=1000):
    """Create a dummy dataset for demonstration"""
    texts = [
        f"This is training example {i}. PantheraML makes training fast and efficient!" 
        for i in range(num_samples)
    ]
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return dataset

def main():
    # 1. Setup production logging
    setup_production_logging(
        level=os.getenv("PANTHERAML_LOG_LEVEL", "INFO"),
        format="json" if os.getenv("PANTHERAML_JSON_LOGS") else "text"
    )
    logger = get_logger(__name__)
    
    # 2. Load configuration
    config = load_config()
    logger.info("Loaded production configuration", extra={"config": config.to_dict()})
    
    # 3. Setup distributed training
    device_manager = setup_distributed()
    logger.info(
        "Distributed training setup complete",
        extra={
            "world_size": device_manager.world_size,
            "rank": device_manager.rank,
            "device": str(device_manager.device),
            "is_tpu": device_manager.is_tpu,
        }
    )
    
    # 4. Setup error handling
    error_handler = ErrorHandler(
        logger=logger,
        config=config,
        checkpoint_dir="./production_checkpoints"
    )
    
    # 5. Setup performance monitoring
    performance_monitor = get_performance_monitor()
    
    # Main training wrapped in error handling
    with error_handler.context():
        with performance_monitor.training_context():
            
            # Load model and tokenizer (use small model for demo)
            model_name = "microsoft/DialoGPT-small"  # Small model for quick demo
            
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not device_manager.is_distributed else None
            )
            
            # Move model to device for distributed training
            if device_manager.is_distributed:
                model = device_manager.to_device(model)
            
            # Prepare model for training with PantheraML optimizations
            model = prepare_model_for_training(
                model,
                use_gradient_checkpointing=True,
                use_reentrant=True,
            )
            
            # Create dataset
            logger.info("Creating training dataset")
            train_dataset = create_dummy_dataset(tokenizer, num_samples=500)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir="./production_output",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                logging_steps=10,
                save_steps=50,
                evaluation_strategy="no",
                save_strategy="steps",
                warmup_steps=10,
                bf16=True,
                dataloader_num_workers=0,
                report_to=None,  # Disable wandb for demo
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Train using PantheraML's optimized trainer
            logger.info("Starting PantheraML training")
            start_time = performance_monitor.start_timer()
            
            stats = unsloth_train(trainer)
            
            training_time = performance_monitor.stop_timer(start_time)
            
            # Log results
            if device_manager.is_main_process:
                metrics = performance_monitor.get_summary()
                logger.info(
                    "Training completed successfully",
                    extra={
                        "training_time": training_time,
                        "final_metrics": stats.metrics,
                        "performance_metrics": metrics,
                    }
                )
                
                print("\n" + "="*60)
                print("🐾 PantheraML Production Training Complete!")
                print("="*60)
                print(f"Training Time: {training_time:.2f}s")
                print(f"Final Loss: {stats.metrics.get('final_loss', 'N/A')}")
                print(f"Performance Metrics: {metrics}")
                print("="*60)

if __name__ == "__main__":
    main()
