# PantheraML Multi-GPU and TPU Training Example
# This script demonstrates how to use PantheraML with multiple GPUs and TPUs

import torch
from transformers import TrainingArguments
from unsloth_zoo.training_utils import unsloth_train
from unsloth_zoo.device_utils import setup_distributed, get_device_manager

def main():
    # Setup distributed training
    device_manager = setup_distributed()
    
    print(f"Device: {device_manager.device}")
    print(f"World Size: {device_manager.world_size}")
    print(f"Rank: {device_manager.rank}")
    print(f"Is TPU: {device_manager.is_tpu}")
    print(f"Is Distributed: {device_manager.is_distributed}")
    print(f"Is Main Process: {device_manager.is_main_process}")
    
    # Example training arguments for multi-GPU/TPU
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        
        # Multi-GPU specific settings
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        
        # For distributed training
        local_rank=-1,  # Will be set automatically by torchrun
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        ddp_broadcast_buffers=False,
        
        # For TPU
        tpu_num_cores=8 if device_manager.is_tpu else None,
    )
    
    # The training will now automatically work with:
    # - Single GPU
    # - Multiple GPUs (using DDP)
    # - TPU (single or multiple cores)
    # - CPU (fallback)
    
    # Example usage with trainer
    # trainer = create_your_trainer(model, training_args, train_dataset, etc.)
    # unsloth_train(trainer)

if __name__ == "__main__":
    main()

# To run with multiple GPUs:
# torchrun --nproc_per_node=4 multi_gpu_example.py

# To run with TPU:
# python multi_gpu_example.py  # TPU will be detected automatically

# To run with single GPU:
# python multi_gpu_example.py
