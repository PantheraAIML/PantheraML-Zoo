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
import math
import datasets
from contextlib import nullcontext
from transformers import set_seed as transformers_set_seed
from transformers import get_scheduler as transformers_get_scheduler
from transformers import Trainer
from transformers.trainer_utils import seed_worker as trainer_utils_seed_worker
from tqdm import tqdm as ProgressBar
from packaging.version import Version
import time
from typing import Any, Optional, List, Dict, Tuple, Union
from .utils import _get_dtype
from .device_utils import (
    get_device_manager, setup_distributed
)
# Production modules
from .production_logging import get_logger
from .production_config import load_config
from .error_handling import ErrorHandler, retry_on_failure, safe_device_operation
from .performance_monitoring import PerformanceMonitor, TrainingMetrics
import os
import re
from contextlib import nullcontext

__all__ = [
    "fix_zero_training_loss",
    "unsloth_train", 
    "prepare_model_for_training",
]


@torch.inference_mode
def fix_zero_training_loss(model, tokenizer, train_dataset):
    """
    Sometimes the labels get masked by all -100s, causing the loss
    to be 0. We check for this!
    """
    # All PantheraML Zoo code licensed under LGPLv3
    if isinstance(train_dataset, datasets.IterableDataset):
        # Skip the check since the code below assumes
        # an indexable dataset
        return
    
    if len(train_dataset) == 0: return

    row = train_dataset[0]
    if type(row) is dict and "labels" in row:

        # Check the first 100 rows
        seen_bad  = 0
        seen_good = 0
        for i, row in enumerate(train_dataset):
            try:    check_tokens = list(set(row["labels"]))
            except: continue
            if len(check_tokens) == 1 and check_tokens[0] == -100: seen_bad += 1
            else: seen_good += 1
            if i >= 100: break
        pass

        # Check ratio
        if seen_bad == 0 and seen_good == 0: return

        elif seen_bad / (seen_bad + seen_good) == 1:
            raise ZeroDivisionError(
                "PantheraML: All labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"
            )
        elif seen_bad / (seen_bad + seen_good) >= 0.9:
            print(
                "PantheraML: Nearly all labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"
            )
    pass
pass


@torch.no_grad
def prepare_model_for_training(
    model                      : Any,
    use_gradient_checkpointing : Optional[Union[bool, str]] = "unsloth",
    use_reentrant              : Optional[bool] = True,
    full_finetuning            : Optional[bool] = False,
    train_layernorms           : Optional[bool] = False,
    train_embedding            : Optional[bool] = False,
    train_lm_head              : Optional[bool] = False,
    float32_mixed_precision    : Optional[bool] = True,
) -> Any:
    # All PantheraML Zoo code licensed under LGPLv3
    assert(use_gradient_checkpointing in (True, False, "unsloth",))
    assert(type(use_reentrant) is bool)
    assert(type(full_finetuning) is bool)
    assert(type(train_layernorms) is bool)
    assert(type(train_embedding) is bool)
    assert(type(train_lm_head) is bool)
    assert(type(float32_mixed_precision) is bool)

    dtype = _get_dtype(model.config.torch_dtype)
    mixed_precision_dtype = torch.float32
    if dtype == torch.float16:
        # We need to upcast to float32
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
    elif dtype == torch.bfloat16 and float32_mixed_precision:
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
    elif dtype == torch.bfloat16:
        mixed_precision_dtype = torch.bfloat16
        os.environ["UNSLOTH_MIXED_PRECISION"] = "bfloat16"
    else:
        mixed_precision_dtype = torch.float32
        os.environ["UNSLOTH_MIXED_PRECISION"] = "float32"
    pass
    
    for name, param in model.named_parameters():
        upcast = False
        requires_grad = False
        if not full_finetuning:
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                upcast = True
                requires_grad = True
            else:
                requires_grad = False
        else:
            if train_layernorms and ("norm." in name or "_layernorm" in name):
                requires_grad = True
                upcast = True # Must upcast layernorms to float32
            if train_embedding and ("embed_tokens" in name or "embedding" in name):
                requires_grad = True
                upcast = False # Can leave in bfloat16
            if train_lm_head and ("lm_head" in name):
                requires_grad = True
                upcast = False # Can leave in bfloat16
            else:
                requires_grad = True
                upcast = False # Can leave in bfloat16
        pass
        # Set training or not
        if requires_grad:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

        # Upcast to float32 if needed
        if requires_grad:
            name = name.replace("base_model", "model", 1)
            while re.search(r'\.(\d+)\.', name) is not None:
                name = re.sub(r'\.(\d+)\.', r'[\1].', name)
            name = name.replace(".weight", "", 1)
            dtype = torch.float32 if upcast else mixed_precision_dtype
            try:
                # Try original name
                exec(f"{name}.to({str(dtype)})")
            except:
                # Maybe model.model
                exec(f"model.{name}.to({str(dtype)})")
        pass

        if ('norm.' in name or '_layernorm' in name) and os.environ.get("UNSLOTH_UPCAST_LAYERNORM", "0") == "1":
            try:
                name = name.replace("base_model", "model", 1)
                while re.search(r'\.(\d+)\.', name) is not None:
                    name = re.sub(r'\.(\d+)\.', r'[\1].', name)
                name = name.replace(".weight", "", 1)
                # Try original name
                exec(f"{name}.to({str(torch.float32)})")
            except:
                # Maybe model.model
                exec(f"model.{name}.to({str(torch.float32)})")
    pass

    # Gradient checkpointing
    m = model
    while hasattr(m, "model"):
        if use_gradient_checkpointing == "unsloth":
            m._offloaded_gradient_checkpointing = True
        if use_gradient_checkpointing == True and hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable()
        m = m.model
    pass
    if use_gradient_checkpointing == "unsloth":
        m._offloaded_gradient_checkpointing = True
    if use_gradient_checkpointing == True and hasattr(m, "gradient_checkpointing_enable"):
        m.gradient_checkpointing_enable()

    # Also set HF version manually to stop failures
    if hasattr(model, "_set_gradient_checkpointing"):
        model._set_gradient_checkpointing()

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    pass

    return model
pass


def get_max_steps(training_args, n_training_samples, train_dataset):
    # Approximately from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2092
    # Determines batch size, max steps, ga etc
    device_manager = get_device_manager()
    world_size = device_manager.world_size
    
    bsz = training_args.per_device_train_batch_size
    ga  = training_args.gradient_accumulation_steps

    total_train_batch_size = bsz * ga * world_size
    max_steps = training_args.max_steps

    if max_steps > 0:
        total_samples_seen = total_train_batch_size * max_steps
        num_train_epochs = math.ceil(total_samples_seen / n_training_samples)
    else:
        num_train_epochs = training_args.num_train_epochs
        steps_per_epoch  = math.ceil(n_training_samples / total_train_batch_size)
        max_steps = math.ceil(steps_per_epoch * num_train_epochs)
        num_train_epochs = math.ceil(num_train_epochs)
    return total_train_batch_size, max_steps, num_train_epochs
pass


def set_training(model):
    # Start training
    model.training = True
    while hasattr(model, "model"):
        model = model.model
        model.training = True
    model.training = True
pass


def unset_training(model):
    # End training
    model.training = False
    while hasattr(model, "model"):
        model = model.model
        model.training = False
    model.training = False
pass


from dataclasses import dataclass
@dataclass
class Trainer_Stats:
    metrics: dict
pass


def unsloth_train(trainer):
    """
    PantheraML Trainer
    1. Fixes gradient accumulation
    2. Scaled down version of HF's trainer
    3. Multi-GPU and TPU compatible
    4. Much less feature complete
    5. Production-ready with monitoring and error handling
    """
    # All PantheraML Zoo code licensed under LGPLv3
    assert(hasattr(trainer, "args"))
    assert(hasattr(trainer, "model"))
    assert(hasattr(trainer, "train_dataset"))
    assert(hasattr(trainer, "data_collator"))

    # Setup production modules
    logger = get_logger(__name__)
    config = load_config()
    error_handler = ErrorHandler(logger=logger, config=config)
    performance_monitor = PerformanceMonitor()
    
    with error_handler.context(), performance_monitor.training_context():
        model = trainer.model
        training_args = trainer.args
        data_collator = trainer.data_collator
        n_training_samples = len(trainer.train_dataset)
        set_training(model)
        transformers_set_seed(training_args.seed)

        # Setup distributed training and get device info
        device_manager = setup_distributed()
        device = device_manager.device
        world_size = device_manager.world_size
        rank = device_manager.rank
        is_main_process = device_manager.is_main_process

        # Log training start
        if is_main_process:
            logger.info(f"Starting PantheraML training on {world_size} device(s)")
            logger.info(f"Training samples: {n_training_samples:,}")
            logger.info(f"Device type: {'TPU' if device_manager.is_tpu else 'CUDA' if torch.cuda.is_available() else 'CPU'}")

        if training_args.dataloader_drop_last:
            raise NotImplementedError(
                "PantheraML: Currently `dataloader_drop_last` is not yet implemented!"
            )
        pass

    if data_collator is None:
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = trainer.tokenizer,
            mlm = False,
            pad_to_multiple_of = 4,
        )
    pass

    # Separate weight decay for parameters
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    decay_parameters = frozenset(Trainer.get_decay_parameter_names(None, model))
    yes_decay, no_decay = [], []
    n_parameters_to_train = 0
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name in decay_parameters: yes_decay.append(param)
        else: no_decay.append(param)
        n_parameters_to_train += param.numel()
    pass
    optimizer_grouped_parameters = [
        {"params" : yes_decay, "weight_decay" : training_args.weight_decay,},
        {"params" : no_decay,  "weight_decay" : 0,}
    ]
    trainable_parameters = \
        optimizer_grouped_parameters[0]["params"] + \
        optimizer_grouped_parameters[1]["params"]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    total_train_batch_size, max_steps, num_train_epochs = \
        get_max_steps(training_args, n_training_samples, trainer.train_dataset)

    # Get LR scheduler
    lr_scheduler = transformers_get_scheduler(
        name = training_args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps = training_args.get_warmup_steps(max_steps),
        num_training_steps = max_steps,
        **getattr(training_args, "lr_scheduler_kwargs", {}),
    )

    # Gradient accumulation and grad norm clipping
    max_grad_norm   = training_args.max_grad_norm
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    bsz = training_args.per_device_train_batch_size
    ga  = training_args.gradient_accumulation_steps

    # Mixed precision scaling
    torch_version = torch.__version__
    if model.config.torch_dtype == torch.float16:
        mixed_precision = "fp16"
        mixed_dtype = torch.float16
        # torch.cuda.amp.autocast is deprecated >= 2.4
        if Version(torch_version) < Version("2.4.0"):
            if device_manager.is_tpu:
                float16_scaler = None  # TPU doesn't use GradScaler
            else:
                from . import DEVICE_TYPE
                if DEVICE_TYPE == "cuda":
                    float16_scaler = torch.cuda.amp.GradScaler()
                else:
                    # For XPU and others, use device-agnostic version
                    float16_scaler = torch.amp.GradScaler("cpu")
        else:
            if device_manager.is_tpu:
                float16_scaler = None  # TPU doesn't use GradScaler
            else:
                # Use device-agnostic GradScaler
                from . import DEVICE_TYPE
                if DEVICE_TYPE in ["cuda", "xpu"]:
                    float16_scaler = torch.amp.GradScaler(DEVICE_TYPE)
                else:
                    float16_scaler = torch.amp.GradScaler("cpu")
    else:
        mixed_precision = "bf16"
        mixed_dtype = torch.bfloat16
        float16_scaler = None
    pass
    
    optimizer.zero_grad()

    # torch.cuda.amp.autocast is deprecated >= 2.4
    torch_version = torch.__version__
    if Version(torch_version) < Version("2.4.0"):
        if device_manager.is_tpu:
            # Create a no-op context manager for TPU
            autocast_context_manager = nullcontext()
        else:
            autocast_context_manager = torch.cuda.amp.autocast(
                dtype = mixed_dtype,
                cache_enabled = False,
            )
    else:
        if device_manager.is_tpu:
            # Use CPU autocast for TPU
            autocast_context_manager = torch.amp.autocast(
                device_type = "cpu",
                dtype = mixed_dtype,
                cache_enabled = False,
            )
        else:
            autocast_context_manager = torch.amp.autocast(
                device_type = "cuda",
                dtype = mixed_dtype,
                cache_enabled = False,
            )
    pass

    step = 0
    accumulated_loss = torch.zeros(1, device = device, dtype = torch.float32)[0]
    
    device_type = "TPU" if device_manager.is_tpu else "CUDA" if torch.cuda.is_available() else "CPU"
    debug_info = \
        f'⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠤⠤⠤⢤⠤⠔⠒⠦⣄⠀⠀⠀⠀⠀⠀⠀⠀  PantheraML - 2x faster free finetuning | Num {device_type}s = {world_size}\n'\
        f'⠀⠀⠀⠀⢀⠠⠤⠤⣄⣠⣀⠀⢀⣠⣤⣦⣤⣤⣤⣤⣵⣌⠛⡦⣄⠀⠀⠀⠀⠀  Num examples = {n_training_samples:,} | Num Epochs = {num_train_epochs:,}\n'\
        f'⠀⠀⠀⢺⣷⣿⣿⡿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣗⣦⣼⣯⡤⢹⣶⣍⠲⢤⣄⠀  Batch size per device = {training_args.per_device_train_batch_size:,} | Gradient Accumulation steps = {training_args.gradient_accumulation_steps}\n'\
        f'⠀⠀⠀⠀⢿⣿⣿⣿⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⠁  Total batch size = {total_train_batch_size:,} | Total steps = {max_steps:,}\n'\
        f'⠀⠀⠀⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀  Number of trainable parameters = {n_parameters_to_train:,}\n'\
        f'⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠛⣿⡿⠁⠀⠀  Device: {device_type} | Rank: {rank}/{world_size}\n'\
        f'⠐⣿⣿⣿⣿⣿⣿⣿⣿⡏⣿⣿⣿⣿⣿⣿⣿⣿⠟⣱⣿⠏⠀⠀⠀⠟⠁⠀⠀⠀\n'\
        f'⠀⠈⢿⣿⣿⣿⣿⣿⣿⡇⢿⣿⣿⣿⣿⣿⣿⣿⢰⣿⣏⠀⠀⠀⠀⠀⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣧⡀⠀⢠⡄⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⠁⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠈⠻⠿⠟⠃⠀⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n'\
        f'⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀'
    if is_main_process:
        print(debug_info)

    # Get per epoch counter
    max_iters_per_epoch = math.ceil(n_training_samples / total_train_batch_size)
    leftover_samples = n_training_samples % total_train_batch_size
    # But also consider leftover steps
    leftover_ga = math.ceil(leftover_samples / bsz)
    if leftover_samples == 0: leftover_ga = ga

    logging_steps = training_args.logging_steps
    # Go through each epoch
    start_time = time.time()
    with ProgressBar(total = max_steps, dynamic_ncols = True) as progress_bar:
        for epoch in range(num_train_epochs):

            # We also need to shuffle the data loader every epoch!
            transformers_set_seed(training_args.seed + epoch)
            train_dataloader_iterator = iter(torch.utils.data.DataLoader(
                trainer.train_dataset,
                batch_size     = bsz,
                sampler        = torch.utils.data.SequentialSampler(trainer.train_dataset),
                num_workers    = training_args.dataloader_num_workers,
                collate_fn     = data_collator,
                pin_memory     = training_args.dataloader_pin_memory,
                drop_last      = training_args.dataloader_drop_last,
                worker_init_fn = trainer_utils_seed_worker,
            ))

            for j in range(max_iters_per_epoch):
                n_batches = leftover_ga if j == (max_iters_per_epoch-1) else ga
                batches = [next(train_dataloader_iterator) for j in range(n_batches)]

                # Count non zeros before loss calc
                n_items = torch.stack([
                    torch.count_nonzero(x["labels"][..., 1:] != -100) for x in batches
                ]).sum()

                # Gradient accumulation
                for batch in batches:
                    input_ids = device_manager.to_device(batch["input_ids"].pin_memory())
                    labels    = device_manager.to_device(batch["labels"].pin_memory())

                    with autocast_context_manager:
                        loss = model(input_ids = input_ids, labels = labels, n_items = n_items).loss
                        # Normalize loss by world size for distributed training
                        if world_size > 1:
                            loss = loss / world_size
                        accumulated_loss += loss.detach()
                    pass

                    if float16_scaler is None:  
                        loss.backward()
                    else: 
                        float16_scaler.scale(loss).backward()
                pass

                # Synchronize gradients across all processes
                if world_size > 1:
                    for param in trainable_parameters:
                        if param.grad is not None:
                            param.grad.data = device_manager.all_reduce(param.grad.data)

                if float16_scaler is None:
                    clip_grad_norm_(trainable_parameters, max_grad_norm)
                    optimizer.step()
                else:
                    float16_scaler.unscale_(optimizer)
                    clip_grad_norm_(trainable_parameters, max_grad_norm)
                    float16_scaler.step(optimizer)
                    float16_scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                # TPU/XLA specific synchronization
                device_manager.mark_step()

                if step % logging_steps == 0 and is_main_process:
                    current_loss = accumulated_loss.cpu().item()
                    progress_bar.write(f"{step}, {round(current_loss, 4)}")
                    
                    # Track performance metrics
                    performance_monitor.track_step_metrics({
                        'step': step,
                        'loss': current_loss,
                        'learning_rate': lr_scheduler.get_last_lr()[0],
                        'epoch': epoch + (j / max_iters_per_epoch),
                    })
                pass
                accumulated_loss.zero_()
                if is_main_process:
                    progress_bar.update(1)

                step += 1
                if step == max_steps: break
            pass
        pass
    pass
    unset_training(model)
    if is_main_process:
        logger.info("PantheraML: Finished training!")
        print("PantheraML: Finished training!")
        
        # Log final performance metrics
        performance_metrics = performance_monitor.get_summary()
        logger.info(f"Training completed - Runtime: {end_time - start_time:.2f}s")
        logger.info(f"Performance metrics: {performance_metrics}")
    end_time = time.time()

    # Return stats
    trainer_stats = Trainer_Stats(metrics = {"train_runtime" : end_time - start_time})
    return trainer_stats
pass

# PantheraML Zoo - Utilities for PantheraML
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
