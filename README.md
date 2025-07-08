<div align="center">

  <h1>🐾 PantheraML Zoo</h1>
  <h3>Production-Ready Multi-GPU/TPU Training Utilities</h3>
  
  <p>
    <strong>Supercharged training with automatic device detection, distributed training, and production monitoring</strong>
  </p>

### PantheraML Zoo - Production Utils for High-Performance Training!

![PantheraML Training](https://i.ibb.co/sJ7RhGG/image-41.png)

</div>

## ✨ Finetune for Free

All notebooks are **beginner friendly**! Add your dataset, click "Run All", and you'll get a 2x faster finetuned model which can be exported to GGUF, Ollama, vLLM or uploaded to Hugging Face.

| Unsloth supports | Free Notebooks | Performance | Memory use |
|-----------|---------|--------|----------|
| **Llama 3.2 (3B)**      | [▶️ Start for free](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)               | 2x faster | 60% less |
| **Llama 3.1 (8B)**      | [▶️ Start for free](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)               | 2x faster | 60% less |
| **Phi-3.5 (mini)** | [▶️ Start for free](https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)               | 2x faster | 50% less |
| **Gemma 2 (9B)**      | [▶️ Start for free](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)               | 2x faster | 63% less |
| **Mistral Small (22B)** | [▶️ Start for free](https://colab.research.google.com/drive/1oCEHcED15DzL8xXGU1VTx5ZfOJM8WY01?usp=sharing)               | 2x faster | 60% less |
| **Ollama**     | [▶️ Start for free](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)               | 1.9x faster | 43% less |
| **Mistral v0.3 (7B)**    | [▶️ Start for free](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)               | 2.2x faster | 73% less |
| **ORPO**     | [▶️ Start for free](https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)               | 1.9x faster | 43% less |
| **DPO Zephyr**     | [▶️ Start for free](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)               | 1.9x faster | 43% less |

- **Kaggle Notebooks** for [Llama 3.1 (8B)](https://www.kaggle.com/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook), [Gemma 2 (9B)](https://www.kaggle.com/code/danielhanchen/kaggle-gemma-7b-unsloth-notebook/), [Mistral (7B)](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook)
- Run [Llama 3.2 1B 3B notebook](https://colab.research.google.com/drive/1hoHFpf7ROqk_oZHzxQdfPW9yvTxnvItq?usp=sharing) and [Llama 3.2 conversational notebook](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)
- Run [Llama 3.1 conversational notebook](https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing) and [Mistral v0.3 ChatML](https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing)
- This [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) is for continued pretraining / raw text
- This [continued pretraining notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) is for learning another language
- Click [here](https://github.com/unslothai/unsloth/wiki) for detailed documentation for Unsloth.


## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 📚 **Documentation & Wiki**              | [Read Our Docs](https://docs.unsloth.ai) |
| <img height="14" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" />&nbsp; **Twitter (aka X)**              |  [Follow us on X](https://twitter.com/unslothai)|
| 💾 **Installation**               | [unsloth/README.md](https://github.com/unslothai/unsloth/tree/main#-installation-instructions)|
| 🥇 **Benchmarking**                   | [Performance Tables](https://github.com/unslothai/unsloth/tree/main#-performance-benchmarking)
| 🌐 **Released Models**            | [Unsloth Releases](https://huggingface.co/unsloth)|
| ✍️ **Blog**                    | [Read our Blogs](https://unsloth.ai/blog)|

## ⭐ Key Features
- All kernels written in [OpenAI's Triton](https://openai.com/research/triton) language. **Manual backprop engine**.
- **0% loss in accuracy** - no approximation methods - all exact.
- No change of hardware. Supports NVIDIA GPUs since 2018+. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40 etc) [Check your GPU!](https://developer.nvidia.com/cuda-gpus) GTX 1070, 1080 works, but is slow.
- Works on **Linux** and **Windows** via WSL.
- Supports 4bit and 16bit QLoRA / LoRA finetuning via [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
- Open source trains 5x faster - see [Unsloth Pro](https://unsloth.ai/) for up to **30x faster training**!
- If you trained a model with 🦥Unsloth, you can use this cool sticker! &nbsp; <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" height="50" align="center" />


## 💾 Installation Instructions

PantheraML Zoo provides production-ready utilities for high-performance training. For stable releases, use `pip install unsloth_zoo`. We recommend `pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"` for the latest features.

```bash
pip install unsloth_zoo
```

**Note:** PantheraML Zoo still builds upon [Unsloth](https://github.com/unslothai/unsloth) for core functionality, so install Unsloth as well!

## 🚀 Multi-GPU and TPU Support

PantheraML now supports distributed training across multiple devices:

### Multi-GPU Training

```python
from unsloth_zoo.training_utils import unsloth_train
from unsloth_zoo.device_utils import setup_distributed

# Setup distributed training automatically
device_manager = setup_distributed()

# Your existing training code works unchanged
trainer = YourTrainer(model, training_args, train_dataset, ...)
unsloth_train(trainer)
```

**Launch with multiple GPUs:**
```bash
# Use torchrun for multi-GPU training
torchrun --nproc_per_node=4 your_training_script.py

# Or use traditional method
python -m torch.distributed.launch --nproc_per_node=4 your_training_script.py
```

### TPU Training

```python
# TPU is detected automatically - no code changes needed!
# Just run your script normally on a TPU instance
python your_training_script.py
```

### Features:
- ✅ **Automatic device detection** (CUDA, TPU, CPU)
- ✅ **Distributed gradient synchronization**
- ✅ **TPU-optimized training loops**
- ✅ **Mixed precision support** across all devices
- ✅ **Gradient accumulation** with proper scaling
- ✅ **Progress tracking** from main process only
- ✅ **Backward compatibility** with existing code

### Environment Variables:
```bash
# For multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# For TPU (on Google Cloud)
export TPU_NAME=your-tpu-name
```

### Device Manager API:
```python
from unsloth_zoo.device_utils import get_device_manager

dm = get_device_manager()
print(f"Device: {dm.device}")           # Current device
print(f"World Size: {dm.world_size}")   # Number of processes
print(f"Rank: {dm.rank}")               # Current process rank
print(f"Is TPU: {dm.is_tpu}")           # Running on TPU?
print(f"Is Main: {dm.is_main_process}") # Main process?

# Move tensors to device
tensor = dm.to_device(your_tensor)

# Synchronize across processes
dm.barrier()

# All-reduce tensors
reduced_tensor = dm.all_reduce(tensor)
```

## 🏭 Production Features

PantheraML Zoo includes production-ready features for enterprise deployment:

### Production Logging
```python
from unsloth_zoo import get_logger, setup_production_logging

# Setup structured logging
setup_production_logging(level="INFO", format="json")
logger = get_logger(__name__)

logger.info("Training started", extra={"model": "llama-7b", "batch_size": 16})
```

### Error Handling & Recovery
```python
from unsloth_zoo import ErrorHandler, with_error_handling

# Automatic checkpointing and recovery
error_handler = ErrorHandler(checkpoint_dir="./checkpoints")

@with_error_handling(error_handler)
def train_model():
    # Your training code here
    pass
```

### Performance Monitoring
```python
from unsloth_zoo import get_performance_monitor, track_metrics

monitor = get_performance_monitor()

with monitor.training_context():
    # Training automatically tracked
    train_model()
    
# Get comprehensive metrics
metrics = monitor.get_summary()
print(f"Throughput: {metrics['tokens_per_second']} tokens/sec")
```

### Configuration Management
```python
from unsloth_zoo import load_config, ProductionConfig

# Load from environment variables and config files
config = load_config()

# Or define programmatically
config = ProductionConfig(
    max_sequence_length=4096,
    enable_checkpointing=True,
    checkpoint_frequency=100,
    enable_performance_monitoring=True
)
```


## License

PantheraML Zoo is licensed under the GNU Affero General Public License.
