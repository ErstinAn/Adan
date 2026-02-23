# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adan (Adaptive Nesterov Momentum Algorithm) is a PyTorch optimizer for deep learning. It uses three exponential moving averages (betas: gradient, gradient difference, second-order moment) and Nesterov momentum for faster convergence. Key advantage: supports 5-10x larger learning rates than Adam/AdamW and can match results in half the training steps.

## Build & Install

```bash
# Full install (with fused CUDA kernels, requires CUDA toolkit + ninja)
pip install .

# Unfused install (CPU-compatible, no CUDA required)
python setup.py install --unfused

# Force CUDA build even without GPU detected
FORCE_CUDA=1 pip install .
```

Build system: `setup.py` with setuptools. CUDA extensions are conditionally compiled via `torch.utils.cpp_extension.CUDAExtension`.

## Architecture

### Core Optimizer (`adan.py`)

Single-file optimizer inheriting from `torch.optim.Optimizer`. Four implementation paths selected via `foreach` and `fused` flags:

| Path | Function | When to use |
|------|----------|-------------|
| `_single_tensor_adan` | Loop-based, one param at a time | Lower peak memory |
| `_multi_tensor_adan` | Uses `torch._foreach_*` ops | Default (`foreach=True`) |
| `_fused_adan_single_tensor` | CUDA kernel per tensor | `fused=True, foreach=False` |
| `_fused_adan_multi_tensor` | CUDA kernel, batched | `fused=True, foreach=True` (fastest) |

### CUDA Kernels (`fused_adan/`)

- `pybind_adan.cpp` — pybind11 bindings exposing `adan_single_tensor()` and `adan_multi_tensor()`
- `fused_adan_kernel.cu` — Single-tensor CUDA kernel with float4 vectorization for float32
- `multi_tensor_adan_kernel.cu` — Multi-tensor variant using NVIDIA's multi_tensor_apply pattern
- Headers in `fused_adan/include/` for kernel declarations and type shims

### Experiment Directories

- `CV/timm/` — ViT, ResNet, ConvNext training with timm framework. `optim_factory.py` integrates Adan into timm's optimizer creation.
- `CV/MAE/` — Masked AutoEncoder pretraining/finetuning
- `NLP/BERT/` — BERT pretraining and GLUE finetuning with YAML configs in `config/`
- `NLP/Transformer-XL/` — Transformer-XL experiments
- `gpt2/` — GPT-2 345M training (`pretrain.sh`)

## Usage Pattern

```python
from adan import Adan
optimizer = Adan(model.parameters(), lr=1e-3, betas=(0.98, 0.92, 0.99),
                 weight_decay=0.02, max_grad_norm=0.0, no_prox=False,
                 foreach=True, fused=False)
```

## Testing

No formal test suite exists. Validation is done through training experiments in `CV/`, `NLP/`, and `gpt2/` directories with reproducible configs.
