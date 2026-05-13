# Bug: `torch.masked_select` fails with bool mask on Neuron (PyTorch Native)

## Summary

`torch.masked_select` crashes on Neuron eager mode when the mask is expanded from a 1D bool tensor to a multi-dimensional bool tensor. The Neuron runtime produces a type mismatch between `i1` (bool) and `i8` tensors internally.

## Environment

- Instance: trn2 (Trainium2)
- PyTorch: 2.11.0 (Neuron native / eager mode)
- torch-neuronx: (from nki_bootcamp_venv)
- CUDA toolkit: 13.0.2
- Python: 3.12

## Reproducer

```python
import torch

device = torch.device("neuron")

# Simulate the pattern from fairchem's radius_graph_pbc
data = torch.randn(500, 3, device=device)
mask_1d = torch.rand(500, device=device) > 0.5  # bool mask

# This fails:
result = torch.masked_select(data, mask_1d.view(-1, 1).expand(-1, 3))
```

## Error

```
RuntimeError: Type mismatch in dependency: module1 output 0 (tensor<500x3xi1>) cannot connect to module2 input 0 (expected tensor<500x3xi8>)
```

## Stack Trace (abbreviated)

```
torch_neuronx/python_ops/torch_mlir/ops/indexing.py:190, in torch_masked_select
    size = mask.sum().item()
torch_neuronx/python_ops/base.py:129, in execute
    result = self._execute_impl(...)
torch_neuronx/python_ops/to_copy.py:110, in _execute_impl
    cpu_dst = copy_neuron_to_cpu(...)
torch_neuronx/python_ops/cast_policy.py:114, in copy_neuron_to_cpu
    _C._nrt_copy_neuron_to_cpu_tensor(neuron_src, cpu_tmp, ...)
RuntimeError: Type mismatch in dependency: module1 output 0 (tensor<500x3xi1>) cannot connect to module2 input 0 (expected tensor<500x3xi8>)
```

## Root Cause

The Neuron `masked_select` implementation in `torch_neuronx/python_ops/torch_mlir/ops/indexing.py` calls `mask.sum().item()` to determine the output size. During this device-to-host copy, the bool tensor (`i1`) is not being correctly cast to `i8` before the transfer.

The issue appears to be in the `copy_neuron_to_cpu` path where a bool (`i1`) tensor is expected as `i8` by the downstream module.

## Workaround

Replace `torch.masked_select` with boolean indexing:

```python
# Instead of:
result = torch.masked_select(data, mask.view(-1, 1).expand(-1, 3))
result = result.view(-1, 3)

# Use:
result = data[mask.unsqueeze(-1).expand(-1, 3)].view(-1, 3)

# Or for 1D:
# Instead of: index1 = torch.masked_select(index1, mask)
# Use:        index1 = index1[mask]
```

## Context

This pattern is used in [fairchem](https://github.com/facebookresearch/fairchem)'s `radius_graph_pbc` function (`fairchem/core/common/utils.py`) which builds neighbor graphs for molecular simulations. It's a core operation for all graph neural network potentials (EquiformerV2, EquiformerV3, GemNet, etc.).

## Impact

Any GNN model using fairchem's graph construction cannot run on Neuron in eager mode without patching `masked_select` → boolean indexing.
