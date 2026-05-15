"""Profile the NKI scatter_reduce kernel in isolation."""
import os

os.environ['NEURON_RT_INSPECT_ENABLE'] = '1'
os.environ['NEURON_RT_INSPECT_DEVICE_PROFILE'] = '1'
os.environ['NEURON_RT_INSPECT_OUTPUT_DIR'] = './profile_nki_output'
os.environ['NEURON_CC_FLAGS'] = '--target trn2 --lnc 1'
os.environ['NEURON_RT_VISIBLE_CORES'] = '0'

import sys
sys.path.insert(0, "kernels")

import torch

device = torch.device("neuron")

from scatter_reduce import nki_scatter_reduce_sum_kernel

# Equiformer-style input: 2 atoms, 20 neighbors each → 40 edges
length = 2
index = torch.cat([
    torch.zeros(20, dtype=torch.int32),
    torch.ones(20, dtype=torch.int32),
]).to(device)

# Warmup
for _ in range(2):
    _ = nki_scatter_reduce_sum_kernel(index, length)

# Timed run (this execution generates the NTFF when profiled)
result = nki_scatter_reduce_sum_kernel(index, length)
print(f"Result: {result.cpu().tolist()}")
print(f"Expected: [20.0, 20.0]")
print(f"NEFF written to: {os.environ['NEURON_RT_INSPECT_OUTPUT_DIR']}")
