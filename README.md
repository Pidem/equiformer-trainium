# EquiformerV3 on AWS Trainium

Benchmarking [EquiformerV3](https://arxiv.org/abs/2604.09130) inference on AWS Trainium2 using PyTorch Native (eager mode).

## Goal

Measure inference throughput (atoms/sec, latency) of EquiformerV3 on Trainium and compare against H100 GPU baselines from the paper.

## Model

EquiformerV3 is an SE(3)-equivariant graph attention Transformer for predicting energy and forces of atomic systems. Key specs of the full model (OC20 S2EF-2M config):

| Parameter | Value |
|-----------|-------|
| Layers | 8 |
| Channels | 128 |
| L_max | 6 |
| M_max | 2 |
| FFN hidden | 512 |
| Attention heads | 8 |
| Parameters | **91M** |
| Cutoff radius | 12 Å |
| Max neighbors | 20 |

## Paper Performance (H100 GPUs)

| Config | Params | Training GPU-hours | Hardware |
|--------|--------|--------------------|----------|
| OC20 S2EF-2M (N=8, L=6) | 91M | 171 | H100 |
| OMat24 (L=4, direct + grad ft) | 30M | 4,782 | 32× H100 |
| OMat24 (L=6, direct + grad ft) | 57M | 9,197 | 32× H100 |
| Matbench Discovery (full pipeline) | 30M | 5,700 | 16-32× H100 |

No per-sample inference latency is published — our Trainium benchmark provides novel data.

## Neuron Compatibility Notes

### Working
- Model instantiation and forward pass (transformer blocks, attention, FFN)
- PyTorch 2.11.0 native eager mode

### Issues Found
- **`torch.masked_select` / boolean indexing** — Neuron has a bug with bool (`i1`) tensors in indexing ops. Workaround: cast mask to `int8` and use `torch.where` + `index_select`. See `docs/neuron_masked_select_bug.md`.
- **`int64` not supported** — auto-cast to `int32` (warning only, non-blocking)
- **Graph generation (`radius_graph_pbc`)** uses dynamic shapes — requires workaround for static-shape accelerators

## Files

| File | Description |
|------|-------------|
| `test_1.py` | Minimal inference test (small model, 2 atoms) |
| `test_2.py` | Full benchmark (91M params, varying atom counts, timing) |
| `requirements.txt` | Python dependencies (torch 2.11.0) |
| `docs/neuron_masked_select_bug.md` | Bug report for Neuron team |

## Setup

```bash
source nki_bootcamp_venv/bin/activate
pip install -r requirements.txt
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
pip install -e equiformer_v3/packages/fairchem-core
python test_1.py
```

## Reference

```bibtex
@article{equiformer_v3,
    title={EquiformerV3: Scaling Efficient, Expressive, and General SE(3)-Equivariant Graph Attention Transformers},
    author={Yi-Lun Liao and Alexander J. Hoffman and Sabrina C. Shen and Alexandre Duval and Sam Walton Norwood and Tess Smidt},
    journal={arXiv preprint arXiv:2604.09130},
    year={2026}
}
```
