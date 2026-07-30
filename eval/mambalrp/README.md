# VSSM MambaLRP

- `rules.py`: MambaLRP propagation rules and temporary model wrappers.
- `attribution.py`: policy targets, pixel relevance, conservation, and faithfulness.
- `plotting.py`: heatmap rendering and result serialization.
- `runner.py`: checkpoint loading, test-environment rollout, and CLI handling.
- `self_test.py`: CPU-only forward-equivalence and conservation tests.

The historical command remains available:

```bash
python eval/visualize_vssm_mambalrp.py --self_test
```

The package can also be run directly:

```bash
python -m eval.mambalrp --self_test
```
