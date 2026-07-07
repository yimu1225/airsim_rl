# Evaluation

This folder contains test-only code for evaluating trained policies in the UE4 test scene.
It does not use or create `EnvGenConfig.json`.

Examples:

```bash
python -m eval.eval_async \
  --algorithm_name CL-VSSM-SAC \
  --seed 29 \
  --load_model models/CL-VSSM-SAC/seed29/async_final.pth \
  --eval_episodes 10
```

Use:

- `eval_async.py` for most non-PPO, non-LSTM-SAC algorithms.
- `eval_ppo.py` for PPO/VSSM-PPO/PL-VSSM-PPO.
- `eval_lstm_sac.py` for LSTM-SAC.

All of these use `eval_env.py`, which opens the UE4 test scene and does not randomize the environment.
