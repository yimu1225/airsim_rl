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

## Fixed-density test in the training scene

`eval_training_density.py` launches the generated training environment and tests
exactly 160, 180, and 200 static obstacles. Each tier runs 100 episodes by
default and reports mean reward, mean successful 3D path length, success rate,
and collision rate.

```bash
python -m eval.eval_training_density \
  --algorithm_name DDPG,TD3,PPO,Transformer-SAC,VSSM-SAC \
  --seed 29 \
  --eval_episodes 100
```

The separate `--layout_seed` (default `20260722`) determines all UE4 layouts.
Keep it unchanged for every algorithm. Persistent manifests under
`results/eval/training_density_path_length/manifests/` verify that episode
configurations match even when algorithms are evaluated in separate commands.
Per-episode CSVs and a four-metric `summary.csv` are written under
`results/eval/training_density_path_length/<algorithm>/seed<seed>/`. The legacy
`results/eval/training_density/` directory is rejected as an output root so the
previous evaluation data cannot be overwritten accidentally.

Each per-episode CSV row records the measured 3D `path_length` in metres (two
decimal places), `is_success`, `has_collided`, `is_timeout`, and the mutually
exclusive `termination_reason`. Normal terminal reasons are
`success`, `collision`, and `timeout`; simulator recovery and unexpected early
termination are marked as `environment_restart` and `other_failure` instead of
being incorrectly counted as timeouts. The cumulative `success_rate` and
`collision_rate` columns are retained for plotting compatibility.

To run only selected density tiers:

```bash
python -m eval.eval_training_density \
  --algorithm_name VSSM-SAC \
  --obstacle_counts 170 210
```

## Plot fixed-density bar charts

After evaluation data are available for all three density tiers, generate the
four metric figures with:

```bash
python -m eval.plot_training_density_bars
```

By default, the plotting script discovers every algorithm directory that has
complete 160/180/200 episode CSV files. It does not require a fixed algorithm
list or seed. If multiple complete seeds exist for one algorithm, all of their
episodes are combined; incomplete algorithms/seeds are skipped with a message.
Use `--algorithms` or `--model_seed` only when an explicit filter is needed.

The script reads the same per-episode CSV schema as the real evaluation and
writes `mean_reward.png`, `mean_path_length.png`, `success_rate.png`, and
`collision_rate.png` plus an aggregated metric table in the `figures/` folder.
