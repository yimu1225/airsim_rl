# Paired AOSA policy explanations

This directory contains a model-agnostic, flow-guided spatiotemporal
occlusion explainer for `CL-VSSM-SAC` and `CL-SAC`. It does not launch AirSim.
It reuses the `base_state` and four-frame `depth` arrays stored by an existing
successful MambaLRP trajectory, so both policies are explained on exactly the
same observations.

The target is each Actor's pre-tanh mean. Three separate absolute-influence
figures and three signed-effect figures are produced per captured step, along
with one combined all-actions signed figure, an original-frames figure, a
whole-frame deletion chart, compressed raw arrays, and metadata. The combined
signed figure has one original row, three CL-VSSM-SAC action rows, and three
CL-SAC action rows. The saved `depth` array is copied without rescaling,
inversion, frame replacement, or temporal reordering. `metadata.json` records
the source NPZ and a SHA-256 digest of each exact float32 depth sequence.

## Validate an existing trajectory without loading either model

```bash
conda run -n AirSim python explainability_aosa/visualize_paired.py \
  --reference_run results/explainability/mambalrp/test_scene/CL-VSSM-SAC/seed25/episode25/run_20260731T014620_934386Z \
  --capture_steps 1 \
  --dry_run
```

## Generate paired explanations

```bash
conda run -n AirSim python explainability_aosa/visualize_paired.py \
  --reference_run results/explainability/mambalrp/test_scene/CL-VSSM-SAC/seed25/episode25/run_20260731T014620_934386Z \
  --capture_steps 1 6 11 \
  --device cuda
```

Omit `--capture_steps` to process every captured step in the reference run.
Omit `--reference_run` to select the newest successful MambaLRP run.

To reproduce a compact paper-style trajectory figure, process all saved steps
but render only one multi-step heatmap:

```bash
conda run -n AirSim python explainability_aosa/visualize_paired.py \
  --reference_run results/explainability/mambalrp/test_scene/CL-VSSM-SAC/seed25/episode25/run_20260731T014620_934386Z \
  --device cuda \
  --summary_only \
  --summary_max_steps 10
```

The command produces one paired summary figure. Every selected step is a
seven-by-four block: the original depth row, three CL-VSSM-SAC action rows,
then three CL-SAC action rows; columns are `t-3`, `t-2`, `t-1`, and `t`. Step
blocks are placed consecutively in one horizontal row without wrapping. The
figure has no overall title or colorbar and uses equal compact horizontal and
vertical gaps. The blue-to-red `turbo` map encodes absolute influence only;
signed promotion/suppression remains available in the separate signed figures.

The default reference depth is `255`, which represents obstacle-free/far space
in this repository. Do not replace it with zero unless the depth semantics are
also changed: zero represents a very close obstacle and is not a neutral
baseline here.

The default 16-by-16 window and stride 4 evaluate 3,364 masks for a four-frame
128-by-128 sample. The dense overlapping masks produce finer maps without
post-hoc Gaussian smoothing. Use `--window_height 24 --window_width 24
--stride 8` for a faster 784-mask run, or `--window_height 32 --window_width 32
--stride 16` for a 196-mask exploratory run.

Use `--motion fixed` for conventional fixed cuboid occlusion. The default
`--motion flow` tracks each rectangular mask across adjacent frames with dense
Farneback optical flow, forming an AOSA-style motion-guided 3-D occlusion tube.

Reference: Uchiyama et al., *Visually Explaining 3D-CNN Predictions for Video
Classification With an Adaptive Occlusion Sensitivity Analysis*, WACV 2023.
