"""Model-agnostic spatiotemporal occlusion explanations for UAV policies.

This module implements an AOSA-style explainer: motion-guided 3-D occlusion
tubes are applied directly to a depth sequence and their effect on each scalar
policy output is accumulated over the occluded voxels.  It deliberately treats
the policy as a black box, so the same explanation unit is used for CNN and
Vision-Mamba policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np


PolicyPredictor = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OcclusionConfig:
    """Configuration for flow-guided spatiotemporal occlusion."""

    window_height: int = 32
    window_width: int = 32
    stride: int = 16
    temporal_radius: int = 1
    reference_depth: float = 255.0
    batch_size: int = 32
    motion: str = "flow"

    def validate(self, depth_shape: Sequence[int]) -> None:
        if len(depth_shape) != 3:
            raise ValueError(
                f"Expected depth shape (T,H,W), got {tuple(depth_shape)}"
            )
        frames, height, width = map(int, depth_shape)
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("Depth dimensions must be positive")
        if self.window_height <= 0 or self.window_width <= 0:
            raise ValueError("Occlusion window dimensions must be positive")
        if self.window_height > height or self.window_width > width:
            raise ValueError(
                "Occlusion window must fit inside the depth image"
            )
        if self.stride <= 0:
            raise ValueError("Occlusion stride must be positive")
        if self.stride > min(self.window_height, self.window_width):
            raise ValueError(
                "Occlusion stride cannot exceed the smallest window dimension; "
                "otherwise some pixels would never be evaluated"
            )
        if self.temporal_radius < 0:
            raise ValueError("Temporal radius must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.motion not in {"flow", "fixed"}:
            raise ValueError("motion must be either 'flow' or 'fixed'")
        if not np.isfinite(self.reference_depth):
            raise ValueError("Reference depth must be finite")


@dataclass
class OcclusionResult:
    """Per-action spatiotemporal influence returned by the explainer."""

    original_output: np.ndarray
    absolute_influence: np.ndarray
    signed_influence: np.ndarray
    coverage: np.ndarray
    frame_deletion_delta: np.ndarray
    candidate_count: int
    forward_flow: np.ndarray
    backward_flow: np.ndarray


@dataclass(frozen=True)
class _Candidate:
    anchor_frame: int
    center_y: float
    center_x: float


def _prepare_depth(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(
            f"Expected depth (T,H,W) or (T,1,H,W), got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("Depth sequence contains non-finite values")
    return value


def _grid_centers(length: int, window: int, stride: int) -> list[float]:
    half = window / 2.0
    first = half
    last = float(length) - half
    if first >= last:
        return [float(length - 1) / 2.0]
    centers = list(np.arange(first, last + 1e-6, stride, dtype=np.float32))
    if centers[-1] < last - 1e-6:
        centers.append(last)
    return [float(value) for value in centers]


def _flow_image(depth_frame: np.ndarray) -> np.ndarray:
    """Convert depth into a stable uint8 image for optical flow."""

    clipped = np.clip(depth_frame, 0.0, 255.0)
    proximity = 255.0 - clipped
    image = cv2.GaussianBlur(proximity.astype(np.float32), (5, 5), 0)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def compute_bidirectional_flow(
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adjacent-frame Farneback flow in both directions."""

    value = _prepare_depth(depth)
    frames, height, width = value.shape
    if frames == 1:
        empty = np.zeros((0, height, width, 2), dtype=np.float32)
        return empty, empty.copy()

    forward: list[np.ndarray] = []
    backward: list[np.ndarray] = []
    images = [_flow_image(value[index]) for index in range(frames)]
    parameters = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    for index in range(frames - 1):
        forward.append(
            cv2.calcOpticalFlowFarneback(
                images[index], images[index + 1], None, **parameters
            ).astype(np.float32)
        )
        backward.append(
            cv2.calcOpticalFlowFarneback(
                images[index + 1], images[index], None, **parameters
            ).astype(np.float32)
        )
    return np.stack(forward), np.stack(backward)


def _median_local_flow(
    flow: np.ndarray,
    center_y: float,
    center_x: float,
    radius: int = 3,
) -> tuple[float, float]:
    height, width = flow.shape[:2]
    y = int(np.clip(round(center_y), 0, height - 1))
    x = int(np.clip(round(center_x), 0, width - 1))
    top = max(0, y - radius)
    bottom = min(height, y + radius + 1)
    left = max(0, x - radius)
    right = min(width, x + radius + 1)
    local = flow[top:bottom, left:right]
    dx = float(np.median(local[..., 0]))
    dy = float(np.median(local[..., 1]))
    if not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0
    return dy, dx


def _tracked_center(
    candidate: _Candidate,
    target_frame: int,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    height: int,
    width: int,
) -> tuple[float, float]:
    y = float(candidate.center_y)
    x = float(candidate.center_x)
    anchor = int(candidate.anchor_frame)
    if target_frame > anchor:
        for index in range(anchor, target_frame):
            dy, dx = _median_local_flow(forward_flow[index], y, x)
            y += dy
            x += dx
    elif target_frame < anchor:
        for index in range(anchor - 1, target_frame - 1, -1):
            dy, dx = _median_local_flow(backward_flow[index], y, x)
            y += dy
            x += dx
    return (
        float(np.clip(y, 0.0, height - 1.0)),
        float(np.clip(x, 0.0, width - 1.0)),
    )


def _window_bounds(center: float, window: int, limit: int) -> tuple[int, int]:
    start = int(round(center - window / 2.0))
    start = int(np.clip(start, 0, limit - window))
    return start, start + window


class SpatiotemporalOcclusionExplainer:
    """Black-box AOSA-style explainer for continuous policy outputs."""

    def __init__(
        self,
        predictor: PolicyPredictor,
        config: OcclusionConfig,
    ) -> None:
        self.predictor = predictor
        self.config = config

    def _candidates(self, shape: Sequence[int]) -> list[_Candidate]:
        frames, height, width = map(int, shape)
        rows = _grid_centers(
            height, self.config.window_height, self.config.stride
        )
        columns = _grid_centers(
            width, self.config.window_width, self.config.stride
        )
        return [
            _Candidate(frame, row, column)
            for frame in range(frames)
            for row in rows
            for column in columns
        ]

    def _candidate_mask(
        self,
        candidate: _Candidate,
        shape: Sequence[int],
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
    ) -> np.ndarray:
        frames, height, width = map(int, shape)
        mask = np.zeros((frames, height, width), dtype=bool)
        first_frame = max(
            0, candidate.anchor_frame - self.config.temporal_radius
        )
        last_frame = min(
            frames - 1,
            candidate.anchor_frame + self.config.temporal_radius,
        )
        for frame in range(first_frame, last_frame + 1):
            if self.config.motion == "flow" and frame != candidate.anchor_frame:
                center_y, center_x = _tracked_center(
                    candidate,
                    frame,
                    forward_flow,
                    backward_flow,
                    height,
                    width,
                )
            else:
                center_y, center_x = (
                    candidate.center_y,
                    candidate.center_x,
                )
            top, bottom = _window_bounds(
                center_y, self.config.window_height, height
            )
            left, right = _window_bounds(
                center_x, self.config.window_width, width
            )
            mask[frame, top:bottom, left:right] = True
        return mask

    @staticmethod
    def _batches(
        values: Sequence[_Candidate], batch_size: int
    ) -> Iterator[Sequence[_Candidate]]:
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]

    def explain(
        self,
        base_state: np.ndarray,
        depth: np.ndarray,
        *,
        max_masks: int | None = None,
    ) -> OcclusionResult:
        """Explain every scalar policy output for one depth sequence."""

        depth_value = _prepare_depth(depth)
        self.config.validate(depth_value.shape)
        base_value = np.asarray(base_state, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(base_value)):
            raise ValueError("Base state contains non-finite values")

        original = np.asarray(
            self.predictor(base_value[None], depth_value[None]),
            dtype=np.float32,
        )
        if original.ndim != 2 or original.shape[0] != 1:
            raise ValueError(
                "Policy predictor must return (B,A); got "
                f"{original.shape}"
            )
        if not np.all(np.isfinite(original)):
            raise RuntimeError("Policy predictor returned non-finite values")
        original_output = original[0]
        action_count = int(original_output.size)

        if self.config.motion == "flow":
            forward_flow, backward_flow = compute_bidirectional_flow(
                depth_value
            )
        else:
            frames, height, width = depth_value.shape
            forward_flow = np.zeros(
                (max(frames - 1, 0), height, width, 2),
                dtype=np.float32,
            )
            backward_flow = np.zeros_like(forward_flow)

        candidates = self._candidates(depth_value.shape)
        if max_masks is not None:
            if int(max_masks) <= 0:
                raise ValueError("max_masks must be positive when provided")
            candidates = candidates[: int(max_masks)]
        absolute_sum = np.zeros(
            (action_count, *depth_value.shape), dtype=np.float64
        )
        signed_sum = np.zeros_like(absolute_sum)
        coverage = np.zeros(depth_value.shape, dtype=np.float64)

        for candidate_batch in self._batches(
            candidates, self.config.batch_size
        ):
            masks = np.stack(
                [
                    self._candidate_mask(
                        candidate,
                        depth_value.shape,
                        forward_flow,
                        backward_flow,
                    )
                    for candidate in candidate_batch
                ],
                axis=0,
            )
            batch_depth = np.broadcast_to(
                depth_value, (len(candidate_batch), *depth_value.shape)
            ).copy()
            batch_depth[masks] = float(self.config.reference_depth)
            batch_base = np.broadcast_to(
                base_value, (len(candidate_batch), base_value.size)
            ).copy()
            perturbed = np.asarray(
                self.predictor(batch_base, batch_depth), dtype=np.float32
            )
            expected_shape = (len(candidate_batch), action_count)
            if perturbed.shape != expected_shape:
                raise ValueError(
                    "Policy predictor returned an unexpected shape: "
                    f"expected {expected_shape}, got {perturbed.shape}"
                )
            if not np.all(np.isfinite(perturbed)):
                raise RuntimeError(
                    "Policy predictor returned non-finite perturbed outputs"
                )
            signed_delta = original_output[None] - perturbed
            mask_values = masks.astype(np.float64)
            absolute_sum += np.einsum(
                "ba,bthw->athw",
                np.abs(signed_delta).astype(np.float64),
                mask_values,
                optimize=True,
            )
            signed_sum += np.einsum(
                "ba,bthw->athw",
                signed_delta.astype(np.float64),
                mask_values,
                optimize=True,
            )
            coverage += mask_values.sum(axis=0)

        safe_coverage = np.where(coverage > 0.0, coverage, 1.0)
        absolute = absolute_sum / safe_coverage[None]
        signed = signed_sum / safe_coverage[None]
        absolute[:, coverage == 0.0] = 0.0
        signed[:, coverage == 0.0] = 0.0

        frame_deleted = np.broadcast_to(
            depth_value, (depth_value.shape[0], *depth_value.shape)
        ).copy()
        for frame in range(depth_value.shape[0]):
            frame_deleted[frame, frame] = float(
                self.config.reference_depth
            )
        frame_base = np.broadcast_to(
            base_value, (depth_value.shape[0], base_value.size)
        ).copy()
        frame_outputs = np.asarray(
            self.predictor(frame_base, frame_deleted), dtype=np.float32
        )
        frame_deletion_delta = (
            original_output[None] - frame_outputs
        ).T.astype(np.float32)

        return OcclusionResult(
            original_output=original_output.astype(np.float32),
            absolute_influence=absolute.astype(np.float32),
            signed_influence=signed.astype(np.float32),
            coverage=coverage.astype(np.float32),
            frame_deletion_delta=frame_deletion_delta,
            candidate_count=len(candidates),
            forward_flow=forward_flow.astype(np.float32),
            backward_flow=backward_flow.astype(np.float32),
        )


def describe_mask_plan(
    depth: np.ndarray,
    config: OcclusionConfig,
) -> dict:
    """Return a model-free summary used by the runner's dry-run mode."""

    value = _prepare_depth(depth)
    config.validate(value.shape)
    frames, height, width = value.shape
    rows = _grid_centers(height, config.window_height, config.stride)
    columns = _grid_centers(width, config.window_width, config.stride)
    return {
        "depth_shape": list(map(int, value.shape)),
        "grid_rows": len(rows),
        "grid_columns": len(columns),
        "candidate_count": frames * len(rows) * len(columns),
        "window": [config.window_height, config.window_width],
        "stride": config.stride,
        "temporal_radius": config.temporal_radius,
        "motion": config.motion,
        "reference_depth": config.reference_depth,
    }
