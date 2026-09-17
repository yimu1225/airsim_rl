from __future__ import annotations

from typing import Dict, List, Tuple

_CANONICAL_ALGORITHMS: Tuple[str, ...] = (
    "TD3",
    "DDPG",
    "VSSM_TD3",
    "AETD3",
    "SAC",
    "SAC_FAE",
    "LSTM_SAC",
    "no-VSSM",
    "no-SB-PER",
    "MM_VSSM_SAC",
    "SVSSM_SAC",
    "SAFE_VSSM_SAC",
    "VSSM-SAC",
    "SB_PER_SVSSM_SAC",
    "Transformer_SAC",
    "SSVM_SAC",
    "SB_PER_MambaCSJA_SAC",
    "PPO",
    "VSSM_PPO",
    "SDDPG",
)

def _normalize_key(name: str) -> str:
    return str(name).strip().lower()


_ALIAS_TO_CANONICAL: Dict[str, str] = {}


for _algo in _CANONICAL_ALGORITHMS:
    _ALIAS_TO_CANONICAL[_normalize_key(_algo)] = _algo
    _ALIAS_TO_CANONICAL[_normalize_key(_algo.replace("_", "-"))] = _algo


def split_curriculum_prefix(algorithm_name: str) -> Tuple[bool, str]:
    name = str(algorithm_name).strip()
    if name.lower().startswith("cl-"):
        return True, name[3:].strip()
    return False, name


def to_internal_core_algorithm_name(algorithm_name: str) -> str:
    _, core_name = split_curriculum_prefix(algorithm_name)
    if not core_name:
        raise ValueError("Algorithm name is empty.")

    core_key = _normalize_key(core_name)
    canonical = _ALIAS_TO_CANONICAL.get(core_key)
    if canonical is not None:
        return canonical

    fallback = core_key.replace("-", "_")
    canonical = _ALIAS_TO_CANONICAL.get(fallback)
    if canonical is not None:
        return canonical

    supported = ", ".join(sorted(_CANONICAL_ALGORITHMS))
    raise ValueError(f"Unknown algorithm '{algorithm_name}'. Supported: {supported}")


def to_internal_algorithm_name(algorithm_name: str) -> str:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    core_name = to_internal_core_algorithm_name(algorithm_name)
    return f"CL-{core_name}" if use_curriculum else core_name


def to_kebab_algorithm_name(algorithm_name: str, upper: bool = False) -> str:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    core_name = to_internal_core_algorithm_name(algorithm_name)
    kebab_name = core_name.replace("_", "-")
    if upper:
        kebab_name = kebab_name.upper()
    return f"CL-{kebab_name}" if use_curriculum else kebab_name


def to_output_algorithm_name(algorithm_name: str) -> str:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    core_name = to_internal_core_algorithm_name(algorithm_name)
    output_name = core_name if core_name == "SAC_FAE" else core_name.replace("_", "-")
    return f"CL-{output_name}" if use_curriculum else output_name


def to_plot_algorithm_label(algorithm_name: str) -> str:
    return to_output_algorithm_name(algorithm_name)


def supported_algorithm_display_names() -> List[str]:
    return [to_output_algorithm_name(algo) for algo in _CANONICAL_ALGORITHMS]


def normalize_algorithm_name_for_config(algorithm_value: str) -> str:
    raw_value = str(algorithm_value).strip()
    if not raw_value:
        return raw_value

    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    return ",".join(to_output_algorithm_name(token) for token in tokens)


def expand_algorithm_spec(algo_spec: str) -> List[str]:
    """Parse explicitly listed algorithm names in their supplied order."""
    value = str(algo_spec).strip()
    if not value:
        return []

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return [to_internal_algorithm_name(token) for token in tokens]


def is_curriculum_algorithm(algorithm_name: str) -> bool:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    return use_curriculum


__all__ = [
    "expand_algorithm_spec",
    "is_curriculum_algorithm",
    "normalize_algorithm_name_for_config",
    "split_curriculum_prefix",
    "to_internal_algorithm_name",
    "to_internal_core_algorithm_name",
    "to_kebab_algorithm_name",
    "to_output_algorithm_name",
    "to_plot_algorithm_label",
    "supported_algorithm_display_names",
]
