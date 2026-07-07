from __future__ import annotations

from typing import Dict, List, Tuple

_CANONICAL_ALGORITHMS: Tuple[str, ...] = (
    "TD3",
    "DDPG",
    "SB_PER_TD3",
    "VSSM_TD3",
    "STV_Patch_TD3",
    "Vim_TD3",
    "ST_Seq_Vim_TD3",
    "STV_Seq_Vim_TD3",
    "SB_PER_VSSM_TD3",
    "SAFE_VSSM_TD3",
    "Mamba_TD3",
    "ST_DualVim_TD3",
    "AETD3",
    "SAC",
    "SAC_Beta",
    "LSTM_SAC",
    "SB_PER_SAC",
    "VSSM_SAC",
    "MM_VSSM_SAC",
    "PER_VSSM_SAC",
    "SVSSM_SAC",
    "SAFE_VSSM_SAC",
    "VSSM_SAC_Beta",
    "SB_PER_VSSM_SAC",
    "SB_PER_VSSM_SAC_Beta",
    "SB_PER_SVSSM_SAC",
    "Mamba_SAC",
    "Transformer_SAC",
    "Mamba_RSAC",
    "PL_Mamba_RSAC",
    "MambaCSJA_SAC",
    "SB_PER_MambaCSJA_SAC",
    "PER_Mamba_SAC",
    "PPO",
    "VSSM_PPO",
    "PL_VSSM_PPO",
    "PL_TD3",
    "PL_SB_PER_TD3",
    "PL_VSSM_TD3",
    "PL_SAC",
    "PL_SAC_Beta",
    "PL_VSSM_SAC",
    "PL_PER_VSSM_SAC",
    "PL_SB_PER_VSSM_SAC",
    "PL_SB_PER_VSSM_SAC_Beta",
    "PL_SB_PER_VSSM_TD3",
    "SDDPG",
)

ALGORITHM_GROUPS: Dict[str, List[str]] = {
    "all": [
        "TD3",
        "DDPG",
        "SB_PER_TD3",
        "VSSM_TD3",
        "STV_Patch_TD3",
        "Vim_TD3",
        "ST_Seq_Vim_TD3",
        "STV_Seq_Vim_TD3",
        "SB_PER_VSSM_TD3",
        "SAFE_VSSM_TD3",
        "Mamba_TD3",
        "ST_DualVim_TD3",
        "AETD3",
        "SAC",
        "SB_PER_SAC",
        "LSTM_SAC",
        "VSSM_SAC",
        "MM_VSSM_SAC",
        "PER_VSSM_SAC",
        "SVSSM_SAC",
        "SAFE_VSSM_SAC",
        "SB_PER_VSSM_SAC",
        "SB_PER_SVSSM_SAC",
        "Mamba_SAC",
        "Transformer_SAC",
        "Mamba_RSAC",
        "PL_Mamba_RSAC",
        "MambaCSJA_SAC",
        "SB_PER_MambaCSJA_SAC",
        "PER_Mamba_SAC",
        "PPO",
        "VSSM_PPO",
        "PL_VSSM_PPO",
        "PL_TD3",
        "PL_SB_PER_TD3",
        "PL_VSSM_TD3",
        "PL_SAC",
        "PL_VSSM_SAC",
        "PL_PER_VSSM_SAC",
        "PL_SB_PER_VSSM_SAC",
        "PL_SB_PER_VSSM_TD3",
        "SDDPG",
    ],
    "base": [
        "TD3",
        "DDPG",
        "SB_PER_TD3",
        "SAC",
        "SB_PER_SAC",
        "PL_TD3",
        "PL_SB_PER_TD3",
        "AETD3",
        "PL_SAC",
    ],
    "seq": [
        "VSSM_TD3",
        "STV_Patch_TD3",
        "Vim_TD3",
        "ST_Seq_Vim_TD3",
        "STV_Seq_Vim_TD3",
        "SB_PER_VSSM_TD3",
        "SAFE_VSSM_TD3",
        "Mamba_TD3",
        "ST_DualVim_TD3",
        "PL_VSSM_TD3",
        "PL_SB_PER_VSSM_TD3",
        "PL_VSSM_PPO",
        "LSTM_SAC",
        "SB_PER_SAC",
        "VSSM_SAC",
        "MM_VSSM_SAC",
        "PER_VSSM_SAC",
        "SVSSM_SAC",
        "SAFE_VSSM_SAC",
        "SB_PER_VSSM_SAC",
        "SB_PER_SVSSM_SAC",
        "Mamba_SAC",
        "Transformer_SAC",
        "Mamba_RSAC",
        "PL_Mamba_RSAC",
        "MambaCSJA_SAC",
        "PER_Mamba_SAC",
        "PL_VSSM_SAC",
        "PL_PER_VSSM_SAC",
        "PL_SB_PER_VSSM_SAC",
    ],
}

ABLATION_OUTPUT_NAMES: Dict[str, str] = {
    "SB_PER_VSSM_SAC": "VSSM-SAC",
    "VSSM_SAC": "no-SB-PER",
    "SB_PER_SAC": "no-VSSM",
    "SAC": "SAC",
}

ABLATION_PLOT_LABELS: Dict[str, str] = {
    "SB_PER_VSSM_SAC": "VSSM-SAC",
    "VSSM_SAC": "no SB-PER",
    "SB_PER_SAC": "no VSSM",
    "SAC": "SAC",
}

ALGORITHM_GROUPS["vssm_sac_ablation"] = [
    "SB_PER_VSSM_SAC",
    "VSSM_SAC",
    "SB_PER_SAC",
    "SAC",
]


def _normalize_key(name: str) -> str:
    return str(name).strip().lower()


_ALIAS_TO_CANONICAL: Dict[str, str] = {}


for _algo in _CANONICAL_ALGORITHMS:
    _ALIAS_TO_CANONICAL[_normalize_key(_algo)] = _algo
    _ALIAS_TO_CANONICAL[_normalize_key(_algo.replace("_", "-"))] = _algo
    for _compact, _separated in (
        ("VSSM_SAC", "VSSM_SAC"),
        ("VSSM_TD3", "VSSM_TD3"),
        ("VSSM_PPO", "VSSM_PPO"),
        ("SVSSM_SAC", "SVSSM_SAC"),
        ("SVSSM_TD3", "SVSSM_TD3"),
    ):
        if _compact in _algo:
            _alias = _algo.replace(_compact, _separated)
            _ALIAS_TO_CANONICAL[_normalize_key(_alias)] = _algo
            _ALIAS_TO_CANONICAL[_normalize_key(_alias.replace("_", "-"))] = _algo

_ALIAS_TO_CANONICAL.update(
    {
        "vssm-sac": "SB_PER_VSSM_SAC",
        "sb-per-vssm-sac": "SB_PER_VSSM_SAC",
        "sb_per_vssm_sac": "SB_PER_VSSM_SAC",
        "no-sb-per": "VSSM_SAC",
        "no_sb_per": "VSSM_SAC",
        "no sb-per": "VSSM_SAC",
        "no-vssm": "SB_PER_SAC",
        "no_vssm": "SB_PER_SAC",
        "no vssm": "SB_PER_SAC",
        "sb-per-sac": "SB_PER_SAC",
        "sb_per_sac": "SB_PER_SAC",
    }
)


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
    if core_key in ALGORITHM_GROUPS:
        raise ValueError(
            f"'{algorithm_name}' is an algorithm group, not a concrete algorithm."
        )

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
    output_name = ABLATION_OUTPUT_NAMES.get(core_name)
    if output_name is None:
        output_name = core_name.replace("_", "-")
    return f"CL-{output_name}" if use_curriculum else output_name


def to_plot_algorithm_label(algorithm_name: str) -> str:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    core_name = to_internal_core_algorithm_name(algorithm_name)
    label = ABLATION_PLOT_LABELS.get(core_name)
    if label is None:
        label = core_name.replace("_", "-")
    return f"CL-{label}" if use_curriculum else label


def supported_algorithm_display_names() -> List[str]:
    return [to_output_algorithm_name(algo) for algo in _CANONICAL_ALGORITHMS]


def normalize_algorithm_name_for_config(algorithm_value: str) -> str:
    raw_value = str(algorithm_value).strip()
    if not raw_value:
        return raw_value

    if "," not in raw_value:
        key = _normalize_key(raw_value)
        if key in ALGORITHM_GROUPS:
            return key
        return to_output_algorithm_name(raw_value)

    normalized: List[str] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        key = _normalize_key(token)
        if key in ALGORITHM_GROUPS:
            normalized.append(key)
        else:
            normalized.append(to_output_algorithm_name(token))
    return ",".join(normalized)


def expand_algorithm_spec(algo_spec: str) -> List[str]:
    value = str(algo_spec).strip()
    if not value:
        return []

    key = _normalize_key(value)
    if key in ALGORITHM_GROUPS:
        return list(ALGORITHM_GROUPS[key])

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return [to_internal_algorithm_name(token) for token in tokens]


def is_curriculum_algorithm(algorithm_name: str) -> bool:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    return use_curriculum


__all__ = [
    "ALGORITHM_GROUPS",
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
