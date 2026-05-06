from __future__ import annotations

from typing import Dict, List, Tuple

_CANONICAL_ALGORITHMS: Tuple[str, ...] = (
    "TD3",
    "DDPG",
    "PER_TD3",
    "ST_Vim_TD3",
    "STV_Patch_TD3",
    "Vim_TD3",
    "ST_Seq_Vim_TD3",
    "STV_Seq_Vim_TD3",
    "PER_ST_Vim_TD3",
    "ST_SVim_TD3",
    "Mamba_TD3",
    "ST_DualVim_TD3",
    "SAC",
    "LSTM_SAC",
    "ST_Vim_SAC",
    "PER_ST_Vim_SAC",
    "PPO",
    "ST_Vim_PPO",
    "PL_ST_Vim_PPO",
    "PL_TD3",
    "PL_PER_TD3",
    "PL_ST_Vim_TD3",
    "PL_SAC",
    "PL_PER_ST_Vim_SAC",
)

ALGORITHM_GROUPS: Dict[str, List[str]] = {
    "all": [
        "TD3",
        "DDPG",
        "PER_TD3",
        "ST_Vim_TD3",
        "STV_Patch_TD3",
        "Vim_TD3",
        "ST_Seq_Vim_TD3",
        "STV_Seq_Vim_TD3",
        "PER_ST_Vim_TD3",
        "ST_SVim_TD3",
        "Mamba_TD3",
        "ST_DualVim_TD3",
        "SAC",
        "LSTM_SAC",
        "ST_Vim_SAC",
        "PER_ST_Vim_SAC",
        "PPO",
        "ST_Vim_PPO",
        "PL_ST_Vim_PPO",
        "PL_TD3",
        "PL_PER_TD3",
        "PL_ST_Vim_TD3",
        "PL_SAC",
        "PL_PER_ST_Vim_SAC",
    ],
    "base": [
        "TD3",
        "DDPG",
        "PER_TD3",
        "SAC",
        "PL_TD3",
        "PL_PER_TD3",
        "PL_SAC",
    ],
    "seq": [
        "ST_Vim_TD3",
        "STV_Patch_TD3",
        "Vim_TD3",
        "ST_Seq_Vim_TD3",
        "STV_Seq_Vim_TD3",
        "PER_ST_Vim_TD3",
        "ST_SVim_TD3",
        "Mamba_TD3",
        "ST_DualVim_TD3",
        "PL_ST_Vim_TD3",
        "PL_ST_Vim_PPO",
        "LSTM_SAC",
        "ST_Vim_SAC",
        "PER_ST_Vim_SAC",
        "PL_PER_ST_Vim_SAC",
    ],
}


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


def normalize_algorithm_name_for_config(algorithm_value: str) -> str:
    raw_value = str(algorithm_value).strip()
    if not raw_value:
        return raw_value

    if "," not in raw_value:
        key = _normalize_key(raw_value)
        if key in ALGORITHM_GROUPS:
            return key
        return to_kebab_algorithm_name(raw_value, upper=False)

    normalized: List[str] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        key = _normalize_key(token)
        if key in ALGORITHM_GROUPS:
            normalized.append(key)
        else:
            normalized.append(to_kebab_algorithm_name(token, upper=False))
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
]
