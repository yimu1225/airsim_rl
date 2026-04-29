from __future__ import annotations

from typing import Dict, List, Tuple

_CANONICAL_ALGORITHMS: Tuple[str, ...] = (
    "td3",
    "ddpg",
    "per_td3",
    "st_vim_td3",
    "stv_patch_td3",
    "vim_td3",
    "st_seq_vim_td3",
    "stv_seq_vim_td3",
    "per_st_vim_td3",
    "st_svim_td3",
    "mamba_td3",
    "st_dualvim_td3",
    "sac",
    "lstm_sac",
    "st_vim_sac",
    "per_st_vim_sac",
    "ppo",
    "st_vim_ppo",
    "td3_asym",
    "per_td3_asym",
    "st_vim_td3_asym",
)

ALGORITHM_GROUPS: Dict[str, List[str]] = {
    "all": [
        "td3",
        "ddpg",
        "per_td3",
        "st_vim_td3",
        "stv_patch_td3",
        "vim_td3",
        "st_seq_vim_td3",
        "stv_seq_vim_td3",
        "per_st_vim_td3",
        "st_svim_td3",
        "mamba_td3",
        "st_dualvim_td3",
        "sac",
        "lstm_sac",
        "st_vim_sac",
        "per_st_vim_sac",
        "td3_asym",
        "per_td3_asym",
        "st_vim_td3_asym",
    ],
    "base": [
        "td3",
        "ddpg",
        "per_td3",
        "sac",
        "td3_asym",
        "per_td3_asym",
    ],
    "seq": [
        "st_vim_td3",
        "stv_patch_td3",
        "vim_td3",
        "st_seq_vim_td3",
        "stv_seq_vim_td3",
        "per_st_vim_td3",
        "st_svim_td3",
        "mamba_td3",
        "st_dualvim_td3",
        "st_vim_td3_asym",
        "lstm_sac",
        "st_vim_sac",
        "per_st_vim_sac",
    ],
}


def _normalize_key(name: str) -> str:
    return str(name).strip().lower()


_ALIAS_TO_CANONICAL: Dict[str, str] = {}


for _algo in _CANONICAL_ALGORITHMS:
    _ALIAS_TO_CANONICAL[_normalize_key(_algo)] = _algo
    _ALIAS_TO_CANONICAL[_normalize_key(_algo.replace("_", "-"))] = _algo


_EXTRA_ALIASES = {
    "st-vimtd3": "st_vim_td3",
    "st_vimtd3": "st_vim_td3",
    "stvimtd3": "st_vim_td3",
    "stvpatchtd3": "stv_patch_td3",
    "vim-td3": "vim_td3",
    "vim_td3": "vim_td3",
    "st-seqvimtd3": "st_seq_vim_td3",
    "stseqvimtd3": "st_seq_vim_td3",
    "stv-seqvimtd3": "stv_seq_vim_td3",
    "stvseqvimtd3": "stv_seq_vim_td3",
    "per-st-vimtd3": "per_st_vim_td3",
    "per_st_vimtd3": "per_st_vim_td3",
    "st-svimtd3": "st_svim_td3",
    "st_svimtd3": "st_svim_td3",
    "st-dualvimtd3": "st_dualvim_td3",
    "stdualvimtd3": "st_dualvim_td3",
    "st-vimtd3-asym": "st_vim_td3_asym",
    "st-vimtd3_asym": "st_vim_td3_asym",
    "st_vimtd3-asym": "st_vim_td3_asym",
    "st_vimtd3_asym": "st_vim_td3_asym",
    "st-vimsac": "st_vim_sac",
    "st_vimsac": "st_vim_sac",
    "stvimsac": "st_vim_sac",
    "lstm-sac": "lstm_sac",
    "lstm_sac": "lstm_sac",
    "lstmsac": "lstm_sac",
    "per-st-vimsac": "per_st_vim_sac",
    "per_st_vimsac": "per_st_vim_sac",
    "perstvimsac": "per_st_vim_sac",
    "st-vimppo": "st_vim_ppo",
    "st_vimppo": "st_vim_ppo",
    "stvimppo": "st_vim_ppo",
}
for _alias, _canonical in _EXTRA_ALIASES.items():
    _ALIAS_TO_CANONICAL[_normalize_key(_alias)] = _canonical


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
    if fallback in _CANONICAL_ALGORITHMS:
        return fallback

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
        return to_kebab_algorithm_name(raw_value, upper=True)

    normalized: List[str] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        key = _normalize_key(token)
        if key in ALGORITHM_GROUPS:
            normalized.append(key)
        else:
            normalized.append(to_kebab_algorithm_name(token, upper=True))
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
