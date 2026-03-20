import os

# Ubuntu native defaults (no WSL2 path assumptions).
# You can override each path via environment variables:
#   AIRLEARNING_UE4_ROOT
#   AIRLEARNING_JSON_FILE
#   AIRLEARNING_UPROJECT
#   UNREAL_EDITOR_PATH
_default_root = os.environ.get("AIRLEARNING_UE4_ROOT", "/home/admin/DRL_Project/airlearning-ue4")

# Valid path to a json file for game configuration
json_file_addr = os.environ.get(
    "AIRLEARNING_JSON_FILE",
    os.path.join(_default_root, "Content", "JsonFiles", "EnvGenConfig.json"),
)

# Path to the game executable or configuration file
game_file = os.environ.get(
    "AIRLEARNING_UPROJECT",
    os.path.join(_default_root, "AirLearning.uproject"),
)

# Directory shared with the unreal host
unreal_host_shared_dir = ""

# Path to the Unreal Engine executable (Linux UE editor path).
unreal_exe_path = os.environ.get(
    "UNREAL_EDITOR_PATH",
    "/home/admin/DRL_Project/UnrealEngine-staging-4.18/Engine/Binaries/Linux/UE4Editor",
)
