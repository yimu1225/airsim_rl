__version__ = "1.1.1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba

# The UAV project uses the vision/SSM modules only.  Import the optional
# language-model API lazily so incompatible Transformers generation helpers do
# not prevent the core Mamba mixer from loading.
def __getattr__(name):
    if name == "MambaLMHeadModel":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        return MambaLMHeadModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
