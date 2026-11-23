# Description: Single source of truth for the repository root used by the pila package.
#              Relative paths inside configs (e.g. "configs/mogi_paras_symdV.json", "synthetic/marapi_grid/...")
#              are resolved against REPO_ROOT, exactly as the legacy modules resolved them against their PARENT_DIR
#              (the repo root). Keeping this identical is what lets legacy configs run unchanged on the new code.
# Date:        2026-11-23
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))     # .../PILA/pila
REPO_ROOT = os.path.dirname(PACKAGE_DIR)                       # .../PILA


# ----------------------------------------------------------------------------------------------------------------
# Machine-specific data roots (paths.yaml)
# ----------------------------------------------------------------------------------------------------------------
# Configs may contain ${NAME} tokens (e.g. "${MINTPY_ROOT}/Etna/S1_TA44/mintpy/geo/geo_timeseries.h5") instead of
# hard-coded absolute paths. NAME is looked up in this order:
#   1. environment variable PILA_<NAME>   (handy on another machine / in a PBS job)
#   2. REPO_ROOT/paths.yaml               (local, gitignored; copy paths.example.yaml and edit)
# A config without tokens is returned unchanged, so every legacy config still runs as before.
import re as _re

_TOKEN = _re.compile(r'\$\{([A-Z][A-Z0-9_]*)\}')
PATHS_FILE = os.path.join(REPO_ROOT, 'paths.yaml')


def load_paths(paths_file=PATHS_FILE):
    """Read paths.yaml into a dict {NAME: path}; empty dict if the file does not exist."""
    if not os.path.exists(paths_file):
        return {}
    import yaml
    with open(paths_file) as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{paths_file} must be a mapping NAME: path, got {type(loaded).__name__}")
    return {str(key): str(value) for key, value in loaded.items()}


def resolve_tokens(obj, paths=None):
    """
    Recursively replace ${NAME} tokens in every string of a config (dict / list / str).

    Inputs:  obj -- config fragment; paths -- {NAME: path} (default: load_paths()).
    Output:  a new object with tokens expanded. Raises KeyError naming the token and where to define it if a
             token is unknown, so a missing data root fails loudly instead of producing a wrong path.
    """
    if paths is None:
        paths = load_paths()

    def expand(text):
        def lookup(match):
            name = match.group(1)
            value = os.environ.get(f'PILA_{name}', paths.get(name))
            if value is None:
                raise KeyError(f"Unknown path token ${{{name}}}: define {name} in {PATHS_FILE} "
                               f"(see paths.example.yaml) or set environment variable PILA_{name}")
            return value
        return _TOKEN.sub(lookup, text)

    if isinstance(obj, dict):
        return {key: resolve_tokens(value, paths) for key, value in obj.items()}
    if isinstance(obj, list):
        return [resolve_tokens(value, paths) for value in obj]
    if isinstance(obj, str) and '${' in obj:
        return expand(obj)
    return obj
