import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from pila.paths import REPO_ROOT as PARENT_DIR  # repo root (legacy name kept: PARENT_DIR)

from .model_phys_smpl import PHYS_VAE_SMPL