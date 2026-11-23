# Description: Interface stub for the supervised method (ON HOLD, see README.md). Defines the two calls every
#              method adapter provides, so the comparison code can treat all methods alike once it is implemented.
# Date:        2026-11-23


def train(manifest_split, config):
    """Train on the manifest's train/val scenarios. Not implemented yet (method on hold)."""
    raise NotImplementedError("Supervised method is on hold (KIV); see methods/supervised/README.md")


def predict(scenario_path, run_dir):
    """Write params.npz / recon.npz / timing.json / meta.json for one scenario. Not implemented yet."""
    raise NotImplementedError("Supervised method is on hold (KIV); see methods/supervised/README.md")
