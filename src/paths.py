"""Where the data lives.

Looks in a few usual places so the same scripts work in the
research folder and in the smaller public repo.
"""
import os


def repo_root():
    # src/paths.py -> src -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_processed():
    """Folder with train.csv / val.csv / test.csv."""
    root = repo_root()
    env = os.environ.get("NEPHRO_DATA_PATH")
    if env and os.path.isdir(env):
        return env

    candidates = [
        os.path.join(root, "data", "processed"),
        os.path.join(root, "..", "Medicine_Results", "ckd_tft_results", "processed"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(os.path.join(path, "test.csv")):
            return path
    # default for the public repo layout
    return os.path.abspath(candidates[0])


def find_models_dir():
    root = repo_root()
    env = os.environ.get("NEPHRO_CKPT_DIR")
    if env:
        return env
    proc = find_processed()
    # .../processed -> .../models
    sibling = os.path.abspath(os.path.join(proc, "..", "models"))
    if os.path.isdir(os.path.dirname(sibling)):
        return sibling
    return os.path.join(root, "results", "models")


def find_reports_dir():
    root = repo_root()
    proc = find_processed()
    # Medicine_Results/ckd_tft_results/processed -> Medicine_Results/reports
    med_root = os.path.abspath(os.path.join(proc, "..", ".."))
    if os.path.basename(med_root) == "Medicine_Results" or os.path.isdir(
        os.path.join(med_root, "reports")
    ):
        return os.path.join(med_root, "reports")
    return os.path.join(root, "results", "reports")
