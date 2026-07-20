"""Where the data lives.

Looks in a few usual places so the same scripts work when
the data directory is set via environment variable or placed
in the default location.
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

    default = os.path.join(root, "data", "processed")
    return os.path.abspath(default)


def find_models_dir():
    root = repo_root()
    env = os.environ.get("NEPHRO_CKPT_DIR")
    if env:
        return env
    return os.path.join(root, "results", "models")


def find_reports_dir():
    root = repo_root()
    return os.path.join(root, "results", "reports")
