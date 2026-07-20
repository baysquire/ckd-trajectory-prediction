"""Run the Paper 1 pipeline.

If data/processed/ already contains train.csv, val.csv, test.csv, the
preprocessing step is skipped. Otherwise, it runs the full pipeline from
the raw MIMIC extract.

Usage:
    python scripts/reproduce_paper1.py
"""
import os
import subprocess
import sys
import time


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _processed_exists():
    proc = os.path.join(_repo_root(), "data", "processed")
    env = os.environ.get("NEPHRO_DATA_PATH", proc)
    return os.path.isfile(os.path.join(env, "test.csv"))


STEPS = [
    ("leakage audit", [sys.executable, "-m", "src.eval.leakage_audit"]),
    ("14k baselines", [sys.executable, "-m", "src.eval.run_baselines_14k"]),
    ("rapid-progressor labels", [sys.executable, "-m", "src.tasks.rapid_progressor"]),
    ("rapid-progressor training", [sys.executable, "-m", "src.tasks.train_rapid_progressor"]),
    ("multi-horizon eval", [sys.executable, "-m", "src.eval.multihorizon"]),
    ("survival analysis", [sys.executable, "-m", "src.tasks.survival"]),
]


def main():
    print("Paper 1 reproduction pipeline")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # optional: run preprocessing if CSVs don't exist yet
    if not _processed_exists():
        raw_path = os.path.join(_repo_root(), "data", "raw", "mimic_ckd_cohort.csv")
        if os.path.isfile(raw_path):
            print("\n--- preprocessing (raw -> processed) ---")
            result = subprocess.run(
                [sys.executable, "-m", "src.data.preprocess"], cwd=_repo_root()
            )
            if result.returncode != 0:
                print("preprocessing failed")
                sys.exit(1)
        else:
            print(f"\nNo processed CSVs found and no raw extract at {raw_path}.")
            print("See docs/DATA.md for instructions.")
            sys.exit(1)

    failed = []
    for name, cmd in STEPS:
        print(f"\n--- {name} ---")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=_repo_root())
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"failed: {name}")
            failed.append(name)
        else:
            print(f"done ({elapsed:.0f}s)")

    if failed:
        print("failed steps:", ", ".join(failed))
        sys.exit(1)
    print("\nall steps completed successfully")


if __name__ == "__main__":
    main()
