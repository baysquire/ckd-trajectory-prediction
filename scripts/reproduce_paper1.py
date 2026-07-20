"""Run the Paper 1 evals in order.

Put train/val/test.csv where src/paths.py can find them, then:

    python scripts/reproduce_paper1.py
"""
import subprocess
import sys
import time


STEPS = [
    ("leakage audit", [sys.executable, "-m", "src.eval.leakage_audit"]),
    ("14k baselines", [sys.executable, "-m", "src.eval.run_baselines_14k"]),
    ("rapid-progressor labels", [sys.executable, "-m", "src.tasks.rapid_progressor"]),
    ("rapid-progressor training", [sys.executable, "-m", "src.tasks.train_rapid_progressor"]),
    ("multi-horizon eval", [sys.executable, "-m", "src.eval.multihorizon"]),
    ("survival analysis", [sys.executable, "-m", "src.tasks.survival"]),
]


def main():
    print("running paper 1 evals")
    failed = []
    for name, cmd in STEPS:
        print(f"\n--- {name} ---")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=".")
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"failed: {name}")
            failed.append(name)
        else:
            print(f"done ({elapsed:.0f}s)")

    if failed:
        print("failed steps:", ", ".join(failed))
        sys.exit(1)
    print("all done")


if __name__ == "__main__":
    main()
