"""
app.py
---------------------------------
Simple command-line entrypoint for the assignment project.

Usage:
    python app.py train    # Runs training pipeline and saves models/metrics

This file is intentionally minimal — the heavy lifting lives in `train.py`.
"""

import argparse
from train import train_and_save_models


def main():
    parser = argparse.ArgumentParser(description="Assignment 2 - Model training CLI")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument("--out", default="model", help="Directory to save models and metrics")
    args = parser.parse_args()

    if args.command == "train":
        df_metrics = train_and_save_models(save_dir=args.out)
        if not df_metrics.empty:
            print("Training completed. Metrics summary:")
            print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
