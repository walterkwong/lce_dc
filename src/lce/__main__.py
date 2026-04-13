"""
entry point

This project is a final year project at University of Bristol,
supervised by Dr Juliette Unwin and Helen Coupland.
The codebase is built upon research and resources from them, extending
Coupland et al. (2023): https://www.researchsquare.com/article/rs-3601343/v1, https://github.com/hcoupland/LifeCourseDLSimulationStudy
"""

from pathlib import Path

import click
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from lce import load_data, run_all_models
from lce.config import ALL_DATA_NAMES, MODEL_NAMES, SPARSE_LCPS

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_SIZE_STARTS = {"default": 2000, "sparse": 4000}


@click.command()
@click.option("--data-dir",   default="data/",          show_default=True,
              type=click.Path(exists=True),              help="Directory containing .npy data files.")
@click.option("--output-dir", default="output/results/", show_default=True,
              type=click.Path(),                         help="Directory for all model outputs.")
@click.option("--model-name", required=True, type=click.Choice(MODEL_NAMES),
              help="Model architecture to train.")
@click.option("--stochasticity", default=0.0, type=float, show_default=True,
              help="Label-noise proportion (0–1). 0 = no noise.")
@click.option("--train-seed",      default=42, type=int, show_default=True,
              help="Seed for weight initialisation and optimisation.")
@click.option("--data-split-seed", default=42, type=int, show_default=True,
              help="Seed for train/test splitting.")
@click.option("--stochasticity-seed", default=42, type=int, show_default=True,
              help="Seed controlling which labels are flipped.")
@click.option("--epochs",           default=50,  type=int, show_default=True,
              help="Maximum training epochs (deep learning models).")
@click.option("--num-optuna-trials", default=5,  type=int, show_default=True,
              help="Number of Optuna hyperparameter trials.")
@click.option("--run-hyper-opt",      is_flag=True, default=False,
              help="Enable Optuna hyperparameter optimisation.")
@click.option("--run-feature-importance", is_flag=True, default=False,
              help="Compute SHAP / PFI after training.")
@click.option("--folds", default=2, type=int, show_default=True,
              help="Number of stratified CV folds (used during hyperopt).")
@click.option("--device", default=DEFAULT_DEVICE, show_default=True,
              help="Compute device: 'cpu', 'cuda', or a GPU index.")
def main(
    data_dir,
    output_dir,
    model_name,
    stochasticity,
    train_seed,
    data_split_seed,
    stochasticity_seed,
    epochs,
    num_optuna_trials,
    run_hyper_opt,
    run_feature_importance,
    folds,
    device,
):
    device     = torch.device(device)
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_data.set_random_seeds(train_seed)

    for data_name in ALL_DATA_NAMES:
        print(f"\n{'=' * 60}")
        print(f"  LCP: {data_name}  |  Model: {model_name}")
        print(f"{'=' * 60}")

        X_raw = np.load(data_dir / "data_X.npy").astype(np.float32)
        Y_raw = np.squeeze(np.load(data_dir / f"data_{data_name}_Y.npy"))
        y_raw = Y_raw[:, -1]  # use the final time-point label

        np.random.seed(0)
        _, test_mask = train_test_split(
            np.arange(len(y_raw)), test_size=5000, stratify=y_raw
        )
        X_test = X_raw[test_mask]
        Y_test = y_raw[test_mask]

        X_pool = np.delete(X_raw, test_mask, axis=0)
        y_pool = np.delete(y_raw, test_mask, axis=0)

        size_start = _SIZE_STARTS["sparse" if data_name in SPARSE_LCPS else "default"]

        for data_size in range(size_start, 20_001, 2_000):
            print(f"\n  -- data_size={data_size} --")

            _, mask = train_test_split(
                np.arange(len(y_pool)), test_size=data_size, stratify=y_pool
            )
            X_trainvalid = X_pool[mask]
            Y_trainvalid = y_pool[mask]

            if stochasticity > 0:
                Y_trainvalid = load_data.add_noise(
                    Y_trainvalid, stoc=stochasticity, randnum=stochasticity_seed
                )

            run_all_models.all_run(
                data_dir=data_dir,
                output_dir=output_dir,
                device=device,
                data_name=data_name,
                data_size=data_size,
                model_name=model_name,
                X_trainvalid=X_trainvalid,
                Y_trainvalid=Y_trainvalid,
                X_test=X_test,
                Y_test=Y_test,
                stoc=stochasticity,
                randnum_train=train_seed,
                randnum_split=data_split_seed,
                randnum_stoc=stochasticity_seed,
                epochs=epochs,
                num_optuna_trials=num_optuna_trials,
                run_hyper_opt=run_hyper_opt,
                run_feature_importance=run_feature_importance,
                folds=folds,
            )


if __name__ == "__main__":
    main()
