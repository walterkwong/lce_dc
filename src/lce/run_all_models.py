import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastai.metrics import APScoreBinary, BrierScore, F1Score, RocAucBinary, accuracy

from lce.models import dl_models, logistic_regression_model, xgboost_model
from lce.models.cnn import run_cnn
from lce.utils import create_filename


def _safe_roc_auc():
    try:
        return RocAucBinary()
    except ValueError:
        return 0


def all_run(
    data_dir: Path,
    output_dir: Path,
    device,
    data_name: str,
    data_size: int,
    model_name: str,
    X_trainvalid: np.ndarray,
    Y_trainvalid: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    stoc: float,
    randnum_train: int,
    randnum_split: int,
    randnum_stoc: int,
    epochs: int = 50,
    num_optuna_trials: int = 5,
    run_hyper_opt: bool = False,
    run_feature_importance: bool = False,
    folds: int = 5,
) -> pd.DataFrame:
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    save_name = create_filename(
        "model_output", data_name, model_name,
        stoc=int(stoc * 100),
        randsp=int(randnum_split),
        randtr=int(randnum_train),
        hype=run_hyper_opt,
        time=timestr,
    )
    output_file = (output_dir / save_name).with_suffix(".csv")

    # Evaluation metrics tracked during tsai training
    tsai_metrics = [
        accuracy,
        F1Score(),
        _safe_roc_auc(),
        BrierScore(),
        APScoreBinary(),
    ]

    common_kwargs = dict(
        model_name=model_name,
        output_file=output_file,
        output_dir=output_dir,
        data_name=data_name,
        stoc=stoc,
        X_trainvalid=X_trainvalid,
        Y_trainvalid=Y_trainvalid,
        X_test=X_test,
        Y_test=Y_test,
        run_hyper_opt=run_hyper_opt,
        run_feature_importance=run_feature_importance,
        epochs=epochs,
        num_optuna_trials=num_optuna_trials,
        randnum_split=randnum_split,
        randnum_train=randnum_train,
        randnum_stoc=randnum_stoc,
        folds=folds,
        device=device,
        save_name=save_name,
        metrics=tsai_metrics,
        timestr=timestr,
    )

    if model_name == "LR":
        return logistic_regression_model.run_logistic_regression(**common_kwargs)

    if model_name == "XGBoost":
        return xgboost_model.run_xgboost(**common_kwargs)

    if model_name == "CNN":
        # CNN has its own self-contained loop
        run_cnn(
            data_dir=data_dir,
            output_csv=output_dir / "cnn_metrics.csv",
            data_names=[data_name],
            size_start=data_size,
            size_stop=data_size,
            size_step=1,
        )
        return pd.DataFrame()

    # Deep learning models
    return dl_models.run_dl_models(**common_kwargs, data_size=data_size)
