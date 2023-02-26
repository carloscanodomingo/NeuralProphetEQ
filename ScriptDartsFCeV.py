#!/usr/bin/env python3
"""
%load_ext autoreload
%autoreload 2
"""
import warnings
import os

warnings.filterwarnings("ignore")
import logging
from datetime import datetime, timedelta
from func_timeout import func_set_timeout, FunctionTimedOut

logging.disable(logging.CRITICAL)
import pandas as pd
import numpy as np
import logging
import pandas as pd
from NPw import ConfigEQ
import sys
from NPw_aux import prepare_EQ, ConfigEQ, prepare_ion_data
import click
import torch

MAX_TIMEOUT = 3600 / 4  # MAX TIME 15min
MAX_VALUE = 200
from FCeV import FCeV, FCeVConfig, METRICS

# click_example.py

from DartsFCeV import NLinearDartsFCeVConfig,TransformerDartsFCeVConfig, DartsFCeVConfig,NHITSDartsFCeVConfig, NBEATSDartsFCeVConfig,RNNDartsFCeVConfig,TCNDartsFCeVConfig, TFTDartsFCeVConfig



@click.argument("seed", nargs=1, type=int, default=100)
@click.argument("n_iteration", nargs=1, default=100)
@click.argument("config", nargs=1, default="NO_NAME")
@click.option("--out_path", required=True)
@click.option("--data_path", required=True)
@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.option("--verbose", type=bool, required=False)
@click.option("--epochs", type=int, required=True)
@click.option("--historic_lenght", type=int, required=True)
@click.option("--training_lenght_days", type=int, required=True)
@click.option("--dropout", type=float, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--learning_rate", type=int, required=True)
@click.option("--n_layers", type=int, required=True)
@click.option("--internal_size", type=int, required=True)
@click.option("--use_gpu", type=bool, required=True)
@click.option("--probabilistic", type=bool, required=True)
@click.option("--patience", type=int, required=True)
@click.option("--total_index", default=12, type=int, help="number of total folds")
@click.option("--current_index", default=0, help="number of the current fold")
@click.option(
    "--offset_start", default=365 * 2, type=int, help="number days for the offset"
)
@click.option(
    "--forecast_type",
    type=click.Choice(["folds", "iteration"]),
    help="Set event type",
    required=True,
)
@click.option(
    "--model",
    type=click.Choice(
        ["TCN", "RNN", "NBEATS", "Transformer", "NHITS", "TFT", "NLinear"]
    ),
    required=True,
)


# TCN Model


@click.option("--TCN_dilation_base", type=int, required=False)
@click.option("--TCN_weight_norm", type=bool, required=False)

# RNN Model


@click.option("--RNN_model", type=click.Choice(["RNN", "LSTM", "GRU"]), required=False)

# NBEATS Model


@click.option("--NBEATS_NHITS_num_stacks", type=int, required=False)
@click.option("--NBEATS_NHITS_num_blocks", type=int, required=False)
@click.option("--NBEATS_NHITS_exp_coef", type=int, required=False)

# Transformer Model


@click.option("--Transf_n_head_divisor", type=int, required=False)
@click.option("--Transf_dim_feedforward", type=int, required=False)

# NHITS Model


@click.option("--NHITS_max_pool_1d", type=bool, required=False)

# TFT Model


@click.option("--TFT_num_attention_heads", type=int, required=False)
@click.option("--TFT_full_attention", type=bool, required=False)
@click.option("--TFT_add_relative_index", type=bool, required=False)
@click.option("--TFT_hidden_cont_size", type=int, required=False)

# NLinear Model


@click.option("--NLinear_const_init", type=int, required=False)
def configure(
    verbose,
    epochs,
    historic_lenght,
    training_lenght_days,
    dropout,
    batch_size,
    learning_rate,
    n_layers,
    internal_size,
    use_gpu,
    probabilistic,
    patience,
    total_index,
    current_index,
    offset_start,
    forecast_type,
    data_path,
    out_path,
    model,
    config,
    n_iteration,
    seed,
    tcn_dilation_base=None,
    tcn_weight_norm=None,
    rnn_model=None,
    nbeats_nhits_num_stacks=None,
    nbeats_nhits_num_blocks=None,
    nbeats_nhits_exp_coef=None,
    transf_n_head_divisor=None,
    transf_dim_feedforward=None,
    nhits_max_pool_1d=None,
    tft_num_attention_heads=None,
    tft_full_attention=None,
    tft_add_relative_index=None,
    tft_hidden_cont_size=None,
    nlinear_const_init=None,
):
    # for reproducibility
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ConfigEQ_d = {
        "dist_start": 1000,
        "dist_delta": 3000,
        "mag_start": 4.5,
        "mag_delta": 2,
        "filter": True,
        "drop": ["arc_cos", "arc_sin"],
    }
    config_events = ConfigEQ(**ConfigEQ_d)

    freq = timedelta(minutes=30)
    df_GNSSTEC, df_covariate, df_eq = prepare_ion_data(data_path, "GRK", freq)
    #df_GNSSTEC = pd.read_pickle("df_GNSSTEC.pkl")
    #df_covariate = pd.read_pickle("df_covariate.pkl")
    #df_eq = pd.read_pickle("df_eq.pkl")
    df_regressor = df_GNSSTEC.reset_index()
    df_other = df_covariate
    df_events = prepare_EQ(df_eq, config_events)

    forecast_length = timedelta(hours=24)
    question_mark_length = timedelta(hours=24)
    # Time to take into account to predict
    historic_lenght = timedelta(days=historic_lenght)
    training_lenght = timedelta(days=training_lenght_days)

    if epochs == 0:
        epochs = None
    if model == "TCN":
        TCN_darts_FCeV_config = {
            "dilation_base": tcn_dilation_base,
            "weight_norm": tcn_weight_norm,
        }
        DartsModelConfig = TCNDartsFCeVConfig(**TCN_darts_FCeV_config)
    elif model == "RNN":
        RNN_darts_FCeV_config = {"RNNmodel": rnn_model}
        DartsModelConfig = RNNDartsFCeVConfig(**RNN_darts_FCeV_config)
    elif model == "NBEATS":
        NBEATS_darts_FCeV_config = {
            "num_stacks": nbeats_nhits_num_stacks,
            "num_blocks": nbeats_nhits_num_blocks,
            "expansion_coefficient_dim": nbeats_nhits_exp_coef,
        }
        DartsModelConfig = NBEATSDartsFCeVConfig(**NBEATS_darts_FCeV_config)
    elif model == "Transformer":
        Transformer_darts_FCeV_config = {
            "n_head_divisor": transf_n_head_divisor,
            "dim_feedforward": transf_dim_feedforward,
        }
        DartsModelConfig = TransformerDartsFCeVConfig(**Transformer_darts_FCeV_config)

    elif model == "NHITS":
        NHITS_darts_FCeV_config = {
            "num_stacks": nbeats_nhits_num_stacks,
            "num_blocks": nbeats_nhits_num_blocks,
            "expansion_coefficient_dim": nbeats_nhits_exp_coef,
            "max_pool_1d": nhits_max_pool_1d,
        }
        DartsModelConfig = NHITSDartsFCeVConfig(**NHITS_darts_FCeV_config)

    elif model == "TFT":
        TFT_darts_FCeV_config = {
            "num_attention_heads": tft_num_attention_heads,
            "full_attention": tft_full_attention,
            "add_relative_index": tft_add_relative_index,
            "hidden_continuous_size": tft_hidden_cont_size,
        }
        DartsModelConfig = TFTDartsFCeVConfig(**TFT_darts_FCeV_config)
    elif model == "NLinear":
        NLinear_darts_FCeV_config = {
            "const_init": nlinear_const_init,
        }
        DartsModelConfig = NLinearDartsFCeVConfig(**NLinear_darts_FCeV_config)

    darts_FCev_config = {
        "DartsModelConfig": DartsModelConfig,
        "dropout": dropout,
        "n_epochs": epochs,
        "n_layers": n_layers,
        "internal_size": internal_size,
        "batch_size": batch_size,
        "learning_rate": 1 / 10**learning_rate,
        "use_gpu": use_gpu,
        "event_type": "Non-Binary",
        "patience": patience,
        "seed": seed,
        "probabilistic": probabilistic,
    }

    darts_FCeV_config = DartsFCeVConfig(**darts_FCev_config)
    FCev_config = {
        "freq": freq,
        "forecast_length": forecast_length,
        "question_mark_length": question_mark_length,
        "training_length": training_lenght,
        "verbose": True,
        "input_length": historic_lenght,
    }

    FCev_config = FCeVConfig(**FCev_config)

    synthetic_events = pd.read_pickle(data_path+"synthetic_raw.pkl")

    # Read SR and EQ data
    if verbose == 0:
        sys.stdout = open(os.devnull, "w")
        # sys.stderr = open(os.devnull, "w")
    df_synth = prepare_EQ(synthetic_events, config_events)
    current_fcev = FCeV(
        FCev_config,
        darts_FCeV_config,
        df_GNSSTEC,
        df_covariate,
        df_events,
        out_path,
        df_synth,
    )
    if forecast_type == "folds":
        current_fcev.create_folds(
            k=total_index, offset_lenght=pd.Timedelta(days=offset_start, hours=12)
        )
        try:
            df_fore = process_fold_with_timeout(current_fcev, current_index)
        except FunctionTimedOut:
            print(str(MAX_VALUE) + "\n")
            sys.exit(0)
        cov_result = (
            current_fcev.get_metrics_from_fc(df_fore["BASE"], METRICS.RMSE)
            .mean()
            .mean()
        )
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(str(cov_result) + "\n")
        sys.exit(0)
    elif forecast_type == "iteration":
        offset_start = pd.Timedelta(days=offset_start)
        current_fcev.create_iteration(offset_start, total_index)
        df_fore = current_fcev.process_iteration(current_index)
        current_fcev.save_results(df_fore)
        sys.exit(0)


@func_set_timeout(MAX_TIMEOUT)
def process_fold_with_timeout(fcev_instance, current_index):
    df_fore = fcev_instance.process_fold(current_index)
    return df_fore


if __name__ == "__main__":
    configure()


# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)
