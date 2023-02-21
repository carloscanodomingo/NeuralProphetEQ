#!/usr/bin/env python3
"""
%load_ext autoreload
%autoreload 2
"""
from NPw_aux import prepare_ion_data
import warnings
import os
warnings.filterwarnings("ignore")
import logging
datapath = os.environ.get("DATA_PATH")
if datapath == None:
    raise KeyError("Enviromental variable data path has to be declared")
from datetime import datetime, timedelta
freq = timedelta(minutes=30)
logging.disable(logging.CRITICAL)

import pandas as pd
import numpy as np
import logging
import pandas as pd

from NPw import  ConfigEQ

import sys
import numpy as np
import pandas as pd

import pandas as pd
from NPw_aux import prepare_EQ, ConfigEQ

from FCeV import METRICS


from FCeV import FCeV, FCeVConfig
# click_example.py
import sys
import click

from DartsFCeV import DartsFCeVConfig, TCNDartsFCeVConfig

from FCeV import FCeV, FCeVConfig


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.option("--verbose", default=False, type=bool, help="verbose")
@click.option("--epochs", default=300, type=int, help="number of epochs")
# @click.option("--datapath", help="Enter datapath")
@click.option("--historic_lenght", default=5, help="Enter historic_lenght")
@click.option(
    "--training_lenght_days",
    default=365,
    help="Enter the training_lenght_days",
    type=int,
)

        
        
        
@click.option("--dropout", default=0.1, help="dropout")
@click.option(
    "--batch_size",
    default=500,
    help="batch_size",
)
@click.option("--learning_rate", default=1e-3, help="learning_rate")


        
@click.option(
    "--dilation_base", default=2, type=int, help="dilation_base"
)
@click.option(
    "--weight_norm", default=True, type=bool, help="weight_norm"
)
@click.option(
    "--kernel_size",
    default=16,
    type=int,
    help="kernel_size",
)
@click.option(
    "--num_filter",
    default=3,
    type=int,
    help="num_filter",
)
@click.option(
    "--patience",
    default=5,
    type=int,
    help="patience",
)


@click.option("--total_folds", default=12, type=int, help="number of total folds")
@click.option("--current_fold", default=0, help="number of the current fold")
def configure(
    epochs,
    verbose,
    historic_lenght,
    training_lenght_days,
    dropout,
    batch_size,
    learning_rate,
    dilation_base,
    weight_norm,
    kernel_size,
    num_filter,
    total_folds,
    current_fold,
    patience,
):
    
    
    ConfigEQ_d = {
        "dist_start": 1000,
        "dist_delta": 3000,
        "mag_start": 4.5,
        "mag_delta": 2,
        "filter": True,
        "drop": ["arc_cos", "arc_sin"],
    }
    config_events = ConfigEQ(**ConfigEQ_d)




    datapath = os.environ.get("DATA_PATH")
    freq = timedelta(minutes=30)

    # df_GNSSTEC = pd.read_pickle("df_GNSSTEC.pkl")
    # df_covariate = pd.read_pickle("df_covariate.pkl")
    # df_eq = pd.read_pickle("df_eq.pkl")
    df_GNSSTEC, df_covariate, df_eq = prepare_ion_data(datapath, "GRK", freq)
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

    TCN_darts_FCeV_config = {
        "dilation_base": dilation_base,
        "weight_norm": weight_norm,
        "kernel_size": kernel_size,
        "num_filter": num_filter}

    TCN_darts_FCeV_config = TCNDartsFCeVConfig(**TCN_darts_FCeV_config)

    darts_FCev_config = {
        "DartsModelConfig": TCN_darts_FCeV_config,
        "dropout":dropout,
        "n_epochs":epochs,
        "batch_size":batch_size,
        "learning_rate": 1/10**learning_rate,
        "use_gpu": False,
        "event_type": "Non-Binary",
        "patience": patience    }
    darts_FCeV_config = DartsFCeVConfig(**darts_FCev_config)
    FCev_config = {
        "freq": freq,
        "forecast_length": forecast_length,
        "question_mark_length": question_mark_length,
        "training_length": training_lenght,
        "verbose": True,
        "input_length": historic_lenght
    }

    FCev_config = FCeVConfig(**FCev_config)

    synthetic_events = pd.read_pickle(datapath + "synthetic_raw.pkl")

    # Read SR and EQ data
    if verbose == 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    df_synth = prepare_EQ(synthetic_events, config_events)  
    current_fcev = FCeV(FCev_config, darts_FCeV_config, df_GNSSTEC, df_covariate, pd.DataFrame(),df_events, df_synth)
    current_fcev.create_folds(k=total_folds, offset_lenght=pd.Timedelta(days=180))
    df_fore, df_uncer = current_fcev.process_fold(current_fold)

    cov_result = current_fcev.get_metrics_from_fc(df_fore["BASE"], METRICS.CoV)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(str(cov_result) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    configure()


# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)