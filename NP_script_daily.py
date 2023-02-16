#!/usr/bin/env python

###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################
import pandas as pd
from neuralprophet import  set_log_level
import logging
import pandas as pd
from datetime import datetime, timedelta
import pickle
from NPw import NPw, ConfigEQ, ConfigNPw, ConfigForecast
from dataclasses import dataclass, asdict
from dateutil.relativedelta import *
from NPw_aux import prepare_ion_data
import ast
from pathlib import Path
import sys
import logging
import datetime
import os
import os.path
import re
import subprocess
import sys
import contextlib
set_log_level("ERROR")
logging.disable(logging.CRITICAL)

# click_example.py
import sys
import click

# initialize result to 0
result = 0
from NPw import NPw, ConfigEQ, ConfigNPw, ConfigForecast, METRICS

@click.command(context_settings=dict(
    ignore_unknown_options=True,
allow_extra_args=True
))

@click.option("--verbose",default = 0,type=int, help="verbose")
@click.option("--epochs",default = 0,type=int, help="number of epochs")
# @click.option("--datapath", help="Enter datapath")
@click.option("--historic_lenght", default= 5,help="Enter historic_lenght")
@click.option(
    "--training_lenght_days", default= 365, help="Enter the training_lenght_days", type=int
)
@click.option("--num_hidden_layers", default= 16, help="num_hidden_layers")
@click.option(
    "--seasonal_mode",
    default = "additive",
    type=click.Choice(["multiplicative", "additive"]),
    help="Set seasonla mode seasonality",
)
@click.option("--d_hidden",default = 16, help="d_hidden")
@click.option("--daily_seasonality", default = True, type=bool, help="Set daily seasonality")
@click.option("--yearly_seasonality",default = True,  type=bool, help="Set yearly seasonality")
@click.option("--seasonal_reg", default = 0, type=float, help="Set regularization coefficient for seasonality")

@click.option("--multivariate_season",default = True, type=bool, help="Set Multivariate seasonality")
@click.option("--multivariate_trend",default = True, type=bool, help="Set Multivariate trend")

@click.option(
    "--event_mode",
    default = "multiplicative",
    type=click.Choice(["multiplicative", "additive"]),
    help="Set event type",
)
@click.option("--trend_regularization",default = True, type=bool, help="Trend regularization")

@click.option("--trend_n_changepoint", default = 10,type=int, help="Number of Changepoit")
@click.option("--sparce_ar",default = 0, type=float, help="sparce_ar")
@click.option(
    "--growth",
    default = "off",
    type=click.Choice(["off", "linear"]),
    help="Set event type",
)

@click.option("--total_folds",default = 12,type=int, help="number of total folds")
@click.option("--current_fold",default = 0, help="number of the current fold")


def configure(epochs, verbose, historic_lenght, training_lenght_days, num_hidden_layers, d_hidden, daily_seasonality, yearly_seasonality, seasonal_mode, seasonal_reg, multivariate_season, multivariate_trend, event_mode, trend_regularization, trend_n_changepoint, sparce_ar, growth,total_folds, current_fold):

    datapath = os.environ.get("DATA_PATH")
    freq = timedelta(minutes=30)
    df_GNSSTEC,df_covariate, df_eq = prepare_ion_data(datapath, "GRK", freq)
    df_regressor = df_GNSSTEC.reset_index()
    df_other = df_covariate
    df_events = df_eq

    forecast_length = timedelta(hours=24)
    question_mark_length = timedelta(hours=24)
    # Time to take into account to predict 
    historic_lenght =  timedelta(days=historic_lenght)
    training_lenght = timedelta(days=training_lenght_days)

    if epochs == 0:
        epochs = None
        
    config_npw_d = {
        "forecast_length": forecast_length,
        "question_mark_length": question_mark_length,
        "freq": freq,
        "training_length": training_lenght,
        "historic_lenght": historic_lenght,
        "drop_missing": False,
        "num_hidden_layers": num_hidden_layers,
        "learning_rate": None,
        "d_hidden": d_hidden,
        "yearly_seasonality": yearly_seasonality,
        "daily_seasonality" : daily_seasonality,
        "weekly_seasonality" : False,
        "seasonal_mode": seasonal_mode,
        "seasonal_reg": seasonal_reg,
        "multivariate_season": multivariate_season,
        "multivariate_trend": multivariate_trend,
        "verbose": False,
        "epochs": epochs,
        "use_gpu": False,
        "event_type": "Non-Binary",
        "event_mode": event_mode,
        "normalization": True,
        "trend_regularization": trend_regularization,
        "trend_n_changepoint": trend_n_changepoint,
        "sparce_AR": sparce_ar,
        "loss_func": "Huber",
        "growth": growth
    }

    config_npw = ConfigNPw(**config_npw_d)
    ConfigEQ_d = {
        "dist_start": 1000,
        "dist_delta": 3000,
        "mag_start": 4.5,
        "mag_delta": 2,
        "filter": True,
        "drop": ["arc_cos", "arc_sin"]
    }
    config_events = ConfigEQ(**ConfigEQ_d)

    synthetic_events = pd.read_pickle(datapath + "synthetic.pkl")

    # Read SR and EQ data
    if verbose == 0:    
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    print("This won't be printed.")
    NPw_o = NPw(config_npw, df_regressor,df_other, config_events, df_events, synthetic_events)
    NPw_o.get_folds(k = total_folds)
    df_forecast, df_uncer = NPw_o.process_fold(1)
    from NPw import METRICS
    cov_result = NPw_o.get_metrics_from_fc(df_forecast["BASE"], METRICS.CoV)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(str(cov_result) + '\n')
    sys.exit(0)

if __name__ == "__main__":
    configure()

# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)
