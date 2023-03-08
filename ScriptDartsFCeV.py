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
from multiprocessing import Process, Queue
from NPw import ConfigEQ
import sys
from NPw_aux import prepare_EQ, ConfigEQ, prepare_ion_data
import click
import torch
import signal
MAX_TIMEOUT = 3600/ 3  # MAX TIME 20min
MAX_VALUE ="Inf" 
from FCeV import FCeV, FCeVConfig, METRICS
import time
import math
# click_example.py

from DartsFCeV import NLinearDartsFCeVConfig,TransformerDartsFCeVConfig, DartsFCeVConfig,NHITSDartsFCeVConfig, NBEATSDartsFCeVConfig,RNNDartsFCeVConfig,TCNDartsFCeVConfig, TFTDartsFCeVConfig



@click.argument("seed", nargs=1, type=int, default=100)
@click.argument("n_iteration", nargs=1, default=100)
@click.argument("config", nargs=1, default="NO_NAME")
@click.option("--out_path", required=True)
@click.option("--data_path", required=True)
@click.option("--log_path", default = None)
@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.option("--verbose", type=bool, required=False)
@click.option("--epochs", type=int, required=True)
@click.option("--historic_lenght", type=int, required=True)
@click.option("--forecast_lenght_hours", type=int, required=True)
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
@click.option(
        "--simulation_scenario",
        type = click.Choice(
            ["TEC", "SALES", "TEC_constant", "TEC_EQ"]
            ),
        default = "TEC_EQ"
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
    log_path,
    verbose,
    epochs,
    historic_lenght,
    forecast_lenght_hours,
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
    simulation_scenario,
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
        # Read SR and EQ data
    config_synthetic = "events"
    if verbose == 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    if  simulation_scenario == "TEC" or simulation_scenario == "TEC_constant" or simulation_scenario == "TEC_EQ":
        ConfigEQ_d = {
        "dist_start": 100,
        "dist_delta": 6000,
        "mag_start": 4.5,
        "mag_delta": 2,
        "filter": 1,
        "drop": ["arc_cos", "arc_sin","depth", "mag", "dist"],
        }
        config_events = ConfigEQ(**ConfigEQ_d)

        freq = timedelta(minutes=30)
        df_GNSSTEC, df_covariate, df_eq = prepare_ion_data(data_path, "GRK", freq)
        
        
        df_signal = df_GNSSTEC
        df_covariates = df_covariate
        
        forecast_length = timedelta(hours=forecast_lenght_hours)
        question_mark_length = timedelta(hours=forecast_lenght_hours)
        # Time to take into account to predict
        historic_lenght = timedelta(days=historic_lenght)
        training_lenght = timedelta(days=training_lenght_days)

        if epochs == 0:
            epochs = None

        if simulation_scenario == "TEC_EQ":
            synthetic_events = pd.read_pickle(data_path+"synthetic_raw.pkl")
            df_events = prepare_EQ(df_eq, config_events)
            df_synth = prepare_EQ(synthetic_events, config_events)
            values = np.arange(0,16,3)
            synthetic_events = pd.DataFrame(values, columns= ["pr"])
            df_all = pd.DataFrame()
            config_synthetic = "constant"
            for ix, eq in df_events.sort_values(by=["pr"]).iterrows():
                event = pd.DataFrame(eq).T
                start_time = event.index[0]
                end_time = start_time + forecast_length - freq
                index = pd.date_range(start_time, end_time, freq = freq)
                df = pd.DataFrame(np.repeat(event.values, len(index)), columns = df_events.columns, index=index)
                df_all = df.combine_first(df_all)
            
        elif simulation_scenario == "TEC_constant":
            df_synth = pd.DataFrame(np.arange(60, 85,1), columns= ["f107"])
            df_events = pd.DataFrame(df_covariate["f107"])
            df_covariate = df_covariate.drop("f107", axis = 1)
            config_synthetic = "constant"
        date_start = pd.Timestamp(2018, 1, 1, 12)

    elif simulation_scenario == "SALES":
        datapath_sales =  data_path + "kaggle/store-sales-time-series-forecasting/"
        df_train =  pd.read_csv(datapath_sales + "train.csv", parse_dates=["date"]).rename(columns = {"date":"ds"})
        df_test =  pd.read_csv(datapath_sales + "test.csv", parse_dates=["date"]).rename(columns = {"date":"ds"})
        df_train_selected = df_train[(df_train["store_nbr"] == 54) & ((df_train["family"] == "AUTOMOTIVE")) ]
        df_train_selected = df_train_selected.reset_index().drop(["store_nbr", 'family', "id", "index"], axis = 1).set_index("ds")
        df_signal = df_train_selected.drop("onpromotion", axis =1) #["sales"]
        df_events = df_train_selected[df_train_selected["onpromotion"] > 0].drop("sales", axis = 1)
        df_events["onpromotion"] = (df_events["onpromotion"] > 0)
        df_events["onpromotion"] = 1
        df_cov1 = df_train[(df_train["store_nbr"] == 54) & ((df_train["family"] == "BREAD/BAKERY")) ]
        df_cov1 = df_cov1.reset_index().drop(["store_nbr", 'family', "id", "index", "onpromotion"], axis = 1).set_index("ds")
        df_cov2 = df_train[(df_train["store_nbr"] == 1) & ((df_train["family"] == "AUTOMOTIVE")) ]
        df_cov2= df_cov2.reset_index().drop(["store_nbr", 'family', "id", "index", "onpromotion"], axis = 1).set_index("ds")

        df_covariates = pd.concat([df_cov1, df_cov2], axis = 1)
        df_covariates.columns =  ["bread_54", "auto_1"]
        df_synth = pd.DataFrame([1], columns = ["onpromotion"])
        forecast_length = timedelta(hours=24 * 4)
        question_mark_length = timedelta(hours=24 * 4)
        # Time to take into account to predict 
        historic_lenght =  timedelta(days=historic_lenght)
        training_lenght = timedelta(days=training_lenght_days)
        freq = pd.Timedelta(days=1)
        date_start = pd.Timestamp(2017, 1, 1)
    else:
        raise KeyError("Simulation type not implemented")
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
    else:
        raise ValueError("Not Valid Model")
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
        "config_synthetic": config_synthetic
    }

    darts_FCeV_config = DartsFCeVConfig(**darts_FCev_config)
    FCev_config = {
        "freq": freq,
        "forecast_length": forecast_length,
        "question_mark_length": question_mark_length,
        "training_length": training_lenght,
        "verbose": verbose,
        "input_length": historic_lenght,
    }

    FCev_config = FCeVConfig(**FCev_config)

    

    # Read SR and EQ data
    if log_path is None:
        if verbose == 0:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
    else:
        sys.stdout = open(log_path, 'w')
        sys.stderr = sys.stdout
    
    current_fcev = FCeV(
        FCev_config,
        darts_FCeV_config,
        df_signal,
        df_covariates,
        df_events,
        out_path,
        df_synth,
    )



    if forecast_type == "folds":
        current_fcev.create_folds(date_start, n_iteration=total_index)
        queue = Queue()
        program = Process(target= process_fold_with_timeout, args=(current_fcev, current_index, queue))
        program.start()
        try:
            df_fore = queue.get(timeout = MAX_TIMEOUT)
        except: 
            queue.close()
            program.terminate()
            del queue
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(str(MAX_VALUE) + "\n", flush=True)
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            sys.exit(0)

        program.join(timeout = 1)
        program.terminate()
        if df_fore is not None:
            queue.close()
            del queue
            cov_result = (
                    current_fcev.get_metrics_from_fc(df_fore["current"], df_fore["BASE"], METRICS.CoV)
            )
            if math.isnan(cov_results):
                cov_results = MAX_VALUE
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(str(cov_result) + "\n")
            sys.exit(0)
        else:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(str(MAX_VALUE) + "\n", flush=True)
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            sys.exit(0)
            pass

    elif forecast_type == "iteration":
        current_fcev.create_iteration(date_start, total_index)
        df_fore = current_fcev.process_iteration(current_index)
        current_fcev.save_results(df_fore)
        sys.exit(0)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    return 0

 
def process_fold_with_timeout(fcev_instance, current_index, queue):
    if fcev_instance.FCeV_model.FCeV_config.verbose == 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    try:
        df_fore = fcev_instance.process_fold(current_index)
    except:
        queue.put(None)
    queue.put(df_fore)
    return 0

def signal_handler(sig, frame):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(str(201) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    handler_sig = [signal.SIGABRT, signal.SIGSEGV]
    for sig in handler_sig:
        try:
            signal.signal(sig, signal_handler)
        except OSError:
            continue
    configure()


# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)