from aux_function_SR import read_data, get_eq_filtered, SR_SENSORS
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_log_level, save, load
import logging
import pandas as pd
from datetime import datetime, timedelta
from NPw import NPw, ConfigEQ, ConfigNPw, ConfigForecast
from dateutil.relativedelta import *

# click_example.py
import sys
import click

# initialize result to 0
result = 0


@click.command()
@click.option("--epochs", default=0, help="Enter the number of epochs")
@click.option("--day", default=1, help="Enter the first day to forecast from 2017-01-01", type=int)
@click.option("--n_iteration", default="0", help="Enter the iteration number")
@click.option(
    "--mode",
    default="daily",
    type=click.Choice(["daily", "monthly"]),
    help="Enter the mode",
)
@click.option("--gpu", default=False, type=bool, help="Use GPU?")
@click.option("--log", default=False, type=bool, help="Use GPU?")
def configure(epochs, day, n_iteration, mode, gpu, log):

    set_log_level("ERROR")
    if epochs == 0:
        epochs = None
    delta = timedelta(minutes=30)
    # Read SR and EQ data
    df = read_data("NPdata.mat")
    arrays = (
        pd.DataFrame(df["NS_mean"])
        .applymap(lambda x: np.array(x, dtype=np.float32))
        .to_numpy()
    )
    # Adatpt to NeuralProphet input data
    NS_mean = np.array(np.stack([np.stack(a[0].squeeze()) for a in arrays]))
    # Create datetime Array
    pd.date_range(start="2018-09-09", end="2020-02-02")
    ds = pd.date_range(start="2016-01-01", end="2021-01-01", freq="30min")
    # Create the prior dataframe
    df_regressor = pd.DataFrame(
        {
            "ds": ds,
            "S0": NS_mean[:, 0],
            "S1": NS_mean[:, 1],
            "S2": NS_mean[:, 2],
            "S3": NS_mean[:, 3],
            "S4": NS_mean[:, 4],
            "S5": NS_mean[:, 5],
            "S6": NS_mean[:, 6],
            "S7": NS_mean[:, 7],
            "S8": NS_mean[:, 8],
            "S9": NS_mean[:, 9],
        }
    )
    config_npw_d = {
        "forecast_length": timedelta(hours=24),
        "freq": timedelta(minutes=30),
        "question_mark_length": timedelta(hours=24),
        "num_hidden_layers": 2,
        "learning_rate": 0.01,
        "n_lags": 5 * 48,
        "d_hidden": 16,
        "verbose": False,
        "epochs": epochs,
        "gpu": gpu,
    }
    config_npw = ConfigNPw(**config_npw_d)

    ConfigEQ_d = {
        "mag_array": df["mag"].to_numpy(),
        "dist_array": df["dist"].to_numpy(),
        "lat_array": df["lat"].to_numpy(),
        "arc_array": df["arc"].to_numpy(),
        "dist_start": 4000,
        "dist_delta": 2000,
        "dist_max": 6000,
        "lat_max": 360,
        "arc_max": 60,
        "mag_start": 5.5,
        "mag_delta": 1,
        "dist_perct": 1000,
    }
    config_events = ConfigEQ(**ConfigEQ_d)

    hours_offsets = [0]
    event_offsets = [None, -timedelta(hours=12), -timedelta(hours=24)]

    start_day = datetime.fromisoformat("2017-01-01T10:00:00")
    NPw_o = NPw(config_npw, df_regressor, config_events)
    start_date = start_day + int(day) * relativedelta(days=+1)
    for index_day in range(int(n_iteration) + 1):
        current_day = start_date + index_day * relativedelta(days=+1)
        print(str(current_day))
        test_metrics = NPw_o.predict_with_offset_hours(
            current_day, hours_offsets, event_offsets
        )
        NPw_o.save_df(current_day.strftime("%m_%d_%Y_%H_%M_%S"))


if __name__ == "__main__":
    configure()
