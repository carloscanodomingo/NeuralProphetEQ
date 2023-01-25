import pandas as pd
from neuralprophet import NeuralProphet, save, load
from dataclasses import dataclass
from datetime import datetime, timedelta
from aux_function_SR import get_eq_filtered, SR_SENSORS
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES, ANIMALS, COLORS
from dateutil.relativedelta import *
import os
from pathlib import Path


@dataclass
class ConfigNPw:
    forecast_length: timedelta
    freq: timedelta
    num_hidden_layers: int
    learning_rate: float
    n_lags: int
    d_hidden: int
    question_mark_length: timedelta
    verbose: bool
    epochs: int
    gpu: bool


@dataclass
class ConfigEQ:
    mag_array: np.ndarray
    dist_array: np.ndarray
    lat_array: np.ndarray
    arc_array: np.ndarray
    dist_start: int
    dist_delta: int
    dist_max: int
    lat_max: int
    arc_max: int
    mag_start: float
    mag_delta: float
    dist_perct: float


@dataclass
class ConfigForecast:
    start_forecast: datetime
    offset_event: timedelta


# Convert from wide format to long format df
def multivariate_df(df):
    var_names = list(df)[1:]
    mv_df = pd.DataFrame()
    for col in var_names:
        aux = df[["ds", col]].copy(deep=True)  # select column associated with region
        aux = aux.iloc[:, :].copy(
            deep=True
        )  # selects data up to 26301 row (2004 to 2007 time stamps)
        aux = aux.rename(
            columns={col: "y"}
        )  # rename column of data to 'y' which is compatible with Neural Prophet
        aux["ID"] = col
        mv_df = pd.concat((mv_df, aux))
    return mv_df


class NPw:
    def __init__(self, config_npw, df, config_event):
        self.NPw_columns = [
            "start_forecast",
            "n_samples",
            "offset_model",
            "RMSE",
            "MSE",
            "MAE",
            "actual_event",
            "expected_event",
        ]
        self.config_npw = config_npw
        # Convert from wide format to long format
        self.input_df = df
        self.name = get_random_name(
            separator="_", combo=[COLORS, ANIMALS], style="lowercase"
        )
        self.input_l_df = multivariate_df(df)
        self.config_event = config_event
        self.input_events = self.get_events()
        self.npw_df = pd.DataFrame(columns=self.NPw_columns)

    def get_next_event(self, start_time):
        loc_eq = np.where(self.input_events > start_time)[0][0]
        return self.input_events.iloc[loc_eq]

    def get_events(self):
        if isinstance(self.config_event, ConfigEQ):
            [_, earthquake_raw, _] = get_eq_filtered(
                self.config_event.dist_array,
                self.config_event.mag_array,
                self.config_event.lat_array,
                self.config_event.arc_array,
                self.config_event.dist_start,
                self.config_event.dist_delta,
                self.config_event.dist_max,
                self.config_event.lat_max,
                self.config_event.arc_max,
                self.config_event.mag_start,
                self.config_event.mag_delta,
                self.config_event.dist_perct,
                1,
                1,
                SR_SENSORS.NS,
            )
        events_eq = self.input_df["ds"][earthquake_raw > 0.0]
        return events_eq

    def add_forecast(self, config_forecast):
        # if isinstance(config_forecast, ConfigForecast) == False:
        #        raise TypeError("Only ConfigForecast are allowed")

        if (
            config_forecast.offset_event != None
            and config_forecast.offset_event > timedelta(minutes=0)
        ):
            raise Warning("offset_event should be negative")
        model_name = get_random_name(
            separator="-", combo=[ADJECTIVES, NAMES], style="lowercase"
        )

        # Insert EQ cases
        start_forecast_time = config_forecast.start_forecast
        question_mark_start = start_forecast_time - self.config_npw.question_mark_length
        # Takes events up to that point

        base_df = self.input_l_df[
            self.input_l_df["ds"]
            < start_forecast_time + self.config_npw.forecast_length
        ]

        n_samples = len(base_df)
        model = self.create_NP_model()
        # Remove all EQ after question_mark_start
        events = self.input_events[self.input_events < question_mark_start]

        # Insert with negative offset
        if config_forecast.offset_event is not None:
            events = pd.concat(
                [events, pd.Series(start_forecast_time + config_forecast.offset_event)]
            )

        [model, df_with_events] = self.add_events_neural_prophet(model, base_df, events)
        df_train, df_test = model.split_df(df_with_events, valid_p=1, local_split=True)

        start_fc = config_forecast.start_forecast
        # Duration between all actual events and the start of the forecast
        dif_duration = np.asarray([event - start_fc for event in self.input_events])
        # Lowest duration between all the actual events and the start of the forecast
        dif_duration_in_s = (dif_duration[np.argmin(np.abs(dif_duration))]).total_seconds()
        actual_event = divmod(dif_duration_in_s, 3600)[0]
        # Question mark force event
        if config_forecast.offset_event != None:
            expected_event = divmod(config_forecast.offset_event.total_seconds(), 3600)[
                0
            ]
        else:
            expected_event = None
        
        # Fit the model with df train
        model = self.fit(model, df_train)
        
        # Compute just one long forecast
        test_metrics = self.one_step_test(model, df_test)
        self.npw_df.loc[model_name] = [
            config_forecast.start_forecast,
            n_samples,
            config_forecast.offset_event,
            test_metrics["RMSE"],
            test_metrics["MSE"],
            test_metrics["MAE"],
            actual_event,
            expected_event,
        ]
        return test_metrics

    def add_events_neural_prophet(self, model, base_df, events):
        # Create EQ dataframe
        df_events = pd.DataFrame(
            {
                "event": "EV",
                "ds": events,
            }
        )
        model = model.add_events(["EV"])
        df_with_events = model.create_df_with_events(base_df, df_events)

        return [model, df_with_events]

    def save_config(self):
        return 0

    def save_df(self, filename):
        path = Path(filename + ".csv")
        for index in range(100):
            if path.is_file():
                path = Path(filename + "_" + str(index + 1) + ".csv")
            else:
                self.npw_df.to_csv(path)
                break

    def fit(self, model, df_train):
        if isinstance(self.config_npw, ConfigNPw):
            model.fit(df_train)
            return model

    def fit_models(self):
        # https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01
        for index in self.npw_df.index:
            if self.npw_df.loc[index]["is_fit"] == False:
                model = self.npw_df.loc[index]["model"]
                self.npw_df.at[index, "train_metrics"] = model.fit(
                    self.npw_df.loc[index]["df_train"]
                )
                self.npw_df.at[index, "is_fit"] = True
                self.npw_df.at[index, "model"] = model

    def create_NP_model(self):
        # trainer_config = {"accelerator":"gpu"}
        if self.config_npw.gpu:
            print("Using GPU")
            trainer_config = {"accelerator": "gpu"}
        else:
            trainer_config = {}

        model = NeuralProphet(
            n_forecasts=int(self.config_npw.forecast_length / self.config_npw.freq),
            growth="off",
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=False,
            n_lags=self.config_npw.n_lags,
            num_hidden_layers=self.config_npw.num_hidden_layers,
            d_hidden=self.config_npw.d_hidden,
            learning_rate=self.config_npw.learning_rate,
            trainer_config=trainer_config,
            trend_global_local="global",
            season_global_local="global",
            epochs=self.config_npw.epochs,
        )
        model.set_plotting_backend("plotly")
        return model

    def one_step_test(self, model, df_test):
        future = model.make_future_dataframe(df_test, n_historic_predictions=0)
        forecast = model.predict(future, decompose=False, raw=True)
        y_predicted = forecast.filter(like="step").to_numpy()
        y_actual = (
            pd.pivot(df_test, index="ds", columns="ID", values="y")[-48:]
            .transpose()
            .to_numpy()
        )
        RMSE = mean_squared_error(y_actual, y_predicted, squared=False)
        MSE = mean_squared_error(y_actual, y_predicted)
        MAE = mean_absolute_error(y_actual, y_predicted)
        print("MSE: " + str(MSE))
        return {"RMSE": RMSE, "MSE": MSE, "MAE": MAE}

    # Predict for differents hours offset and differents event offset
    def predict_with_offset_hours(self, start_day, hours_offsets, event_offsets):
        for hours_offset in hours_offsets:
            start_forecast = start_day + relativedelta(hours=+hours_offset)
            self.predict_with_offset_events(start_forecast, event_offsets)

            
            
    def predict_with_offset_events(self, start_forecast, event_offsets):
        for event_offset in event_offsets:
            config_fc = ConfigForecast(
                start_forecast=start_forecast, offset_event=event_offset
            )
            test_metrics = self.add_forecast(config_fc)
            print(
                str(start_forecast) + " " + str(event_offset) + " " + str(test_metrics)
            )

    def _____one_step_test(self):
        # https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01
        for index in self.npw_df.index:
            if self.npw_df.loc[index]["is_fit"] == True:
                future = self.npw_df.loc[index]["model"].make_future_dataframe(
                    self.npw_df.loc[index]["df_test"], n_historic_predictions=0
                )
                forecast = self.npw_df.loc[index]["model"].predict(
                    future, decompose=False, raw=True
                )
                y_predicted = forecast.filter(like="step").to_numpy()
                y_actual = (
                    pd.pivot(
                        self.npw_df.loc[index]["df_test"],
                        index="ds",
                        columns="ID",
                        values="y",
                    )[-48:]
                    .transpose()
                    .to_numpy()
                )
                self.npw_df.at[index, "test_metrics"] = mean_squared_error(
                    y_actual, y_predicted
                )

    def load_class(filename):
        with open(filename + ".pkl", "rb") as f:
            return load(f)
    
    def get_metrics_from_folder(dir_path):
        # folder path

        # list to store files
        res = []

        # Iterate directory
        for file in Path(dir_path).rglob("*.csv")
        print(res)