import pandas as pd
from neuralprophet import NeuralProphet, forecaster, load
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from aux_function_SR import get_eq_filtered, SR_SENSORS
from NPw_aux import prepare_eq
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES, ANIMALS, COLORS
from dateutil.relativedelta import *
import dateutil.parser
import os
from pathlib import Path
from enum import Enum

import warnings
from IPython.display import display
from IPython.core.display import HTML
import json

from sdv.tabular import GaussianCopula
class EventFC(Enum):
    NP = 0


@dataclass
class ConfigNPw:
    type = EventFC.NP
    forecast_length: pd.Timedelta
    freq: pd.Timedelta
    num_hidden_layers: int
    learning_rate: float
    d_hidden: int
    question_mark_length: pd.Timedelta
    verbose: bool
    epochs: int
    gpu: bool
    binary_event: bool
    historic_lenght: pd.Timedelta
    drop_missing: bool
    multivariate_global: bool
    training_length: pd.Timedelta
    normalization: bool
    regularization: float


    def __post_init__(self):
        self.forecast_length = pd.Timedelta(self.forecast_length)
        self.historic_lenght = pd.Timedelta(self.historic_lenght)
        self.freq = pd.Timedelta(self.freq)
        self.question_mark_length = pd.Timedelta(self.question_mark_length)
        self.training_length = pd.Timedelta(self.training_length)


@dataclass
class ConfigEQ:
    dist_start: int
    dist_delta: int
    mag_start: float
    mag_delta: float
    filter: bool


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


METRICS = ["MAE", "RMSE"]


class NPw:
    def __init__(self, config_npw, df, config_events, df_events):
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
        self.df_events = df_events
        # Convert from wide format to long format
        self.input_df = df
        self.name = get_random_name(
            separator="_", combo=[COLORS, ANIMALS], style="lowercase"
        )
        self.input_l_df = multivariate_df(df)
        self.config_events = config_events
        self.input_events, self.transform = self.get_events(self.config_events)
        self.synthetic_events = self.get_synthetic_events()
        self.npw_df = pd.DataFrame(columns=self.NPw_columns)
        self.n_forecasts = int(self.config_npw.forecast_length / self.config_npw.freq)
        self.n_lags = int(self.config_npw.historic_lenght / self.config_npw.freq)
        self.metrics_test = pd.DataFrame(columns=METRICS)
        self.metrics_train = pd.DataFrame(columns=METRICS)
        self.folds = list()
        self.n_fold = 0
        self.list_keys_fold = list()
        self.list_test_RMSE = list()

        self.list_test_MAE = list()
    def get_synthetic_events (self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = GaussianCopula()
            model.fit(self.input_events)
            synthetic_events = pd.DataFrame()
            pre_synthetic_events = model.sample(100)
            for column_name in self.input_events.columns:
                synthetic_events = pd.concat([synthetic_events, pre_synthetic_events.nlargest(1, column_name)])
                
        return synthetic_events 
    def get_events(self, config_event):
        if True:  # isinstance(config_event, ConfigEQ):
            events = prepare_eq(
                self.df_events,
                config_event.dist_start,
                config_event.dist_delta,
                config_event.mag_start,
                config_event.mag_delta,
                config_event.filter,
            )
            return events
        else:
            raise TypeError("The configuration file is not valid")
        return events

    def get_folds(self, k=5):
        if self.config_npw.type == EventFC.NP:
            self.folds = list()
            # Own fold method
            start_folds = int(self.config_npw.training_length / self.config_npw.freq)
            len_folds = int((len(self.input_df) - start_folds) / k)
            for index_fold in range(1, k + 1):
                fold = self.input_df[: start_folds + (index_fold * len_folds)]
                self.folds.append(fold)
            # self.folds = model.crossvalidation_split_df(self.input_l_df, self.config_npw.freq, k, fold_pct, fold_overlap_pct)
            self.n_fold = 0

    def process_fold(self, index_fold):
        if index_fold < len(self.folds):
            df_train = self.folds[index_fold]
            start_date = df_train.iloc[-self.n_forecasts]["ds"]
            config_fc = ConfigForecast(start_forecast=start_date, offset_event=None)
            df_RMSE, df_MAE = self.add_forecast(config_fc)
            self.list_keys_fold.append(start_date)
            self.list_test_RMSE.append(df_RMSE)
            self.list_test_MAE.append(df_MAE) 
        else:
            raise Warning("K Folds finished")
    def get_results(self):
        return pd.concat(self.list_test_RMSE, keys = self.list_keys_fold)
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
        min_idx = np.argmin(
            np.abs(self.input_df["ds"] - config_forecast.start_forecast)
        )
        start_forecast_time = self.input_df["ds"].iloc[min_idx]
        question_mark_start = start_forecast_time - self.config_npw.question_mark_length
        # Takes events up to that point

        base_df = self.input_l_df[
            (
                self.input_l_df["ds"]
                < start_forecast_time + self.config_npw.forecast_length
            )
            & (
                self.input_l_df["ds"]
                > start_forecast_time - self.config_npw.training_length
            )
        ]

        n_samples = len(base_df)
        model = self.create_NP_model()
        # Remove all EQ after question_mark_start
        dates_events = self.input_events.index
        # Push event fordware forecast time
        current_events_dates = dates_events + self.config_npw.forecast_length
        # Take the events until the forecast time
        current_events_dates = current_events_dates[current_events_dates < start_forecast_time]

        # # Insert with negative offset
        # if config_forecast.offset_event is not None:
        #     current_events_dates = pd.concat(
        #         [
        #             pd.Series(current_events_dates),
        #             # Put an extra event in the forecast time
        #             pd.Series(start_forecast_time + config_forecast.offset_event),
        #         ]
        #     )
        if self.config_npw.binary_event:
            [model, df_with_events] = self.add_events_neural_prophet(
                model, base_df, ["EV"], current_events_dates
            )
            current = df_with_events.set_index("ds")
        else:
            # Get anmes of event parameter
            column_names = self.input_events.columns
            # Add all names to the model and create a df with dates event for every column
            [model, df_with_events] = self.add_events_neural_prophet(
                model, base_df, column_names, current_events_dates
            )
            # for each column name do
            for current_column_name in column_names:
                # Get the values of the column
                current_column = self.input_events.loc[:, current_column_name]
                # Select just the needed for this fc

                column_events = current_column[dates_events < question_mark_start]

                # if config_forecast.offset_event is not None:
                #     column_events = pd.concat(
                #         [column_events, pd.Series(np.mean(column_events))]
                #     )
                # convert datetime to index of the df
                current = df_with_events.set_index("ds")
                # For each column name
                for (current_date, current_event) in zip(
                    current_events_dates, column_events
                ):
                    current.loc[current_date, current_column_name] = current_event
                df_with_events = current.reset_index()
        self.base_df = base_df
        df_train, df_test = model.split_df(df_with_events, valid_p=1, local_split=True)
        start_fc = config_forecast.start_forecast
        self.df_test = df_test
        self.df_train = df_train
        # Duration between all actual events and the start of the forecast
        dif_duration = np.asarray([event - start_fc for event in dates_events])
        # Lowest duration between all the actual events and the start of the forecast
        dif_duration_in_s = (
            dif_duration[np.argmin(np.abs(dif_duration))]
        ).total_seconds()
        actual_event = divmod(dif_duration_in_s, 3600)[0]
        # Question mark force event
        if config_forecast.offset_event != None:
            expected_event = divmod(config_forecast.offset_event.total_seconds(), 3600)[
                0
            ]
        else:
            expected_event = None
        # Fit the model with df train

        self.model, train_metrics = self.fit(model, df_train)
        # Compute just one long forecast
        df_RMSE, df_MAE= self.test(model, start_forecast_time, df_test)
        # self.npw_df.loc[model_name] = [
        #     config_forecast.start_forecast,
        #     n_samples,
        #     config_forecast.offset_event,
        #     actual_event,
        #     expected_event,
        # ]
        return df_RMSE, df_MAE

    def plot(self, model, df):
        future = model.make_future_dataframe(
            df, periods=60 // 5 * 24 * 7, n_historic_predictions=True
        )
        forecast = model.predict(future)
        fig = model.plot(
            forecast[forecast["ID"] == "noa1"]
        )  # fig_comp = m.plot_components(forecast)
        fig_param = model.plot_parameters()

    def add_events_neural_prophet(self, model, base_df, name_events, dates_event):
        # Create EQ dataframe
        df_complete = pd.DataFrame()
        for name_event in name_events:
            current_df_events = pd.DataFrame(
                {
                    "event": name_event,
                    "ds": dates_event,
                }
            )
            df_complete = pd.concat([df_complete, current_df_events])

        model = model.add_events(list(name_events))
        df_with_events = model.create_df_with_events(base_df, df_complete)

        return [model, df_with_events]

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
            if self.config_npw.verbose == False:
                train_metrics = model.fit(df_train, minimal=True)
            else:
                train_metrics = model.fit(df_train, minimal=False)
            return model, train_metrics
        else:
            print("ERROR")

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

        if self.config_npw.multivariate_global:
            global_local = "global"
        else:
            global_local = "local"

        model = NeuralProphet(
            n_forecasts=self.n_forecasts,
            daily_seasonality="auto",
            # growth="discontinuous",
            yearly_seasonality=True,
            changepoints_range=0.95,
            # Number of potential trend changepoints to include.
            # n_changepoints=300,
            # Parameter modulating the flexibility of the automatic changepoint selection.
            trend_reg=self.config_npw.regularization,
            weekly_seasonality=False,
            n_lags=self.n_lags,
            num_hidden_layers=self.config_npw.num_hidden_layers,
            d_hidden=self.config_npw.d_hidden,
            learning_rate=self.config_npw.learning_rate,
            trainer_config=trainer_config,
            season_global_local=global_local,
            trend_global_local=global_local,
            epochs=self.config_npw.epochs,
            seasonality_mode="multiplicative",
            impute_missing=True,
            impute_rolling=24,
            impute_linear=100,
            drop_missing=self.config_npw.drop_missing,
        )
        model.set_plotting_backend("plotly")
        return model
    def test(self, model, start_forecast_time, df_test):
        # Without events:
        periods = 5
        df_results_RMSE = pd.DataFrame()
        df_results_MAE = pd.DataFrame()
        list_synthetic_RMSE = list()
        list_keys= list()
        df_RMSE, df_MAE = self.one_step_test(model, start_forecast_time, df_test)
        list_keys.append("BASE")
        list_synthetic_RMSE.append(df_RMSE)
        df_results_MAE = pd.concat([df_results_MAE, df_MAE])
        for idx, synthetic_event in self.synthetic_events.iterrows():
            # dt_events = pd.date_range(start = start_forecast_time, end = start_forecast_time + self.config_npw.forecast_length, periods = 5).values
            # To homogenize the dates in the CSV
            # dt_events = pd.date_range(start = 0, end = self.config_npw.forecast_length, periods = periods).values
            event_offset = [self.config_npw.forecast_length / (periods + 1)* x for x in range(1, periods + 1)]
            df_dt  = pd.DataFrame()
            for dt in event_offset:
                current_df_test = df_test.copy()
                unique_dates = current_df_test["ds"].unique()
                closest_date = unique_dates[np.abs(unique_dates - (start_forecast_time +dt).to_numpy()).argmin()]
                idx_closest = current_df_test.index[current_df_test["ds"] == closest_date]
                for synthetic_var in synthetic_event.index:
                    current_df_test.loc[idx_closest, synthetic_var] = synthetic_event[synthetic_var]
                df_RMSE, df_MAE = self.one_step_test(model, start_forecast_time, current_df_test)
                df_RMSE = df_RMSE.rename(str(dt.total_seconds()))#pd.to_datetime(str()).strftime("%m/%d/%Y, %H:%M:%S"))
                df_MAE.columns = [str(closest_date)]
                df_dt = pd.concat([df_dt, df_RMSE], axis = 1)
                df_results_MAE = pd.concat([df_results_MAE, df_MAE], axis = 1)
            list_synthetic_RMSE.append(df_dt)
            list_keys.append("SYNT" + str(idx))
        df_results_RMSE = pd.concat(list_synthetic_RMSE, axis = 1, keys = list_keys)
        return df_results_RMSE, df_results_MAE 
    def one_step_test(self, model, start_forecast_time, df_test):

        future = model.make_future_dataframe(
            df_test, n_historic_predictions=self.n_forecasts
        )
        forecast = model.predict(future, decompose=False, raw=True)

        forecast = forecast[forecast["ds"] == start_forecast_time].T

        forecast = forecast.drop(["ds", "ID"]).reset_index().drop("index", axis=1)
        df_test = pd.pivot(df_test, index="ds", columns="ID", values="y")[
            -(self.n_forecasts) :
        ]
        df_test = df_test.dropna(axis=1, how = "all")
        forecast.columns = df_test.columns + "_pred"
        df_test.columns = df_test.columns + "_current"
        # Improve with pd.concat keys
        df_all = pd.concat([df_test.reset_index(), forecast], axis=1).drop("ds", axis=1)
        df_all.columns = df_all.columns.str.split("_", expand=True)
        df_all = df_all.swaplevel(axis=1)
        RMSE = df_all["current"].sub(df_all["pred"]).pow(2).mean(axis=0).pow(1 / 2)
        MAE = df_all["current"].sub(df_all["pred"]).abs().mean(axis=0)
        return pd.Series(RMSE.squeeze()), MAE
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

    def get_df_from_folder(self, dir_path):
        # Iterate directory
        df_output = pd.DataFrame()
        for file in Path(dir_path).glob("*.csv"):
            with file as f:
                self.npw_df = pd.concat([self.npw_df, pd.read_csv(f)])

    def remove_experiments(self, case):
        self.npw_df = self.npw_df.loc[self.npw_df["expected_event"] != case]

    def get_binary_perform(self, metrics, min_limit, max_limit, config_events=None):
        df = self.get_binary_results(metrics, min_limit, max_limit, config_events)
        # Get the confusion matrix
        cm = confusion_matrix(df["actual_class"], df["predicted_class"])
        # Now the normalize the diagonal entries
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        return np.nan_to_num(cm.diagonal()), cm

    def pre_binary_perform_fast(self, metric):

        df_pre = pd.DataFrame()
        # df data from the object
        df = self.npw_df
        # Remove NA
        df = df.fillna(0)

        date_format = "%d-%m-%Y %H:%M:%S"
        # Group by the forecasting day
        for current_day in df["start_forecast"].unique():
            # Auto Get datetime format from string a
            current_day_datetime = dateutil.parser.parse(current_day)
            # Select all the cases for the same day
            current_df = df[df["start_forecast"] == current_day].reset_index()

            # Get the location of no event
            loc_no_event = np.where(current_df["expected_event"] == 0)[0]

            best_class = np.argmin(current_df[metric])

            if best_class == loc_no_event:
                predicted_class = 0
            else:
                predicted_class = 1
            dict_pre = {
                "dates": current_day_datetime,
                "predicted_class": predicted_class,
            }
            df_pre = pd.concat([df_pre, pd.DataFrame.from_dict([dict_pre])])
        return df_pre

    def get_binary_perform_fast(self, df, min_limit, max_limit, config_events=None):

        if config_events == None:
            events_df_dates = self.input_events["dates"]
        else:
            events_df_dates = self.get_events(config_events)["dates"]
        if len(events_df_dates) == 0:
            return [np.zeros(len(df["predicted_class"].unique()) + 1), None]

        # Diference between events and each row in seconds
        dif = [
            (
                events_df_dates.iloc[np.argmin(abs(events_df_dates - current_day))]
                - current_day
            ).total_seconds()
            for current_day in df["dates"]
        ]
        ground_truth = [
            int(closest_eq) in range(-min_limit * 3600, max_limit * 3600 - 1)
            for closest_eq in dif
        ]
        # Get the confusion matrix
        cm = confusion_matrix(ground_truth, df["predicted_class"])
        # Now the normalize the diagonal entries
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        results = np.append(np.nan_to_num(cm_norm.diagonal()), (cm[1, 1] / cm[0, 0]))

        return [results, cm]

    def get_binary_results(
        self, metric, min_limit, max_limit, config_events=None, minimal=False
    ):

        if config_events == None:
            events_df_dates = self.input_events
        else:
            # This can be reduced
            events_df_dates = self.get_events(config_events)

        df_output = pd.DataFrame()

        # df data from the object
        df = self.npw_df

        # Remove NA
        df = df.fillna(0)

        date_format = "%d-%m-%Y %H:%M:%S"
        # Group by the forecasting day
        for current_day in df["start_forecast"].unique():
            # Auto Get datetime format from string a
            current_day_datetime = dateutil.parser.parse(current_day)

            # Get the index of the closest event based on envents_df
            loc_eq = np.argmin(abs(events_df_dates["dates"] - current_day_datetime))

            # IF there is a event in the next 24 hours it should be discarted.

            # Get the closest event to the forecasting day and convert to a dictionary
            event = events_df_dates.iloc[loc_eq].to_dict()

            # Time diference between the event and the current hour
            dif_event_in_s = (event["dates"] - current_day_datetime).total_seconds()

            dif_event = divmod(dif_event_in_s, 3600)[0]
            # Select all the cases for the same day
            current_df = df[df["start_forecast"] == current_day].reset_index()

            # Get the location of no event
            loc_no_event = np.where(current_df["expected_event"] == 0)[0]

            # [int(a in range(-24, 24)) for a in current_df.npw_df["actual_event"]]

            # If there was an event in the last min_limit hours, it will count as 0 else 1
            if np.any((dif_event >= -min_limit) & (dif_event < max_limit)):
                actual_class = 1
            else:
                actual_class = 0
            loc_min = np.argmin(
                np.abs(current_df["actual_event"] - current_df["expected_event"])
            )
            ref = np.mean(current_df["actual_event"])

            best_class = np.argmin(current_df[metric])

            if best_class == loc_no_event:
                predicted_class = 0
            else:
                predicted_class = 1

            #            for index_row in range

            current_metrics = current_df[metric]

            metrics_event_value = [
                (event_metrics - current_metrics.iloc[loc_no_event]).values[0]
                for event_metrics in current_metrics.drop(loc_no_event)
            ]
            names = [
                "event_" + metric + "_" + str(index)
                for index in range(len(metrics_event_value))
            ]
            d_events = dict(zip(names, metrics_event_value))

            # Get number of classes == rows for a particular day
            n_classes = len(current_df)

            diff_metrics = np.abs(
                current_df[metric][predicted_class] - current_df[metric][actual_class]
            )
            error = np.mean(current_df[metric])
            if actual_class == 0:
                if predicted_class == 0:
                    result = "TN"
                else:
                    result = "FP"
            else:
                if actual_class == 0:
                    result = "FN"
                else:
                    result = "TP"

            dict_output = {
                "date": current_day_datetime,
                "ref": ref,
                "best_class": best_class,
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "type": result,
                "diff_metrics": diff_metrics,
                "n_classes": n_classes,
                **d_events,
                **event,
                "dif_event": dif_event,
            }
            df_output = pd.concat([df_output, pd.DataFrame.from_dict([dict_output])])

        df_output = df_output.set_index("date").sort_values("date")
        return df_output

    def _____get_binary_results(self, metric, config_events=None):
        if config_events == None:
            config_events = self.config_events
        events_df = self.get_events(config_events)

        df_output = pd.DataFrame()

        # df data from the object
        df = self.npw_df

        # Remove NA
        df = df.fillna(0)

        date_format = "%d-%m-%Y %H:%M:%S"
        # Group by the forecasting day
        for current_day in df["start_forecast"].unique():
            # Auto Get datetime format from string a
            current_day_datetime = dateutil.parser.parse(current_day)
            # Get the index of the closest event based on envents_df
            loc_eq = np.argmin(abs(events_df["dates"] - current_day_datetime))

            # Get the closest event to the forecasting day and convert to a dictionary
            event = events_df.iloc[loc_eq].to_dict()

            # Time diference between the event and the current hour
            dif_event_in_s = (event["dates"] - current_day_datetime).total_seconds()

            dif_event = divmod(dif_event_in_s, 3600)[0]
            # Select all the cases for the same day
            current_df = df[df["start_forecast"] == current_day].reset_index()

            # Get the location of no event
            loc_no_event = np.where(current_df["expected_event"] == 0)[0]

            # If there was an event in the last 24 hours, it will count as 0 else 1
            if np.any(
                (current_df["actual_event"] >= -24) & (current_df["actual_event"] < 0)
            ):
                actual_class = 1
            else:
                actual_class = 0
            loc_min = np.argmin(
                np.abs(current_df["actual_event"] - current_df["expected_event"])
            )
            ref = np.mean(current_df["actual_event"])

            best_class = np.argmin(current_df[metric])

            if best_class == loc_no_event:
                predicted_class = 0
            else:
                predicted_class = 1

            #            for index_row in range

            current_metrics = current_df[metric]

            metrics_event_value = [
                (event_metrics - current_metrics.iloc[loc_no_event]).values[0]
                for event_metrics in current_metrics.drop(loc_no_event)
            ]
            names = [
                "event_" + metric + "_" + str(index)
                for index in range(len(metrics_event_value))
            ]
            d_events = dict(zip(names, metrics_event_value))

            # Get number of classes == rows for a particular day
            n_classes = len(current_df)

            diff_metrics = np.abs(
                current_df[metric][predicted_class] - current_df[metric][actual_class]
            )
            error = np.mean(current_df[metric])
            if actual_class == 0:
                if predicted_class == 0:
                    result = "TN"
                else:
                    result = "FP"
            else:
                if actual_class == 0:
                    result = "FN"
                else:
                    result = "TP"

            dict_output = {
                "date": current_day_datetime,
                "ref": ref,
                "best_class": best_class,
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "type": result,
                "diff_metrics": diff_metrics,
                "n_classes": n_classes,
                **d_events,
                **event,
                "dif_event": dif_event,
            }
            df_output = pd.concat([df_output, pd.DataFrame.from_dict([dict_output])])

        df_output = df_output.set_index("date").sort_values("date")
        return df_output

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

    @staticmethod
    def save_config(config_path, config_npw, config_events):
        config_npw_json = json.dumps(
            [asdict(config_npw), asdict(config_events)],
            indent=4,
            sort_keys=True,
            default=str,
        )
        with open(config_path, "w") as outfile:
            outfile.write(config_npw_json)

    @staticmethod
    def load_config(path):
        with open("sample.json", "r") as infile:
            data = infile.read()
            data = json.loads(data)
        config_npw = ConfigNPw(**data[0])
        config_events = ConfigEQ(**data[1])
        return (config_npw, config_events)

    @staticmethod
    def load_class(filename):
        with open(filename + ".pkl", "rb") as f:
            return load(f)

    @staticmethod
    def get_metrics_from_folder(dir_path):
        # folder path

        # list to store files
        res = []
        metrics = "MSE"
        # Iterate directory
        df_output = pd.DataFrame()
        for file in Path(dir_path).glob("*.csv"):
            with file as f:
                df = pd.read_csv(f)
                df = df.fillna(0)
                df_current_output = NPw.get_metrics_from_df(metrics, df)
                df_output = pd.concat([df_output, df_current_output])
        df_output = df_output.set_index("date").sort_values("date")
        return df_output

    @staticmethod
    def print_df(df):
        display(HTML(df.to_html()))

    @staticmethod
    def get_metrics_from_df2(metrics, df):
        df_output = pd.DataFrame()
        # df = df.loc[df["expected_event"] != -24]
        # df = df.loc[df["actual_event"] != -24]

        for current_day in df["start_forecast"].unique():
            current_day_datetime = dateutil.parser.parse(current_day)
            # Get all the cases for the same day
            current_df = df[df["start_forecast"] == current_day].reset_index()
            # Take the lowest as the reference

            n_classes = len(current_df)
            actual_class = loc_min
            predicted_class = better_marc_loc
            diff_metrics = np.abs(
                current_df[metrics][predicted_class] - current_df[metrics][actual_class]
            )
            if actual_class == 0:
                if predicted_class == 0:
                    result = "TN"
                else:
                    result = "FP"
            else:
                if predicted_class == 0:
                    result = "FN"
                else:
                    result = "TP"
            dict_output = {
                "date": current_day_datetime,
                "ref": ref,
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "type": result,
                "diff_metrics": diff_metrics,
                "n_classes": n_classes,
            }
            df_output = pd.concat([df_output, pd.DataFrame.from_dict([dict_output])])
        return df_oup
