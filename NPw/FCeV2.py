from neuralprophet import NeuralProphet, forecaster, load
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from aux_function_SR import get_eq_filtered, SR_SENSORS
from NPw_aux import drop_scale, prepare_eq
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES, ANIMALS, COLORS
from dateutil.relativedelta import *
import os
import dateutil.parser
from pathlib import Path
from enum import Enum
from typing_extensions import Literal

import warnings
from IPython.display import display
from IPython.core.display import HTML
import json

from sdv.tabular import GaussianCopula
from darts import TimeSeries


@dataclass
class NeuralProphet_config:
    d_hidden: int
    num_hidden_layers: int
    yearly_seasonality: bool
    daily_seasonality: bool
    weekly_seasonality: bool
    seasonal_mode: Literal["additive", "multiplicative"]
    seasonal_reg: float
    multivariate_season: bool
    multivariate_trend: bool
    verbose: bool
    epochs: int
    drop_missing: bool
    impute_missing = True
    impute_linear = 48
    impute_rolling = 48
    normalization: bool
    event_mode: Literal["additive", "multiplicative"]
    growth: Literal["off", "Lineal"]
    trend_n_changepoint: int
    trend_regularization: float
    sparce_AR: float
    loss_func: Literal["Huber", "MSE", "MAE", "L1-Loss", "kl_div"]


class FCeV_type(Enum):
    NeuralProphet = 0
    Darts_TCN = 10
    Darts_TFT = 11


class FCeV_models(Enum):
    NeuralProphet = 0
    Darts = 1


def get_FCeV_model(fcev_type):
    print(fcev_type)
    if fcev_type is FCeV_type.NeuralProphet:
        print("NEURAL")
        return FCeV_models.NeuralProphet
    if fcev_type.value == FCeV_type.Darts_TCN.value:
        print("ENTER HERE")
        return FCeV_models.Darts
    else:
        print("Other things")


@dataclass
class FCeVConfig:
    """

    Attributes:
        forecast_length:
        question_mark_length:
    """

    freq: pd.Timedelta
    type: FCeV_type
    forecast_length: pd.Timedelta
    question_mark_length: pd.Timedelta
    end_date_train: datetime
    end_date_val: datetime
    learning_rate: float
    ts_duration: pd.Timedelta
    use_gpu: bool
    event_type: Literal["None", "Binary", "Non-Binary"]

    def __post_init__(self):
        self.forecast_length = pd.Timedelta(self.forecast_length)
        self.question_mark_length = pd.Timedelta(self.question_mark_length)


# Convert from wide format to long format df
def multivariate_df(df):
    """Function to convert columns of a DF to a single column with with ID column

    Args:
        df (): Input dataframe has to have one column named DS

    Returns:
        Return de dataframe with ID equal to each column

    """
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


# Funtion to convert from dataframe to Dart Timeseries
def df_to_ts(df):
    if df.empty:
        return None
    else:
        return TimeSeries.from_dataframe(df)


class FCeV:
    def __init__(
        self,
        FCev_config,
        df_input,
        df_past_covariates,
        df_future_covariates,
        df_events,
        synthetic_events=pd.DataFrame(),
    ):
        if FCev_config is FCeVConfig:
            self.FCev_config = FCev_config
        else:
            raise ValueError("Not valid FCev config file")
        self.FCeV_model = get_FCeV_model(self.FCev_config.type)
        if self.FCeV_model == FCeV_models.Darts:
            self.input_series = df_to_ts(df_input)
            if self.input_series == None:
                raise KeyError("A non empty dataframe has to be pass for df_input")
            self.past_covariates = df_to_ts(df_past_covariates)
            self.future_covariates = df_to_ts(df_future_covariates)
            self.events_ts = df_to_ts(df_events)
            self.synthetic_events = synthetic_events
            self.duration = pd.Timedelta(self.input_series.duration)
            self.start_date = pd.Timestamp(self.input_series.start_time())
        else:
            raise KeyError("Model not implemented")
        # Convert from wide format to long format
        if synthetic_events.empty:
            self.synthetic_events = None  # self.get_synthetic_events()
        else:
            self.synthetic_events = self.synthetic_events
        self.n_forecasts = int(self.FCev_config.forecast_length / self.FCev_config.freq)
        self.n_lags = int(self.FCev_config.ts_duration / self.FCev_config.freq)

        self.folds = pd.DataFrame(columns=["start_date", "end_date"])

    # IT has to be change completly
    def get_synthetic_events(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianCopula()
            model.fit(self.events_ts)
            synthetic_events = pd.DataFrame()
            pre_synthetic_events = model.sample(100)

    def get_folds(self, k=5):
        """Function to split all data into folds

        Args:
            k ():  number of folds
        """
        # Own fold method
        self.folds = pd.DataFrame(columns=["start_date", "end_date"])
        start_folds = self.start_date + (self.duration // 2)
        len_folds = self.duration / (2 * (k + 1))
        for index_fold in range(k + 1):
            end_fold = start_folds + (index_fold * len_folds)
            start_fold = end_fold - self.FCev_config.ts_duration
            self.folds.loc[index_fold] = [start_fold, end_fold]

    def process_fold(self, index_fold):
        """Function to prodcess each fold

        Args:
            index_fold (): index of the folds, has to be lower than the number of folds in get folds

        Returns:
            df_forecast (): Dataframe with the forecast
            df_uncertainty (): Dataframe with the values of uncertainty of that forecast

        """
        if index_fold < len(self.folds):
            start_fold, end_fold = self.folds.iloc[index_fold]
            df_forecast, df_uncertainty = self.process_forecast(start_fold, end_fold)
            return df_forecast, df_uncertainty
        else:
            raise ValueError("K Folds finished")

    def get_ts(self, ts, start_datetime, end_datetime):
        if self.FCeV_model == FCeV_models.Darts:
            return ts.slice(start_datetime, end_datetime)

    def add_events_to_model(self):
        dates_events = self.input_events.index
        # Push event fordware forecast time
        current_events_dates = dates_events + (self.config_npw.forecast_length / 2)
        # Take the events until the forecast time
        current_events_dates = current_events_dates[
            current_events_dates < start_forecast_time
        ]

        if self.config_npw.event_type == "Binary":
            [model, df_with_events] = self.add_events_neural_prophet(
                model, base_df, ["EV"], current_events_dates
            )
            current = df_with_events.set_index("ds")
        elif self.config_npw.event_type == "Non-Binary":
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
                # convert datetime to index of the df
                current = df_with_events.set_index("ds")
                # For each column name
                for current_date, current_event in zip(
                    current_events_dates, column_events
                ):
                    current.loc[current_date, current_column_name] = current_event
                df_with_events = current.reset_index()
        elif self.config_npw.event_type == "None":
            df_with_events = base_df
        else:
            raise KeyError("EVENT TYPE NOT VALID")

    def process_forecast(self, start_datetime, end_datetime):
        """Train the model and process a forecast

        Args:
            start_datetime (): start time for the trining data set
            end_datetime ():

        Returns:

        """
        start_forecast_time = start_datetime - self.FCev_config.forecast_length
        question_mark_start = (
            start_forecast_time - self.FCev_config.question_mark_length
        )
        # Takes events up to that point
        start_base = end_datetime
        end_base = start_forecast_time - self.FCev_config.forecast_length
        base_ts = self.get_ts(self.input_series, start_base, end_base)

        model = self.create_FCeV_model()
        # Remove all EQ after question_mark_start
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

        self.model, train_metrics = self.fit(model, base_ts)
        # Compute just one long forecast
        df_RMSE, df_MAE = self.test(model, start_forecast_time, df_test)
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

        model = model.add_events(list(name_events), mode=self.config_npw.event_mode)
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

    def create_FCeV_model(self):
        # trainer_config = {"accelerator":"gpu"}
        if self.config_npw.use_gpu:
            print("Using GPU")
            trainer_config = {"accelerator": "gpu"}
        else:
            trainer_config = {}

        if self.config_npw.multivariate_season:
            multivariate_season = "global"
        else:
            multivariate_season = "local"
        if self.config_npw.multivariate_trend:
            multiplicative_trend = "global"
        else:
            multiplicative_trend = "local"
        pi = 0.8  # prediction interval
        qts = [
            (1 - pi) / 2,
            pi + (1 - pi) / 2,
        ]  # quantiles based on the prediction interval
        model = NeuralProphet(
            n_forecasts=self.n_forecasts,
            growth=self.config_npw.growth,
            changepoints_range=0.95,
            weekly_seasonality=self.config_npw.weekly_seasonality,
            yearly_seasonality=self.config_npw.yearly_seasonality,
            daily_seasonality=self.config_npw.daily_seasonality,
            seasonality_mode=self.config_npw.seasonal_mode,
            seasonality_reg=self.config_npw.seasonal_reg,
            n_lags=self.n_lags,
            num_hidden_layers=self.config_npw.num_hidden_layers,
            d_hidden=self.config_npw.d_hidden,
            learning_rate=self.config_npw.learning_rate,
            trainer_config=trainer_config,
            trend_global_local=multiplicative_trend,
            season_global_local=multivariate_season,
            epochs=self.config_npw.epochs,
            quantiles=qts,
            impute_missing=True,
            impute_rolling=1000,
            impute_linear=1000,
            drop_missing=self.config_npw.drop_missing,
            n_changepoints=self.config_npw.trend_n_changepoint,
            trend_reg=self.config_npw.trend_regularization,
            ar_reg=self.config_npw.sparce_AR,
            loss_func=self.config_npw.loss_func,
        )
        model.set_plotting_backend("plotly")
        return model

    def test(self, model, start_forecast_time, df_test):
        # Without events:
        periods = 5
        df_results_RMSE = pd.DataFrame()
        df_results_MAE = pd.DataFrame()
        list_synthetic_RMSE = list()
        list_synthetic_MAE = list()
        list_keys = list()
        df_RMSE, df_MAE = self.one_step_test(model, start_forecast_time, df_test)
        df_RMSE.columns = "0"
        list_keys.append("BASE")
        list_synthetic_RMSE.append(df_RMSE)
        list_synthetic_MAE.append(df_MAE)
        if self.config_npw.event_type == "Binary":
            current_events = pd.DataFrame(pd.Series({"EV": 1.0}), columns=["EV"])
        elif self.config_npw.event_type == "Non-Binary":
            current_events = self.synthetic_events
        else:
            current_events = pd.DataFrame()

        for idx, synthetic_event in current_events.iterrows():
            # dt_events = pd.date_range(start = start_forecast_time, end = start_forecast_time + self.config_npw.forecast_length, periods = 5).values
            # To homogenize the dates in the CSV
            # dt_events = pd.date_range(start = 0, end = self.config_npw.forecast_length, periods = periods).values
            event_offset = [
                self.config_npw.forecast_length / (periods + 1) * x
                for x in range(1, periods + 1)
            ]
            df_dt_MAE = pd.DataFrame()
            df_dt_RMSE = pd.DataFrame()
            for dt in event_offset:
                current_df_test = df_test.copy()
                unique_dates = current_df_test["ds"].unique()
                closest_date = unique_dates[
                    np.abs(
                        unique_dates - (start_forecast_time + dt).to_numpy()
                    ).argmin()
                ]
                idx_closest = current_df_test.index[
                    current_df_test["ds"] == closest_date
                ]
                for synthetic_var in synthetic_event.index:
                    current_df_test.loc[idx_closest, synthetic_var] = synthetic_event[
                        synthetic_var
                    ]
                events_df = pd.DataFrame(synthetic_event).copy()
                events_df["ds"] = closest_date

                df_RMSE, df_MAE, uncer = self.one_step_test(
                    model, start_forecast_time, current_df_test
                )
                df_RMSE = df_RMSE.rename(
                    str(dt.total_seconds())
                )  # pd.to_datetime(str()).strftime("%m/%d/%Y, %H:%M:%S"))
                df_MAE = df_MAE.rename(
                    str(dt.total_seconds())
                )  # pd.to_datetime(str()).strftime("%m/%d/%Y, %H:%M:%S"))
                df_dt_RMSE = pd.concat([df_dt_RMSE, df_RMSE], axis=1)
                df_dt_MAE = pd.concat([df_dt_MAE, df_MAE], axis=1)
            list_synthetic_RMSE.append(df_dt_RMSE)
            list_synthetic_MAE.append(df_dt_MAE)
            list_keys.append("SYNT_" + str(idx))
        df_results_RMSE = pd.concat(list_synthetic_RMSE, axis=1, keys=list_keys)
        df_results_MAE = pd.concat(list_synthetic_MAE, axis=1, keys=list_keys)
        return df_results_RMSE, df_results_MAE

    def one_step_test(self, model, start_forecast_time, df_test):
        future = model.make_future_dataframe(
            df_test, n_historic_predictions=self.n_forecasts
        )
        forecast = model.predict(future, decompose=False, raw=True)

        forecast = forecast[forecast["ds"] == start_forecast_time]
        steps_name = ["step" + str(x) for x in range(self.n_forecasts)]
        forecast_unce = forecast[forecast.columns.difference(steps_name)].drop(
            ["ds", "ID"], axis=1
        )

        forecast_unce.columns = [
            "step" + str(word)
            for name in range(self.n_forecasts)
            for word in (str(name) + "___low", str(name) + "___high")
        ]
        forecast_unce.columns = forecast_unce.columns.str.split("___", expand=True)
        forecast_unce = forecast_unce.swaplevel(axis=1)

        forecast_unce = (
            forecast_unce["high"].sub(forecast_unce["low"]).abs().T.mean(axis=0)
        )

        forecast = forecast[steps_name]
        forecast = forecast.reset_index().drop("index", axis=1).T
        df_test = pd.pivot(df_test, index="ds", columns="ID", values="y")[
            -(self.n_forecasts) :
        ]
        forecast.columns = df_test.columns
        forecast_unce.columns = df_test.columns
        # Too late to find a better solution
        non_valid_column = [
            name_valid_column
            for name_valid_column in df_test
            if df_test[name_valid_column].isnull().values.all()
        ]
        df_test = df_test.dropna(axis=1, how="all")
        forecast = forecast.drop(
            non_valid_column, axis=1
        )  # [list(df_test.columns.values)]
        forecast.columns = df_test.columns + "___pred"
        df_test.columns = df_test.columns + "___current"
        # Improve with pd.concat keys
        df_all = pd.concat([df_test.reset_index(), forecast], axis=1).drop("ds", axis=1)
        df_all.columns = df_all.columns.str.split("___", expand=True)
        df_all = df_all.swaplevel(axis=1)
        RMSE = df_all["current"].sub(df_all["pred"]).pow(2).mean(axis=0).pow(1 / 2)
        MAE = df_all["current"].sub(df_all["pred"]).abs().mean(axis=0)
        print(forecast_unce)
        return pd.Series(RMSE.squeeze()), MAE, forecast_unce

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
            test_metrics = self.process_forecast(config_fc)
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
