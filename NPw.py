import pandas as pd
from neuralprophet import NeuralProphet, save, load
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from aux_function_SR import get_eq_filtered, SR_SENSORS
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES, ANIMALS, COLORS
from dateutil.relativedelta import *
import dateutil.parser
import os
from pathlib import Path
from IPython.display import display, HTML
import json


@dataclass
class ConfigNPw:
    forecast_length: pd.Timedelta
    freq: pd.Timedelta
    num_hidden_layers: int
    learning_rate: float
    n_lags: int
    d_hidden: int
    question_mark_length: pd.Timedelta
    verbose: bool
    epochs: int
    gpu: bool
    binary_event: bool
    def __post_init__(self):
            self.forecast_length = pd.Timedelta(self.forecast_length)
            self.freq = pd.Timedelta(self.freq)
            self.question_mark_length = pd.Timedelta(self.question_mark_length)

@dataclass
class ConfigEQ:
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
        self.input_events = self.get_events(self.config_events)
        self.npw_df = pd.DataFrame(columns=self.NPw_columns)

    def get_next_event(self, start_time):
        loc_eq = np.where(self.input_events["dates"] > start_time)[0][0]
        return self.input_events.loc["dates"].iloc[loc_eq]

    def get_events(self, config_event):
        if True:  # isinstance(config_event, ConfigEQ):
            [_, earthquake_raw, _] = get_eq_filtered(
                self.df_events,
                config_event.dist_start,
                config_event.dist_delta,
                config_event.dist_max,
                config_event.lat_max,
                config_event.arc_max,
                config_event.mag_start,
                config_event.mag_delta,
                config_event.dist_perct,
                1,
                1,
                SR_SENSORS.NS,
            )
            arc = self.df_events.loc[:, "arc"][earthquake_raw > 0.0]
            arc = np.abs(np.cos(np.deg2rad(90 + arc)))
            events = pd.DataFrame(
                {
                    "dates": self.input_df["ds"][earthquake_raw > 0.0],
                    "dist": self.df_events.loc[:, "dist"][earthquake_raw > 0.0] / 1000,
                    "mag": self.df_events.loc[:, "mag"][earthquake_raw > 0.0],
                    "arc": arc,
                }
            )
        else:
            raise TypeError("The configuration file is not valid")
        return events

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
        dates_events = self.input_events["dates"]
        mag_events = self.input_events["mag"]
        dist_events = self.input_events["dist"]
        current_events_dates = dates_events[dates_events < question_mark_start]

        # Insert with negative offset
        if config_forecast.offset_event is not None:
            current_events_dates = pd.concat(
                [
                    current_events_dates,
                    pd.Series(start_forecast_time + config_forecast.offset_event),
                ]
            )
        if self.config_npw.binary_event:
            [model, df_with_events] = self.add_events_neural_prophet(
                model, base_df, ["EV"], current_events_dates
            )
            current = df_with_events.set_index("ds")
        else:
            # Get anmes of event parameter
            column_names = self.input_events.columns.drop("dates")
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

                if config_forecast.offset_event is not None:
                    column_events = pd.concat(
                        [column_events, pd.Series(np.mean(column_events))]
                    )
                # convert datetime to index of the df
                current = df_with_events.set_index("ds")
                # For each column name
                for (current_date, current_event) in zip(
                    current_events_dates, column_events
                ):
                    current.loc[current_date, current_column_name] = current_event
                df_with_events = current.reset_index()
        
        df_train, df_test = model.split_df(df_with_events, valid_p=1, local_split=True)
        start_fc = config_forecast.start_forecast
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

    def add_events_neural_prophet(self, model, base_df, name_events, dates_event):
        # Create EQ dataframe
        df_complete = pd.DataFrame()
        for name_event in name_events:
            current_df_events = pd.DataFrame({"event": name_event, "ds": dates_event,})
            df_complete = pd.concat([df_complete, current_df_events])

        model = model.add_events(list(name_events))
        df_with_events = model.create_df_with_events(base_df, df_complete)

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
            if self.config_npw.verbose == False:
                model.fit(df_train, minimal=True)
            else:
                model.fit(df_train, minimal=False)
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
        config_npw_json = json.dumps([asdict(config_npw), asdict(config_events)], indent=4, sort_keys=True, default=str)
        with open(config_path, "w") as outfile:
            outfile.write(config_npw_json)
    @staticmethod
    def load_config(path):
        with open("sample.json", "r") as infile:
            data=infile.read();
            data=json.loads(data)
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
        return df_output
