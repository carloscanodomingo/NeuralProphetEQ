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
import pickle
from IPython.display import display
from IPython.core.display import HTML
import json
from DartsFCeV import DartsFCeV, DartsFCeVConfig
import lzma

class METRICS(Enum):
    CoV = (0,)
    mape = (1,)
    marre = (2,)
    RMSE = (3,)
    ZSCORE = (4,)
    CosSim = (5,)
    PRESS = (6,)
    RMSE_sigma = (7,)
    DTW = (8, )



@dataclass
class FCeVConfig:
    freq: pd.Timedelta
    forecast_length: pd.Timedelta
    question_mark_length: pd.Timedelta
    training_length: pd.Timedelta
    input_length: pd.Timedelta
    verbose: bool

    def __post_init__(self):
        self.forecast_length = pd.Timedelta(self.forecast_length)
        self.question_mark_length = pd.Timedelta(self.question_mark_length)
@dataclass
class FCeVResultsData:
    df_input: pd.DataFrame
    df_covariate:pd.DataFrame
    df_events: pd.DataFrame
    df_synthetics:pd.DataFrame
    fcev_config: FCeVConfig

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


class FCeV:
    def __init__(
        self,
        FCeV_config,
        model_FCeV_config,
        df_input,
        df_covariates,
        df_events,
        output_path,
        synthetic_events=pd.DataFrame(),
    ):
        if not isinstance(FCeV_config, FCeVConfig):
            raise ValueError("FCeV_config has to be a FCeVConfig instance")

        # Move this check to th outside class
        if df_input.empty:
            raise KeyError("A non empty dataframe has to be pass for df_input")
        if isinstance(model_FCeV_config, DartsFCeVConfig):
            self.FCeV_model = DartsFCeV(
                FCeV_config,
                model_FCeV_config,
                df_input,
                df_covariates,
                df_events,
                synthetic_events,
            )
        else:
            raise KeyError("Model not implemented")
        self.start_date = self.FCeV_model.start_date()
        self.output_path = output_path 
        self.duration = self.FCeV_model.duration()
        self.end_date = self.start_date + self.duration
        self.FCeV_config = FCeV_config
        self.folds = None
        self.iterations = None
        self.index_dates = df_input.reset_index()["ds"]


    def get_results_data(self):
        return {"df_input": self.FCeV_model.get_df_input(),
                "df_covariates":  self.FCeV_model.get_df_covariates(),
                 "df_events":  self.FCeV_model.get_events(),
                "df_synthetics":             self.FCeV_model.get_synthetics(),
                "forecast_length":               self.FCeV_config.forecast_length,
                "input_length":               self.FCeV_config.input_length,
                "training_length": self.FCeV_config.training_length,
                "freq": self.FCeV_config.freq, 
                "n_forecast": self.FCeV_model.n_forecasts,
               }

    def save_results(self, df_fore):
        file_name_config = f"{self.output_path}{type(self.FCeV_model).__name__}_config.cpkl"
        config_path = Path(file_name_config)
        if not config_path.is_file():
            with open(config_path, 'wb') as f:
                pickle.dump(self.get_results_data(), f)
        file_name_df_fore = df_fore["BASE"].index[0].strftime("%Y_%m_%d_%H_%M_%S")
        with open(f"{self.output_path}df_forecast_{file_name_df_fore}.pkl", 'wb') as f:
            pickle.dump(df_fore, f)


    def create_folds(self, date_start, n_iteration):
        """Function to split all data into folds

        Args:
            k ():  number of folds
        """

        # Own fold method
        self.folds = pd.DataFrame(columns=["start_date", "end_date"])
        print(self.start_date)
        start_folds = date_start
        len_folds = (self.end_date - start_folds) / ( n_iteration+ 1)
        for index_fold in range(n_iteration):
            end_fold = start_folds + ((index_fold + 1) * len_folds)
            index_end_fold = np.argmin(np.abs(self.index_dates - end_fold))
            end_fold = self.index_dates[index_end_fold]
            start_fold = end_fold - self.FCeV_config.training_length
            if start_fold < self.start_date:
                start_fold = self.start_date
            self.folds.loc[index_fold] = [start_fold, end_fold]

    def process_fold(self, index_fold):
        """Function to prodcess each fold

        Args:
            index_fold (): index of the folds, has to be lower than the number of folds in get folds

        Returns:
            df_forecast (): Dataframe with the forecast
            df_uncertainty (): Dataframe with the values of uncertainty of that forecast

        """
        if self.folds is None:
            raise ValueError("Folds hasnt been created yet")
        if index_fold >= len(self.folds):
            raise ValueError(f"Index folds exceed number of folds: {len(self.folds)}")
        start_fold, end_fold = self.folds.iloc[index_fold]
        # df_forecast, df_uncertainty = 
        df_forecast = self.FCeV_model.process_forecast(
            start_fold, end_fold
        )
        return df_forecast

    def process_iteration(self, index_fold):

        if self.iterations is None:
            raise ValueError("Iterations hasnt been created yet")
        if index_fold >= len(self.iterations):
            raise ValueError(f"Index index_iteration exceed number of iterations: {len(self.iterations)}")
        start_fold, end_fold = self.iterations.iloc[index_fold]
        # df_forecast, df_uncertainty = 
        df_forecast  = self.FCeV_model.process_forecast(
            start_fold, end_fold
        )
        return df_forecast

    def create_iteration(self, date_start, n_iteration):
        self.iterations = pd.DataFrame(columns=["start_date", "end_date"])
        len_iteration = self.FCeV_config.forecast_length
        max_number_iteration = int((self.end_date- date_start- self.FCeV_config.forecast_length) // len_iteration)
        if n_iteration is None:
            n_iteration = max_number_iteration
        else:
            n_iteration = n_iteration if max_number_iteration > n_iteration else max_number_iteration
        for index_iter in range(n_iteration):
            end_fold = date_start  + (index_iter * self.FCeV_config.forecast_length)
            index_end_fold = np.argmin(np.abs(self.index_dates - end_fold))
            end_fold = self.index_dates[index_end_fold]
            start_fold = end_fold - self.FCeV_config.training_length
            if start_fold < self.start_date:
                start_fold = self.start_date
            self.iterations.loc[index_iter] = [start_fold, end_fold]

    @staticmethod
    def summarize_metrics(df_forecast, metrics):

        mean_pred_values = df_forecast["current"].mean().mean()
        return df_forecast.mean() / mean_pred_values
    @staticmethod
    def get_metrics_from_fc(df_current, df_pred, metrics):
        current = df_current.droplevel(0, 1)
        if metrics is METRICS.CoV:
            mean_of_cov = 0
            
            for idx, group in df_pred.groupby(level=0, axis = 1):
                mean_of_cov = mean_of_cov + group.droplevel(0, 1).sub(current).pow(2).mean().pow(1/2).div(current.mean(0)).mean()
            print(mean_of_cov)
            mean_of_cov = mean_of_cov /  len(df_pred.groupby(level=0, axis = 1))
            return (100 * mean_of_cov)
        
        
        if metrics is METRICS.RMSE:
            mean_of_rmse = 0
            for idx, group in df_pred.groupby(level=0, axis = 1):
                mean_of_rmse = mean_of_rmse + group.droplevel(0, 1).sub(current).pow(2).mean().pow(1/2).mean()
            mean_of_rmse = mean_of_rmse /  len(df_pred.groupby(level=0, axis = 1))
            return (mean_of_rmse)
        else:
            raise ValueError("Not Implemented Yet")
            

    

    @staticmethod
    def read_result_dict(result_path):
        all_dict = {}
        for index_path in sorted(Path(result_path).rglob("*")):
            with open(index_path, 'rb') as f:
                x = pickle.load(f)
                for key_outer, value_outer in x.items():
                    # NO INNER DICT
                    if isinstance(value_outer, pd.DataFrame):
                        if key_outer in all_dict:
                            all_dict[key_outer] = pd.concat([all_dict[key_outer], value_outer]).sort_index()
                        else:
                            all_dict[key_outer] = value_outer.sort_index()
                    # Inner dict
                    else:
                        
                        if key_outer not in all_dict:    
                            all_dict[key_outer] = {}
                        for key_inner, value_inner in value_outer.items():
                           
                            if key_inner in all_dict[key_outer]:
                                all_dict[key_outer][key_inner] = pd.concat([all_dict[key_outer][key_inner], value_inner]).sort_index()
                            else:
                                all_dict[key_outer][key_inner] = value_inner.sort_index()
        return all_dict
    @staticmethod
    def read_result_list(result_path):
        all_dict = {}
        for index_path in sorted(Path(result_path).rglob("*")):
            with open(index_path, 'rb') as f:
                x = pickle.load(f)
                for key_outer, value_outer in x.items():
                    # NO INNER DICT
                    if isinstance(value_outer, pd.DataFrame):
                        if key_outer not in all_dict:
                            all_dict[key_outer] = list()
                        all_dict[key_outer].append(value_outer)
                    # Inner dict
                    else:
                        
                        if key_outer not in all_dict:    
                            all_dict[key_outer] = {}
                        for key_inner, value_inner in value_outer.items():
                            if isinstance(value_inner, pd.DataFrame):
                                if key_inner not in all_dict[key_outer]:
                                    all_dict[key_outer][key_inner] = list()
                                all_dict[key_outer][key_inner].append(value_inner)
        return all_dict
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
