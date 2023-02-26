from neuralprophet import NeuralProphet, forecaster, load
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from aux_function_SR import get_eq_filtered, SR_SENSORS
from NPw_aux import drop_scale, prepare_eq
import numpy as np
from sklearn.metrics import  confusion_matrix
import dateutil.parser
from pathlib import Path
from enum import Enum
from typing_extensions import Literal
from pytorch_lightning.callbacks import EarlyStopping
from IPython.display import display
from IPython.core.display import HTML
import json
import torch
import darts
from darts.dataprocessing import transformers

from darts.models import TCNModel, TFTModel, RNNModel, NBEATSModel, NHiTSModel, TransformerModel, NLinearModel

# from FCeV import FCeVConfig
covariate_types = {"Future", "Past"}
class DartModelMetaClass(type):
    """ Metaclass to force Model class to contains specific attributes """
    def __new__(cls, name, bases, dct):
        if "covariate_type" not in dct:
            raise TypeError("Covariate type has to be define in the dataclass model config")

        if dct["covariate_type"] not in covariate_types:
            raise ValueError(f"Covariate type has to be one of the following {covariate_types}")

        x = super().__new__(cls, name, bases, dct)
        return x
@dataclass
class DartsFCeVConfig:
    """

    Attributes:
        forecast_length:
        question_mark_length:
    """

    DartsModelConfig: None
    learning_rate: float
    use_gpu: bool
    event_type: Literal["None", "Binary", "Non-Binary"]
    n_layers: int
    internal_size: int
    dropout: float
    batch_size: int
    n_epochs: int
    patience: int
    seed: int
    probabilistic: bool

@dataclass
class TCNDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Past"
    dilation_base: float
    weight_norm: bool

@dataclass
class NHITSDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Past"
    num_stacks :int 
    # num_layers: int 
    num_blocks:int
    # layer_widths: int
    expansion_coefficient_dim: int
    max_pool_1d: bool

@dataclass
class TransformerDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Past"
    # d_model: int
    n_head_divisor: int
    dim_feedforward: int
    activation: Literal["relu", "GLU", "Bilinear", "ReGLU", "GEGLU"] = "relu"
    norm_type: Literal[None, "LayerNorm"  "RMSNorm ", "LayerNormNoBias"] =None


@dataclass
class RNNDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Future"
    RNNmodel: Literal['RNN', 'LSTM', 'GRU'] = "GRU"

@dataclass
class NLinearDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Future"
    const_init: bool#Hidden dim

@dataclass
class NBEATSDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Past"
    num_stacks :int 
    # num_layers: int 
    num_blocks:int
    # layer_widths: int
    expansion_coefficient_dim: int


@dataclass
class TFTDartsFCeVConfig(metaclass = DartModelMetaClass):
    covariate_type = "Future"
    # lstm_layers: int
    # hidden_size: int
    num_attention_heads: int
    full_attention: bool
    add_relative_index: bool
    hidden_continuous_size: int


class DartsFCeV:
    def __init__(
        self,
        FCeV_config,
        Darts_FCeV_config,
        df_input,
        df_covariates,
        df_events,
        synthetic_events=pd.DataFrame(),
    ):
        """

        Argsn
            FCeV_config ():
            Darts_FCev_config ():
            df_input ():
            df_past_covariates ():
            df_events ():
            synthetic_events ():
        """

        self.FCeV_config = FCeV_config
        if not isinstance(Darts_FCeV_config, DartsFCeVConfig):
            raise ValueError("Not valid FCev config file")
        self.Darts_FCeV_config = Darts_FCeV_config
        # self.input_series =  darts.TimeSeries.from_dataframe(
        #     df_input
        # ) 
        self.input_series =  self.df_to_ts(df_input)
        self.n_forecasts = int(self.FCeV_config.forecast_length / self.FCeV_config.freq)
        self.n_lags = int(self.FCeV_config.input_length / self.FCeV_config.freq)
        if not df_covariates.empty:
            self.covariates = self.df_to_ts(df_covariates)
        else:
            self.covariates = None
        self.df_events = df_events
        # Convert from wide format to long format
        if synthetic_events.empty:
            self.synthetic_events = None  # self.get_synthetic_events()
        else:
            self.synthetic_events = self.prepare_synthetic_events(synthetic_events, 5)


        self.index_date = df_input.index
        self.folds = pd.DataFrame(columns=["start_date", "end_date"])

        #Create the Darts model
        self.model = self.create_dart_model()


    def df_to_ts(self, df):
        """ Function to convert a pandas dataframe to a Darts timeseries
    
        Args:
            df (): Input dataframe

        Returns:
           TimeSeries asociate to the Dataframe input 
        """
        df.fillna( method= "bfill")
        df = df.fillna(method= "ffill")
        df = df.fillna(0.0)
        ts =  darts.TimeSeries.from_dataframe(df, fill_missing_dates=True, freq=self.FCeV_config.freq)
        ts =  darts.utils.missing_values.fill_missing_values(ts , fill=0.0)
        return ts
    def start_date(self):
        """Function to get the first date of the timeseries in the upper class FCeV

        Returns:
           First date of the TImeseries 
        """
        return self.input_series.start_time()

    def duration(self):
        """ Function to get the total duration fo the timeseries from the upper class

        Returns:
            total lenght of the timeseries
            
        """
        return self.input_series.duration
    def preprocess_series(self, series, drop_before,drop_after, split_after):
        series = series.astype("float32")
        series = series.drop_before(drop_before)
        series = series.drop_after(drop_after)
        scaler = transformers.Scaler()
        train_series, val_series = series.split_before(split_after)
        train_series = scaler.fit_transform(train_series)
        val_series = scaler.transform(val_series)
        series_scaler = scaler.transform(series)
        return train_series, val_series, series_scaler, scaler
    def prepare_synthetic_events(self,synthetic_events,  periods):
        if synthetic_events is None:
            raise ValueError("Syntheticevents not valid")
        all_dict = {}
        for idx, synthetic_event in synthetic_events.iterrows():
            event_offset = [
                int(self.n_forecasts/ (periods + 1) * x)
                for x in range(1, periods + 1)
            ]
            current_dict = {}
            for idx_offset in event_offset:
                current_event = pd.DataFrame(0.0, index=np.arange(self.n_forecasts), columns = synthetic_events.columns)
                current_event.iloc[idx_offset] = synthetic_event
                current_dict[f"{idx_offset}"] = current_event
            all_dict[f"SYNTH_{idx}"] = current_dict
        return all_dict

    def process_forecast(self, start_datetime, end_datetime):
        """Train the model and process a forecast

        Args:
            start_datetime (): start time for the trining data set
            end_datetime ():

        Returns:

        """
        start_forecast_time = end_datetime - self.FCeV_config.forecast_length
        question_mark_start = (
            start_forecast_time - self.FCeV_config.question_mark_length
        )
        print(question_mark_start)

        df_events = self.df_events.iloc[self.df_events.index < question_mark_start]
        df_events = self.df_events.reindex(self.index_date).fillna(0.0)
        series_ts = self.df_to_ts(df_events)

        # Takes events up to that point
        train_base, val_base, series_base, scaler_base = self.preprocess_series(
            self.input_series, start_datetime, end_datetime, start_forecast_time
        )
        if self.covariates is not None:
            (
                train_covariate,
                val_covariate,
                series_covariate,
                scaler_covariate,
            ) = self.preprocess_series(self.covariates,  start_datetime, end_datetime, start_forecast_time)
        else:
            train_covariate = None
            val_covariate = None
            series_past_covariate = None
            series_covariate = None

        (
                train_events,
                val_events,
                series_events,
                scaler_events,
        ) = self.preprocess_series(series_ts, start_datetime, end_datetime, start_forecast_time)
        # Todo Improve val 
        training_val = series_base[-(len(val_base) + self.n_lags):]


        self.fit(train_base,training_val, series_covariate,  series_events )
        # Compute just one long forecast
        df_forecast = self.test(val_base, series_covariate, series_events,scaler_events)
        return df_forecast   
    def save_df(self, filename):
        path = Path(filename + ".csv")
        for index in range(100):
            if path.is_file():
                path = Path(filename + "_" + str(index + 1) + ".csv")
            else:
                self.npw_df.to_csv(path)
                break
    def fit(self, train_series,val_series, train_covariate, train_events ):
        """

        Args:
            train_series (): 
            val_series (): 
            train_covariate (): 
            train_events (): 
        """
        if train_covariate is not None:
            covariates_with_events = train_covariate.concatenate(train_events, axis = 1)
        else:
            covariates_with_events = train_events

        if self.Darts_FCeV_config.DartsModelConfig.covariate_type == "Future":
            self.model = self.model.fit(train_series, val_series= val_series,val_future_covariates = covariates_with_events, future_covariates=covariates_with_events, epochs = self.Darts_FCeV_config.n_epochs,verbose = self.FCeV_config.verbose)
        else: 
            self.model = self.model.fit(train_series, val_series= val_series,val_past_covariates = covariates_with_events, past_covariates=covariates_with_events, epochs = self.Darts_FCeV_config.n_epochs,verbose = self.FCeV_config.verbose)

    def create_dart_model(self):
        early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=self.Darts_FCeV_config.patience, mode = "min")
        if self.Darts_FCeV_config.use_gpu == True:
            trainer_kwargs = {
                "accelerator": "gpu",
                 "callbacks": [early_stopper],
            }
        else:
            trainer_kwargs = {
                "accelerator": "cpu",
                "devices" : 1,
                 "callbacks": [early_stopper],
            }
        if self.Darts_FCeV_config.probabilistic:
            likelihood = darts.utils.likelihood_models.GaussianLikelihood()
        else:
            likelihood = None
        # trainer_config = {"accelerator":"gpu"}
        if isinstance(self.Darts_FCeV_config.DartsModelConfig, TCNDartsFCeVConfig):
            model = TCNModel(
                input_chunk_length=self.n_lags,
                output_chunk_length=self.n_forecasts,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                dropout=self.Darts_FCeV_config.dropout,
                dilation_base=self.Darts_FCeV_config.DartsModelConfig.dilation_base,
                weight_norm=self.Darts_FCeV_config.DartsModelConfig.weight_norm,
                kernel_size=self.Darts_FCeV_config.internal_size,
                num_filters=self.Darts_FCeV_config.n_layers,
                force_reset=True,
                likelihood=likelihood,
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size
            )
        elif isinstance(self.Darts_FCeV_config.DartsModelConfig, TFTDartsFCeVConfig):
   
            model = TFTModel(input_chunk_length = self.n_lags,
                output_chunk_length = self.n_forecasts,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                hidden_size=self.Darts_FCeV_config.internal_size,
                lstm_layers=self.Darts_FCeV_config.n_layers, 
                num_attention_heads=self.Darts_FCeV_config.DartsModelConfig.num_attention_heads,  
                full_attention=self.Darts_FCeV_config.DartsModelConfig.full_attention, 
                dropout=self.Darts_FCeV_config.dropout,
                add_relative_index=False, 
                force_reset=True,
                likelihood=likelihood,
                loss_fn=torch.nn.MSELoss(),
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size
                             )
        elif isinstance(self.Darts_FCeV_config.DartsModelConfig, RNNDartsFCeVConfig):
   
            model = RNNModel(input_chunk_length = self.n_lags,
                training_length= self.n_forecasts,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                hidden_dim=self.Darts_FCeV_config.internal_size, 
                model=self.Darts_FCeV_config.DartsModelConfig.RNNmodel, 
                n_rnn_layers=self.Darts_FCeV_config.n_layers,  
                dropout=self.Darts_FCeV_config.dropout,
                force_reset=True,
                likelihood=likelihood,
                loss_fn=torch.nn.MSELoss(),
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size
                             )
        elif isinstance(self.Darts_FCeV_config.DartsModelConfig, NBEATSDartsFCeVConfig):
   
            model = NBEATSModel(input_chunk_length = self.n_lags,
                output_chunk_length= self.n_forecasts,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                generic_architecture=True,
                num_stacks=self.Darts_FCeV_config.DartsModelConfig.num_stacks,  
                num_layers=self.Darts_FCeV_config.n_layers,
                num_blocks=self.Darts_FCeV_config.DartsModelConfig.num_blocks,  
                layer_widths=self.Darts_FCeV_config.internal_size,  
                expansion_coefficient_dim=self.Darts_FCeV_config.DartsModelConfig.expansion_coefficient_dim,  
                dropout=self.Darts_FCeV_config.dropout,
                force_reset=True,
                likelihood=likelihood,
                loss_fn=torch.nn.MSELoss(),
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size
                             )
        elif isinstance(self.Darts_FCeV_config.DartsModelConfig, NHITSDartsFCeVConfig):
   
            model = NHiTSModel(input_chunk_length = self.n_lags,
                output_chunk_length= self.n_forecasts,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                num_stacks=self.Darts_FCeV_config.DartsModelConfig.num_stacks,  
                num_layers=self.Darts_FCeV_config.n_layers,  
                num_blocks=self.Darts_FCeV_config.DartsModelConfig.num_blocks,  
                layer_widths=self.Darts_FCeV_config.internal_size,  
                MaxPool1d= self.Darts_FCeV_config.DartsModelConfig.max_pool_1d,  
                dropout=self.Darts_FCeV_config.dropout,
                force_reset=True,
                likelihood=likelihood,
                loss_fn=torch.nn.MSELoss(),
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size
                             )
        elif isinstance(self.Darts_FCeV_config.DartsModelConfig, TransformerDartsFCeVConfig):
                model = TransformerModel(
                input_chunk_length = self.n_lags,
                output_chunk_length= self.n_forecasts,
                dropout=self.Darts_FCeV_config.dropout,
                force_reset=True,
                likelihood=likelihood,
                loss_fn=torch.nn.MSELoss(),
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                d_model=self.Darts_FCeV_config.internal_size,  
                nhead= self.Darts_FCeV_config.internal_size // self.Darts_FCeV_config.DartsModelConfig.n_head_divisor,  
                num_encoder_layers=self.Darts_FCeV_config.n_layers,       
                num_decoder_layers=self.Darts_FCeV_config.n_layers,       
                dim_feedforward=self.Darts_FCeV_config.DartsModelConfig.dim_feedforward, 
                activation=self.Darts_FCeV_config.DartsModelConfig.activation, 
                norm_type= self.Darts_FCeV_config.DartsModelConfig.norm_type, 
                )
        elif isinstance(self.Darts_FCeV_config.DartsModelConfig, NLinearDartsFCeVConfig):
                model = NLinearModel(
                input_chunk_length = self.n_lags,
                output_chunk_length= self.n_forecasts,
                force_reset=True,
                likelihood=likelihood,
                loss_fn=torch.nn.MSELoss(),
                pl_trainer_kwargs=trainer_kwargs,
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                optimizer_kwargs = {"lr": self.Darts_FCeV_config.learning_rate},
                random_state = self.Darts_FCeV_config.seed,
                batch_size = self.Darts_FCeV_config.batch_size,
                n_epochs=self.Darts_FCeV_config.n_epochs,
                const_init= self.Darts_FCeV_config.DartsModelConfig.const_init, 
            )
        else:
            raise ValueError("ModelNotImplemented")
        return model

    def test(self, val_series, val_covariate,  val_events, scaler_events):
        forecast_result = {}
        forecast_uncer = {}
        df_forecast = self.one_step_test( val_series,val_covariate, val_events)
        forecast_result["BASE"] = df_forecast
        # forecast_uncer["BASE"] = df_uncer
        if self.synthetic_events is None:
            return forecast_result, forecast_uncer
 
        for synth_key, synth_dict in self.synthetic_events.items():
            offset_forecast = {}
            offset_uncertainty = {}
            for offset, df_offset in synth_dict.items():
                df_synth = df_offset.set_index(val_series.time_index)
                ts_syhtn = self.df_to_ts(df_synth)
                if self.Darts_FCeV_config.DartsModelConfig.covariate_type == "Past":
                    ts_syhtn = ts_syhtn.shift(-self.n_forecasts)
                    ts_syhtn = scaler_events.transform(ts_syhtn)
                    before = val_events.drop_after(ts_syhtn.start_time())
                    after = val_events.drop_before(ts_syhtn.end_time())
                    ts_syhtn = before.append(ts_syhtn).append(after)
                else:
                    ts_syhtn = scaler_events.transform(ts_syhtn)
                    ts_syhtn = val_events.drop_after(ts_syhtn.start_time()).append(ts_syhtn)
                ts_syhtn = ts_syhtn.astype("float32")
                df_forecast = self.one_step_test( val_series, val_covariate, ts_syhtn)
                offset_forecast[f"{offset}"] = df_forecast
                # offset_uncertainty[f"{offset}"] = df_uncertainty
            forecast_result[f"{synth_key}"] = offset_forecast
            # forecast_uncer[f"{synth_key}"] = offset_uncertainty

        return forecast_result
        
    def one_step_test(self,  base, covariate,events):
        if self.Darts_FCeV_config.probabilistic:
            n_samples = 1000
        else:
            n_samples = 1
        if covariate is None:
            covariates_with_events = events
        else: 
            covariates_with_events = covariate.concatenate(events, axis = 1)
        if self.Darts_FCeV_config.DartsModelConfig.covariate_type == "Future":
            forecast = self.model.predict(self.n_forecasts, future_covariates = covariates_with_events, verbose = False,num_samples=n_samples)
        else: 
            forecast = self.model.predict(self.n_forecasts, past_covariates = covariates_with_events, verbose = False,num_samples=n_samples)
        results = base.slice_intersect(forecast)
        if self.Darts_FCeV_config.probabilistic:
            df_forecast = pd.concat([results.pd_dataframe(), forecast.mean().pd_dataframe(), forecast.std().pd_dataframe()], keys= ["current", "pred", "uncer"], axis = 1)
        else:
            df_forecast = pd.concat([results.pd_dataframe(), forecast.mean().pd_dataframe()], keys= ["current", "pred"], axis = 1)
        return df_forecast

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
