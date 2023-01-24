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
        aux = aux.iloc[:, :].copy(deep=True)  # selects data up to 26301 row (2004 to 2007 time stamps)
        aux = aux.rename(columns={col: "y"})  # rename column of data to 'y' which is compatible with Neural Prophet
        aux["ID"] = col
        mv_df = pd.concat((mv_df, aux))
    return mv_df

class NPw:
    def __init__(self, config_npw, df,  event_config):
        self.NPw_columns = ['start_forecast', "n_samples", "offset_model", 'RMSE', "MSE", "MAE", "actual_event", "expected_event"]
        self.config_npw = config_npw
        # Convert from wide format to long format
        self.input_df = df
        self.name = get_random_name(separator="_", combo=[COLORS, ANIMALS], style="lowercase") 
        self.input_l_df = multivariate_df(df)
        self.event_config = event_config
        self.input_events = self.get_events()
        self.npw_df = pd.DataFrame(columns = self.NPw_columns)

    def get_next_event(self, start_time):
        loc_eq = np.where(self.input_events  > start_time)[0][0]
        return(self.input_events.iloc[loc_eq])
    
    
    def get_events(self):
        if isinstance(self.event_config, ConfigEQ):
             [_ , earthquake_raw, _] = get_eq_filtered(
            self.event_config.dist_array,
            self.event_config.mag_array,
            self.event_config.lat_array,
            self.event_config.arc_array,
            self.event_config.dist_start,
            self.event_config.dist_delta,
            self.event_config.dist_max,
            self.event_config.lat_max,
            self.event_config.arc_max,
            self.event_config.mag_start,
            self.event_config.mag_delta,
            self.event_config.dist_perct,
            1,
            1,
            SR_SENSORS.NS,
        );
        events_eq = self.input_df["ds"][earthquake_raw > 0.0]
        return events_eq
                
    def add_forecast(self, config_forecast):
        #if isinstance(config_forecast, ConfigForecast) == False:
        #        raise TypeError("Only ConfigForecast are allowed")
                
        if config_forecast.offset_event != None and config_forecast.offset_event > timedelta(minutes = 0) :
            raise Warning("offset_event should be negative")
        model_name = get_random_name(separator="-", combo=[ADJECTIVES, NAMES],style="lowercase") 

        # Insert EQ cases
        start_forecast_time = config_forecast.start_forecast
        question_mark_start = start_forecast_time - self.config_npw.question_mark_length
        # Takes events up to that point
 
        base_df = self.input_l_df[self.input_l_df["ds"] < start_forecast_time +  self.config_npw.forecast_length]
        
        n_samples = len(base_df)
        model = self.create_NP_model()
        # Remove all EQ after question_mark_start
        events = self.input_events[self.input_events  < question_mark_start]

        # Insert with negative offset
        if config_forecast.offset_event is not None:
            events = pd.concat([events, pd.Series(start_forecast_time + config_forecast.offset_event)])
        
        [model, df_with_events] = self.add_eq_neural_prophet(model, base_df, events)
        df_train, df_test = model.split_df(df_with_events, valid_p=1, local_split=True)

        start_fc = config_forecast.start_forecast
        #dif_duration = events - start_fc
        dif_duration = np.asarray([event - start_fc for event  in self.input_events])
        duration_in_s = (dif_duration[np.argmin(np.abs(dif_duration))]).total_seconds() 
        #duration_in_s = (dif_duration.iloc[np.argmin(np.abs(dif_duration))]).total_seconds()  
        actual_event = divmod(duration_in_s, 3600)[0]  
        if config_forecast.offset_event != None:
            expected_event = divmod(config_forecast.offset_event.total_seconds()  , 3600)[0]  
        else:
            expected_event = None
      
        train_metrics = model.fit(df_train)
        test_metrics = self.one_step_test(model, df_test)
        self.npw_df.loc[model_name] = [config_forecast.start_forecast,n_samples,  config_forecast.offset_event, test_metrics["RMSE"], test_metrics["MSE"], test_metrics["MAE"], actual_event, expected_event]
        return test_metrics
        
    def add_eq_neural_prophet(self,model, base_df, events):
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
    def save_df(self,filename):
        path = Path(filename + ".csv")
        for index in range(100):
            if path.is_file():
                path = Path(filename + "_" + str(index + 1)+ ".csv")
            else:
                self.npw_df.to_csv(path)
                break
            
    
    def fit_models(self):
        # https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01
        for index in self.npw_df.index:
            if self.npw_df.loc[index]["is_fit"]  == False:
                model = self.npw_df.loc[index]["model"]
                self.npw_df.at[index, "train_metrics"] = model.fit(self.npw_df.loc[index]["df_train"])
                self.npw_df.at[index, "is_fit"] = True
                self.npw_df.at[index, "model"]  = model

                
    def create_NP_model(self):
        trainer_config = {"accelerator":"gpu"}
        model = NeuralProphet(
        n_forecasts = int(self.config_npw.forecast_length / self.config_npw.freq),
        growth="off",
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=False,
        n_lags= self.config_npw.n_lags,
        num_hidden_layers=self.config_npw.num_hidden_layers,
        d_hidden=self.config_npw.d_hidden,
        learning_rate=self.config_npw.learning_rate,
        #trainer_config = trainer_config,
        trend_global_local="global",
        season_global_local="global",
        )
        model.set_plotting_backend('plotly')
        return model
        

    
    def one_step_test(self,model, df_test):
        future = model.make_future_dataframe(df_test , n_historic_predictions=0)
        forecast = model.predict(future, decompose = False, raw = True)
        y_predicted = forecast.filter(like="step").to_numpy()
        y_actual = pd.pivot(df_test,  index = "ds", columns="ID", values="y")[-48:].transpose().to_numpy()
        RMSE = mean_squared_error(y_actual, y_predicted, squared=False)
        MSE = mean_squared_error(y_actual, y_predicted)
        MAE = mean_absolute_error(y_actual, y_predicted)
        print(MSE)
        return ({"RMSE": RMSE, "MSE": MSE, "MAE": MAE})
    
    # Predict for differents hours offset and differents event offset
    def predict_with_offset_hours(self, start_day, hours_offsets, event_offsets):
        for hours_offset in hours_offsets:
            start_forecast = start_day + relativedelta(hours=+hours_offset)
            self.predict_with_offset_events(start_forecast, event_offsets)
            
    def predict_with_offset_events(self, start_forecast, event_offsets):
        for event_offset in event_offsets:
            config_fc = ConfigForecast(start_forecast = start_forecast, offset_event = event_offset)
            test_metrics = self.add_forecast(config_fc)
            print(str(start_forecast) + " " + str(event_offset) + " " + str(test_metrics))
            
        
    def _____one_step_test(self):
        # https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01
        for index in self.npw_df.index:
            if self.npw_df.loc[index]["is_fit"]  == True:
                future = self.npw_df.loc[index]["model"].make_future_dataframe(self.npw_df.loc[index]["df_test"] , n_historic_predictions=0)
                forecast = self.npw_df.loc[index]["model"].predict(future, decompose = False, raw = True)
                y_predicted = forecast.filter(like="step").to_numpy()
                y_actual = pd.pivot(self.npw_df.loc[index]["df_test"],  index = "ds", columns="ID", values="y")[-48:].transpose().to_numpy()
                self.npw_df.at[index, "test_metrics"] = mean_squared_error(y_actual, y_predicted)
    def load_class(filename):
            with open(filename + ".pkl", "rb") as f:
                return load(f)