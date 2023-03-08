import dateutil.parser
from pathlib import Path
from enum import Enum
import pickle
from IPython.display import display
from IPython.core.display import HTML
import json
from fastcore import foundation
from fastai import tabular
import fastai
from plotnine import ggplot, aes, facet_grid, labs, geom_line,geom_point, theme, geom_ribbon,theme_minimal,scale_color_brewer
import torch 
from DartsFCeV import DartsFCeV, DartsFCeVConfig
from fastai.tabular.all import *
from sklearn.metrics.pairwise import cosine_similarity
import dtaidistance
from FCeV import FCeVResultsData


class SUMMARY_METRICS(Enum):
    CoV = (0,)
    mape = (1,)
    marre = (2,)
    RMSE = (3,)
    ZSCORE = (4,)
    CosSim = (5,)
    PRESS = (6,)
    RMSE_sigma = (7,)
    DTW = (8, )
    ZDENS = (9, )
    
    
    
def get_summary(current, pred, metrics):
        if metrics is SUMMARY_METRICS.CoV:
            return (
                (current
                .sub(pred)
                .pow(2)
                .pow(1 / 2) / current).mean()
            ) * 100
        elif metrics is SUMMARY_METRICS.RMSE:
            return np.mean(np.power(np.mean(np.power(np.subtract(pred,current),2),0),1/2))
        elif metrics is SUMMARY_METRICS.PRESS:
            return np.mean(np.sum(np.pow(np.subtract(pred,current), 2),0))
        elif metrics is SUMMARY_METRICS.RMSE_sigma:
            return (
            current
            .sub(pred)
            .pow(2).add(2 * uncer.pow(2)).pow(1/2).mean()
        )
        elif metrics is SUMMARY_METRICS.DTW:
            return current.combine(pred, lambda x, y: np.repeat(dtaidistance.dtw.distance_fast(x.values.astype("double"),y.values.astype("double")), len(x))).mean().mean()
        elif metrics is SUMMARY_METRICS.ZDENS:
            if uncer.empty:
                raise ValueError("For compute Z score the uncertainty has to be included")
            return pd.DataFrame(scipy.stats.norm(pred,uncer ).pdf(current), columns = uncer.columns, index = uncer.index).mean().mean()
        elif metrics is SUMMARY_METRICS.ZSCORE:
            if uncer.empty:
                raise ValueError("For compute Z score the uncertainty has to be included")
            return pd.DataFrame(scipy.stats.norm().pdf(current.sub(pred).div(uncer)), columns = uncer.columns, index = uncer.index)
        elif metrics is SUMMARY_METRICS.CosSim:
            if uncer.empty:
                raise ValueError("For compute Z score the uncertainty has to be included")
            total_sim = 10
            value = 0

            for i in range(total_sim):
                value = value + np.diag(cosine_similarity(pred.T, current.T.add(np.random.normal(1) * uncer.T))).mean()
            return 1 - (value / total_sim)
        else:
            raise ValueError("Not Implemented Yet")
            
    
def get_summary2(current, pred, metrics, uncer = pd.DataFrame()):
        if metrics is SUMMARY_METRICS.CoV:
            return (
                (current
                .sub(pred)
                .pow(2)
                .pow(1 / 2) / current).mean()
            ) * 100
        elif metrics is SUMMARY_METRICS.RMSE:
            return (
            current
            .sub(pred)
            .pow(2)
            .pow(1 / 2).mean()
        ) * 100
        elif metrics is SUMMARY_METRICS.PRESS:
            return (
            current
            .sub(pred)
            .pow(2).sum()
        )
        elif metrics is SUMMARY_METRICS.RMSE_sigma:
            return (
            current
            .sub(pred)
            .pow(2).add(2 * uncer.pow(2)).pow(1/2).mean()
        )
        elif metrics is SUMMARY_METRICS.DTW:
            return current.combine(pred, lambda x, y: np.repeat(dtaidistance.dtw.distance_fast(x.values.astype("double"),y.values.astype("double")), len(x))).mean().mean()
        elif metrics is SUMMARY_METRICS.ZDENS:
            if uncer.empty:
                raise ValueError("For compute Z score the uncertainty has to be included")
            return pd.DataFrame(scipy.stats.norm(pred,uncer ).pdf(current), columns = uncer.columns, index = uncer.index).mean().mean()
        elif metrics is SUMMARY_METRICS.ZSCORE:
            if uncer.empty:
                raise ValueError("For compute Z score the uncertainty has to be included")
            return pd.DataFrame(scipy.stats.norm().pdf(current.sub(pred).div(uncer)), columns = uncer.columns, index = uncer.index)
        elif metrics is SUMMARY_METRICS.CosSim:
            if uncer.empty:
                raise ValueError("For compute Z score the uncertainty has to be included")
            total_sim = 10
            value = 0

            for i in range(total_sim):
                value = value + np.diag(cosine_similarity(pred.T, current.T.add(np.random.normal(1) * uncer.T))).mean()
            return 1 - (value / total_sim)
        else:
            raise ValueError("Not Implemented Yet")
            

def plot_results(df_forecast):
    current_forecast = df_forecast.stack(level=1).reset_index(1)
    if "uncer" in current_forecast.columns:
        current_forecast["uncer_min"] = current_forecast['pred'] - current_forecast['uncer']
        current_forecast["uncer_max"] = current_forecast['pred'] + current_forecast['uncer']
    else:
        current_forecast["uncer_min"] = current_forecast['pred']
        current_forecast["uncer_max"] = current_forecast['pred']

    plot = (ggplot(current_forecast.reset_index()) +  # What data to use
         aes(x="ds")  # What variable to use
        + geom_ribbon(aes(y = "pred", ymin = "uncer_min", ymax = "uncer_max", fill = "component"), alpha = .4) 
        + geom_line(aes(y="current", color = "component"),size = 1.5)  # Geometric object to use for drawing
        + geom_line(aes(y="pred", color = "component"),linetype="dashed",size = 1.5 )  # Geometric object to use for drawing
        + theme_minimal() 
        +theme(legend_position="bottom", figure_size=(10, 6))
        + scale_color_brewer(type="qual", palette="Set1")
            )
    return plot

def predict_with_tabai(df_tab, df_test, dropout, layers):
        
        y_name = "current"
        min_y = np.min(df_tab[y_name]) - 1
        df_tab[y_name] = np.log(df_tab[y_name] - min_y)
        len_train = int(len(df_tab) * 0.8)
        df_train = df_tab.iloc[:len_train];
        df_val = df_tab.iloc[len_train + 1:];
        splits = (foundation.L(range(len(df_train))), foundation.L(range(len(df_train) + 1, len(df_train) + len(df_val))))
        to = fastai.tabular.core.TabularPandas(df_tab, procs=[tabular.core.FillMissing, tabular.core.Normalize],
                       cont_names = list(df_tab.drop(y_name, axis = 1).columns.values),
                       y_block=fastai.data.block.RegressionBlock(),
                       y_names=y_name,
                       splits=splits)
        dls = to.dataloaders(bs=200)
        max_log_y = np.max(df_tab[y_name])*1.2
        y_range = torch.tensor([0, max_log_y]); y_range
        tc = tabular.model.tabular_config(ps=[0.1, 0.01], embed_p=dropout, y_range=y_range)
        learn = tabular.learner.tabular_learner(dls, layers=[layers,layers],
                                metrics=fastai.metrics.exp_rmspe,
                                config=tc,
                                loss_func=fastai.losses.MSELossFlat())
        learn.recorder.silent = True
        with learn.no_bar(), learn.no_logging():
            lr = learn.lr_find(show_plot=False)
            learn.fit_one_cycle(100, lr)

        dl = learn.dls.test_dl(df_test)
        raw_test_preds = learn.get_preds(dl=dl)
        learn.validate(dl=dl)
        test_preds = (np.exp(raw_test_preds[0])+ min_y).numpy().T[0]
        df_test["pred_ai"] = test_preds
        return df_test
    
def predict_from_metrics(df, df_events, metric, input_length, synth):
        selected_elements = [True if element in str(list(synth.index)) else False for element in df.columns.levels[0]]
        remove_columns = df.columns.levels[0][np.logical_not(selected_elements)].drop("current")
        df_test = df.drop(remove_columns, axis = 1, level = 0).head()
        df_results = pd.DataFrame()
        for index in range(len(df) // input_length):
            df_test = df.iloc[index * input_length: (index + 1) * input_length - 1]
            list_values = list()
            list_index = list()
            list_colums = list()
            start_index = df_test.index.mean().round(freq='s')
            expected = df_events.loc[df_test.index]
            df_forecast.columns = df_forecast.columns.str.split("_", expand=True)
            for name, group in df_test.drop(["BASE", "EMPTY"], axis = 1).groupby(level='CF', axis = 1):
                pred = group[name]["pred"]
                current = group[name]["current"]
                uncer = group[name]["uncer"]
                value = 0 # np.random.normal(0, 1, 4)
                simulation = pred + value * uncer
                curret_metrics = get_summary(current, simulation, metric, uncer).mean().mean()
                list_values.append(pd.DataFrame(curret_metrics, columns = synth.loc[name].values, index = [start_index]))
            if metric == SUMMARY_METRICS.ZSCORE or metric == SUMMARY_METRICS.ZDENS:
                df_out = pd.concat(list_values, axis = 1)
                pred1 = df_out.idxmax(1).values[0]
                num_elements = 4
                df_prob = df_out.copy()
                highest_prob = pd.DataFrame(-1*np.sort(-1*df_prob.values,axis=1)[:, :num_elements])
                highest_prob_scaled = highest_prob.sub(highest_prob.min(1),0).div(highest_prob.max(1).sub(highest_prob.min(1)),0)
                #highest_prob_scaled = highest_prob.div(highest_prob.max(1))
                highest_prob_scaled_norm = highest_prob_scaled.div(highest_prob_scaled.sum(1), axis=0)
                inference = df_prob.columns[np.argsort(-1*df_prob.values,axis=1)[:, :num_elements]]
                pred2 = (highest_prob_scaled_norm * inference).sum(1).values[0]
                df_out["pred"] = pred1
                df_out["pred2"] = pred2
                df_out["current"] = expected.mean().values[0]
                df_results = pd.concat([df_results, df_out])
            else:
                df_out = pd.concat(list_values, axis = 1)
                df_out["pred"] = df_out.idxmin(1).values[0]
                df_out["current"] = expected.mean().values[0]
                df_results = pd.concat([df_results, df_out])
        return df_results
    
def read_result(result_path):
        all_dict = {}
        value_list = list()
        key_list = list()
        result_config = list(Path(result_path).rglob("*config.cpkl"))
        FCeV_results_data = None
        if len(result_config) == 1:
            with open(result_config[0], 'rb') as f:
                FCeV_results_data = pickle.load(f)
        else:
            print(f"No config files: {len(result_config)}")
        list_pandas = list()
        for index_path in sorted(Path(result_path).rglob("*.pkl")):
            with open(index_path, 'rb') as f:
                x = pickle.load(f)
                list_pandas.append(x)
        df_result = pd.concat(list_pandas, names=["CF", "type", "component"])
        return df_result, FCeV_results_data
    
def read_result2(result_path):
    all_dict = {}
    value_list = list()
    key_list = list()
    result_config = list(Path(result_path).rglob("*config.cpkl"))
    FCeV_results_data = None
    if len(result_config) == 1:
        with open(result_config[0], 'rb') as f:
            FCeV_results_data = pickle.load(f)
    else:
        print(f"No config files: {len(result_config)}")
    for index_path in sorted(Path(result_path).rglob("*.pkl")):
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
                    for key_inner, value_inner in value_outer.items():
                        name = f"{key_outer}_{key_inner}"
                        if name in all_dict:
                            all_dict[name] = pd.concat([all_dict[name], value_inner]).sort_index()
                        else:
                            all_dict[name] = value_inner.sort_index()
    value_list = [values for values in all_dict.values()]
    key_list = [keys for keys in all_dict.keys()]
    df_result = pd.concat(value_list, keys = key_list, axis = 1, names=["CF", "type", "component"])
    return df_result, FCeV_results_data