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
from plotnine import ggplot, aes, facet_grid, labs, geom_line,geom_point, theme, geom_ribbon,theme_minimal,scale_color_brewer, scale_fill_brewer, xlab, ylab
import plotnine as p9
import torch 
from DartsFCeV import DartsFCeV, DartsFCeVConfig
from fastai.tabular.all import *
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import dtaidistance
from FCeV import FCeVResultsData
import time
import xskillscore as xs
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns

size_seaborn_font = 2
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
    CRPS = (10,)
    
def get_results(df,FCeV_results_data,  metric, mean):
    input_length = FCeV_results_data["n_forecast"]
    df_events = FCeV_results_data["df_events"]
    df_input = FCeV_results_data["df_input"]
    freq = FCeV_results_data["freq"]
    index_synth = list(FCeV_results_data["df_synthetics"].keys())
    values_synth = [value.max().values[0] for value in FCeV_results_data["df_synthetics"].values()]
    CF_events = pd.DataFrame(values_synth, index = index_synth, columns = ["CF"])
    list_columns = [True if value in CF_events.index else False for value in df.columns.levels[0]]
    remove_columns = df.columns.levels[0][np.logical_not(list_columns)]
    df_current_base = df["current"].sort_index()
    df_pred = df.drop(remove_columns, axis = 1, level = 0)
    df_current_base = df_current_base.loc[df_pred.index].values
    #df_current_base = np.tile(df_current_base, len(df_pred.columns.levels[1]) - 1)
    num_components = len(df_pred.columns.levels[2])
    expected = FCeV_results_data["df_events"].loc[df_pred.index]
    df_results = pd.DataFrame()
    list_result = list()
    start = time.time()
    
    for index in range(len(df_pred) // input_length):
        index_selected = np.arange(index * input_length, (index + 1) * input_length )
        df_test = df_pred.iloc[index_selected]
        start_index = df_test.index.mean().round(freq='s')
        dict_values = {}
        for idx_CF, group_CF in df_test.groupby(level=0, axis = 1):
            df_current = df_current_base[index_selected,:]
            if mean == True:
                group_CF = group_CF.mean(axis = 1,level=2)
                
            if metric == SUMMARY_METRICS.CRPS:
                ds = pd.DataFrame(df_current, columns = ["y"], index = group_CF.index).to_xarray()
                df_score  = pd.melt(group_CF.droplevel(0, 1), var_name='member', value_name='yhat', ignore_index=False)#.reset_index(drop=True)
                df_score.set_index(["member"], append = True, inplace = True)
                ds['yhat'] = df_score.to_xarray()['yhat']
                dict_values[idx_CF] = ds.xs.crps_ensemble('y', 'yhat').to_numpy()
                
            if metric == SUMMARY_METRICS.CoV:
                df_current = np.tile(df_current, group_CF.shape[1] // num_components)
                dict_values[idx_CF] = np.median(np.divide(np.power(np.mean(np.power(np.subtract(group_CF,df_current),2),0),1/2), np.mean(df_current,0)))
            elif metric == SUMMARY_METRICS.RMSE:
                df_current = np.tile(df_current, group_CF.shape[1] // num_components)
                dict_values[idx_CF] = np.mean(np.power(np.mean(np.power(np.subtract(group_CF,df_current),2),0),1/2))
            elif metric is SUMMARY_METRICS.ZSCORE:
                group_CF = group_CF.mean(axis = 1,level=2)
                df_current = df_current_base[index_selected,:]
            elif metric is SUMMARY_METRICS.CosSim:
                df_current = np.tile(df_current, group_CF.shape[1] // num_components)
                dict_values[idx_CF]  = 1 - np.mean(np.diag(cosine_similarity(np.transpose(df_current), np.transpose(group_CF))))
            elif metric is SUMMARY_METRICS.DTW:
                import dtaidistance
            elif metric is SUMMARY_METRICS.PRESS:
                dict_values[idx_CF] =  np.mean(np.sum(np.power(np.subtract(df_current,group_CF), 2),0))

        result = pd.DataFrame(dict_values,  index = [group_CF.index.mean()])
        #result["pred"] = result["diff"] > 20
        result["current"] = np.max(expected.iloc[index_selected])[0]  
        result["current_time"] = np.argmax(expected.iloc[index_selected])
        list_result.append(result)
    print(f"Duration: {time.time()-start}")   
    df_result = pd.concat(list_result)
    return df_result

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
            
            
            
def plot_scatter_hour(df_result, current_values, CF, path, palete):
    
    df_violin = df_result.copy()
    df_violin = df_violin[current_values > 0]
    df_violin.columns = df_violin.columns.str.split("-", expand=True)
    column = df_violin.pop("EMPTY")
    ref = column
    time = df_violin["current_time"].squeeze()
    df_violin = df_violin.drop(["current", "current_time"], 1)
    df_violin = pd.DataFrame(df_violin.T.values / ref.T.values, index = df_violin.columns).T
    df_violin = df_violin[CF].sort_index(1)
    df_violin = df_violin.min(level = 1, axis = 1)
    columns = df_violin.columns

    selected = df_violin.apply(lambda x: columns[np.argmin(x)], axis = 1)
    selected_min = df_violin.apply(lambda x: min(x), axis = 1)

    df_visualize = pd.DataFrame([selected.values, selected_min.values, time.values]).T
    df_visualize.columns = columns=["H", "CFR", "Ground Truth Hour"]
    df_visualize["H"] = df_visualize["H"].astype('int')
    df_visualize["Ground Truth Hour"] = df_visualize["Ground Truth Hour"].astype('int')
    sns.set(rc={'figure.figsize':(20,8)})
    sns.set(font_scale = size_seaborn_font)
    # Create a visualization
    # Set the palette using the name of a palette:
    #sns.color_palette("Spectral", as_cmap=True)
    #sns.set_palette("Spectral")
    sns_plot = sns.scatterplot(
        data=df_visualize,
        x="Ground Truth Hour", y="CFR",hue = 'H',  s=600, alpha=1, palette= sns.color_palette(palete, as_cmap=True)
    )
    sns_plot.legend(title='hypotesis', loc='upper left', frameon=True)
    for lh in sns_plot.legend_.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [200] 

    
        # You can also use lh.set_sizes([50])
    fig = sns_plot.get_figure()
    
    fig.savefig(path, bbox_inches='tight')


def plot_roc(df_result, current_values, CF_level, CF_hour,treshold_list, path):

    df_compress = df_result.copy()
    df_ref = df_compress[["EMPTY"]]

    df_compress.columns = df_compress.columns.str.split("-", expand=True)
    df_compress = df_compress[CF_level]
    df_compress.head()

    df_compress = df_compress.mean(level=1, axis = 1)
    df_compress = df_compress[CF_hour]
    df_compress = df_compress.mean(1)
    df_compress = -(df_compress.values.T  /df_ref.values.T)
    df_compress = pd.DataFrame(df_compress).T


    list_roc = list()
    list_index = list()
    for threshold in treshold_list:
        y_pred_prob = df_compress.mean(1)
        #y_test = (df_result["current"] >thrshold_precipitation)  & (df_result["current_time"] > 8) & ( df_result["current_time"] < 18)
        y_test = (df_result["current"] > threshold)  
        FPR, TPR, _ = sklearn.metrics.roc_curve(y_test,  y_pred_prob, drop_intermediate = False)
        pd_roc = pd.DataFrame([FPR,TPR]).T
        pd_roc.columns = ["FPR", "TPR"]
        list_roc.append(pd_roc)
        list_index.append(f"{threshold}")


    pd_all_roc = pd.concat(list_roc, keys = list_index, axis =1)

    if len(pd_all_roc.columns.levels[0]) > 1:
        pd_all_roc = pd_all_roc.unstack().unstack(level=1).reset_index(level=1, drop=True).rename_axis('level').reset_index()
    else:
        pd_all_roc = pd_all_roc.unstack().unstack(level=1).reset_index(level=1, drop=True).rename_axis('level').reset_index()
    sns.set(rc={'figure.figsize':(10,10)})
    sns.set_palette("pastel")
    sns.set(font_scale = size_seaborn_font)
    sns_plot = sns.lineplot(data=pd_all_roc, x="FPR", y="TPR", hue="level", linewidth=3)
    sns_plot.legend(title='Level', loc='lower right', frameon=True)
    fig = sns_plot.get_figure()
    fig.savefig(path, bbox_inches='tight')
    
    
    
def plot_roc_hours(df_result, CF_level, CF_hour, path):
    df_compress = df_result.copy()
    df_ref = df_compress[["EMPTY"]]

    df_compress.columns = df_compress.columns.str.split("-", expand=True)
    df_compress = df_compress[CF_level]
    df_compress.head()

    df_compress = df_compress.mean(level=1, axis = 1)
    df_compress = df_compress[CF_hour]
    df_compress = df_compress.mean(1)
    df_compress = -(df_compress.values.T  /df_ref.values.T)
    df_compress = pd.DataFrame(df_compress).T


    list_roc = list()
    list_index = list()

    thrshold_precipitation = 1
    list_roc = list()
    list_index = list()
    for hour_offset in np.arange(0,13,2):
        y_pred_prob = df_compress.mean(1)
        y_test = (df_result["current"] )  & (df_result["current_time"] >= hour_offset ) & ( df_result["current_time"] < 12 + hour_offset)
        FPR, TPR, _ = sklearn.metrics.roc_curve(y_test,  y_pred_prob, drop_intermediate = False)
        pd_roc = pd.DataFrame([FPR,TPR]).T
        pd_roc.columns = ["FPR", "TPR"]
        list_roc.append(pd_roc)
        list_index.append(f"{hour_offset}:00 to {hour_offset+12}:00")  
    pd_all_roc = pd.concat(list_roc, keys = list_index, axis =1)

    if len(pd_all_roc.columns.levels[0]) > 1:
        pd_all_roc = pd_all_roc.unstack().unstack(level=1).reset_index(level=1, drop=True).rename_axis('level').reset_index()
    else:
        pd_all_roc = pd_all_roc.unstack().unstack(level=1).reset_index(level=1, drop=True).rename_axis('level').reset_index()
    sns.set(rc={'figure.figsize':(10,10)})
    sns.set_palette("pastel")
    sns.set(font_scale = size_seaborn_font)
    sns_plot = sns.lineplot(data=pd_all_roc, x="FPR", y="TPR", hue="level", linewidth=3)
    sns_plot.legend(title='Level', loc='lower right', frameon=True)
    fig = sns_plot.get_figure()
    fig.savefig(path, bbox_inches='tight')

def plot_violin_level(df_result, current_values, CF, path, limit = None):
    
    df_violin = df_result.copy()
    df_violin.columns = df_violin.columns.str.split("-", expand=True)
    column = df_violin.pop("EMPTY")
    ref = column
    df_violin = df_violin.drop(["current", "current_time"], 1)
    df_violin = df_violin.loc[:, pd.IndexSlice[:, CF]]
    df_violin = pd.DataFrame(df_violin.T.values / ref.T.values, index = df_violin.columns).T
    df_violin["class"] = current_values
    sns.set(rc={'figure.figsize':(20,8)})
    df_violin = pd.melt(df_violin, id_vars="class",var_name='hypotesis Level', value_name='CFR')
    sns.set(font_scale = size_seaborn_font)
    if len((df_violin["class"].unique() == 2)):
        split = True
    else: 
        split = False
    sns.set_palette("pastel")
    sns_plot = sns.violinplot(data=df_violin, x = "hypotesis Level", y="CFR", hue = "class",split = split,   showfliers = False, inner = "quartile", scale = "width")
   
    sns_plot.legend(title='Ground Truth', loc='upper left', frameon=True)
    if limit is not None:
        sns_plot.set(ylim=limit)
    fig = sns_plot.get_figure()
    fig.savefig(path, bbox_inches='tight')
    
    
def plot_violin_hour(df_result,current_values, CF, path, limit = None):
    
    df_result_split = df_result.copy()
    ref = df_result_split[["EMPTY"]]
    df_result_split.columns = df_result_split.columns.str.split("-", expand=True)
    df_result_split = df_result_split[CF].sort_index(1)
    df_result_split = df_result_split.mean(level=1, axis = 1)
    df_result_split.columns = [int(values) for values in df_result_split.columns]
    df_violin = df_result_split.copy()
    df_violin = pd.DataFrame(df_violin.T.values / ref.T.values, index = df_violin.columns).T
    df_violin["class"] = current_values
    if len(df_violin["class"].unique()) == 2:
        split = True
    else: 
        split = False
    print(split)
    sns.set(rc={'figure.figsize':(20,8.27)})
    sns.set(font_scale = size_seaborn_font)
    df_violin = pd.melt(df_violin, id_vars="class",var_name='Hypothesis Hours', value_name='CFR')
    sns.set_palette("pastel")
    sns_plot = sns.violinplot(data=df_violin, x = "Hypothesis Hours", y="CFR", hue = "class", split=split, showfliers = False, inner = "quartile", scale = "width")
    sns_plot.legend(title='Ground Truth', loc='upper left', frameon=True)
    if limit is not None:
        sns_plot.set(ylim=limit)
    
    fig = sns_plot.get_figure()
    fig.savefig(path, bbox_inches='tight')
    
    
    
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
            

def plot_results(df_current, df_pred, start_day, end_day, name = "Prediction", size = (15, 6)):
    df_current = df_current[(df_current.index > start_day) & (df_current.index <= end_day)]
    df_pred = df_pred[(df_pred.index > start_day) & (df_pred.index <= end_day)]
    
    current = pd.DataFrame(df_current.mean(1), columns =  df_current.columns.levels[-1])
    pred = pd.DataFrame(df_pred.mean(1), columns =  df_pred.columns.levels[-1])
    uncer =  pd.DataFrame(df_pred.std(1), columns =  df_pred.columns.levels[-1])
    label_forecast = ["current", "pred", "uncer"]
    values_forecast = [current, pred,  uncer]
    current_forecast = pd.concat(values_forecast,  keys = label_forecast, axis= 1)
    current_forecast.columns = current_forecast.columns.set_names("component", level = 1)
    current_forecast = current_forecast.stack(level=1).reset_index(1)
    if "uncer" in current_forecast.columns:
        current_forecast["uncer_min"] = current_forecast['pred'] - current_forecast['uncer']
        current_forecast["uncer_max"] = current_forecast['pred'] + current_forecast['uncer']
    else:
        current_forecast["uncer_min"] = current_forecast['pred']
        current_forecast["uncer_max"] = current_forecast['pred']

    plot = (ggplot(current_forecast.reset_index()) +  # What data to use
         aes(x="ds", color = "component", fill = "component")  # What variable to use
        + geom_ribbon(aes(y = "pred", ymin = "uncer_min", ymax = "uncer_max"), alpha = .2, color = 'none') #
        + geom_line(aes(y="pred"),linetype="dashed",size = 1.5 )  # Geometric object to use for drawing 
        + geom_line(aes(y="current"),size = 1.5)  # Geometric object to use for drawing 
        + theme_minimal() 
        +theme(legend_position="bottom", figure_size=size, axis_title_x=p9.element_text(size=16, colour="black"),axis_title_y=p9.element_text(size=16, colour="black"))
        +xlab("Time")
        + ylab(name)
            
        + labs(color='none')
        + scale_color_brewer(type="qual", palette="Set2", name="none", guide= False)
        + scale_fill_brewer(type="qual", palette="Set2", name="none", guide= False))
    return plot


def plot_compare_results(df_pred_0, df_pred_1, start_day, end_day):
    df_pred_0 = df_pred_0[(df_pred_0.index > start_day) & (df_pred_0.index <= end_day)]
    df_pred_1 = df_pred_1[(df_pred_1.index > start_day) & (df_pred_1.index <= end_day)]
    
    pred_0 = pd.DataFrame(df_pred_0.mean(1), columns =  df_pred_0.columns.levels[-1])
    pred_1 = pd.DataFrame(df_pred_1.mean(1), columns =  df_pred_1.columns.levels[-1])
    label_forecast = ["pred_0", "pred_1"]
    values_forecast = [pred_0, pred_1]
    current_forecast = pd.concat(values_forecast,  keys = label_forecast, axis= 1)
    current_forecast.columns = current_forecast.columns.set_names("Predicted value", level = 0)
    current_forecast.columns = current_forecast.columns.set_names("component", level = 1)
    current_forecast = current_forecast.stack(level=1).reset_index(1)

    plot = (ggplot(current_forecast.reset_index()) +  # What data to use
         aes(x="ds", y = "Predicted value",color = "component", fill = "component")  # What variable to use
        + geom_line(aes(y="pred_0"),size = 1)  # Geometric object to use for drawing 
        + geom_line(aes(y="pred_1"),size = 1 , linetype= "dotted")  # Geometric object to use for drawing 
        + theme_minimal() 
        +theme(legend_position="bottom", figure_size=(15, 6))
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




def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig