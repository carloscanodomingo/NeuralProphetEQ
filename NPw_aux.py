from datetime import timedelta
import pandas as pd
from pathlib import Path
import numpy as np
import dateutil.parser
import pyproj
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
start_day = "2016-01-01"
end_day = "2021-01-01"
freq = timedelta(hours=1)
station_path = "stations/"
data_path = str(Path.home()) + "/data/ion/"
station_list_path = "station_list.csv"
eq_path = str(Path.home()) + "/data/EQ/"

# ToDO the problem is that the currrent algorithm to reduce de number of EQ is based on PR and last too much,
# it would be better to process in the first step usign a base pr criteria and them calculate the real PR
# For each config event.
m1_base = 5.5
m2_base = 7.5
d1_base = 2000
d2_base = 2500


def process_ionosphere_files():
    station_names = Path(data_path + "raw/").glob("*/")
    for station in station_names:
        p = station.rglob("*")
        files = [x for x in p if x.is_file()]
        current_df = pd.DataFrame()
        number = 0
        for file in files:
            current_df = pd.concat([current_df, pd.read_csv((file), compression="xz")])
            number = number + 1
        current_df.rename(columns={"tec": station.name}).set_index("dates").to_csv(
            str(station) + ".csv"
        )


def read_iono_data(station_list, freq):
    stations_files = Path(data_path + station_path).glob("*.csv")
    # print(list(stations_files))
    list_stations = [
        station
        for station in stations_files
        if any(x in station.name for x in station_list)
    ]
    ds = pd.date_range(start=start_day, end=end_day, freq="30s")
    df = pd.DataFrame({"dates": ds}).set_index("dates")
    for station in list_stations:
        current_df = pd.read_csv(station)
        current_df["dates"] = pd.to_datetime(current_df["dates"])
        current_df = current_df.set_index("dates").sort_index()
        df = pd.merge(df, current_df, how="outer", left_index=True, right_index=True)
    df.index.names = ["ds"]
    df = df.resample(rule=freq).mean()
    df.drop(df.index[-1], inplace=True)

    return df


def read_EQ_data(dir_path):
    p = Path(dir_path).glob("*.csv")
    all_pd = pd.DataFrame()

    valid_colums = ["time", "longitude", "latitude", "depth", "mag", "type"]
    for file in p:
        print(file)
        all_pd = pd.concat(
            [
                all_pd,
                pd.read_csv(
                    file, parse_dates=["time"], date_parser=dateutil.parser.parse
                ).loc[:, valid_colums],
            ]
        )
        all_pd["time"] = pd.to_datetime(all_pd["time"])
    all_pd = (
        all_pd[all_pd["type"] == "earthquake"]
        .drop("type", axis=1)
        .set_index("time")
        .tz_convert(None)
        .sort_index()
    )
    all_pd.index.names = ["dates"]

    return all_pd


""


def process_eq(df_events, station_names, freq):
    # Create geodesic object to compute dist and azimuth
    geodesic = pyproj.Geod(ellps="WGS84")
    # Read all the station available:
    df_stations = pd.read_csv(station_list_path).set_index("station_id")
    # Compute the mean lat and long of th eselected stations ToDo: Improve this aspect
    mean_location = df_stations.loc[station_names].describe().loc["mean"]
    # To reduce the number of EQ and also the time for computing
    # lambda function to get the azimuth and dist
    f_geo = lambda x: geodesic.inv(
        mean_location["longitude"],
        mean_location["latitude"],
        x["longitude"],
        x["latitude"],
    )
    df_eq_results = pd.DataFrame(
        df_events.apply(f_geo, axis=1).tolist(), columns=["azimuth", "bazi", "dist"]
    ).drop("bazi", axis=1)
    # Km
    df_events["dist"] = df_eq_results["dist"].values / 1000
    df_events["arc_cos"] = np.abs(np.cos(np.deg2rad(df_eq_results["azimuth"].values)))
    df_events["arc_sin"] = np.abs(
        np.sin(np.deg2rad(90 - df_eq_results["azimuth"].values))
    )
    df_events["pr"] = get_pr(df_events, d1_base, d2_base, m1_base, m2_base)
    # Resample and select the EQ with the lowest value of PR
    # Not the most pretty solution but i cant see any other option.
    # https://github.com/pandas-dev/pandas/issues/47963
    grouper_pr = df_events.resample(freq, origin="2016-01-01 00:00:00")[["pr"]]
    idxmax = [np.argmax(group[1]) for group in grouper_pr if group[1].empty == False]
    grouper_all = df_events.resample(freq, origin="2016-01-01 00:00:00")
    list_groups = [group for group in grouper_all if group[1].empty == False]
    list_series = list()
    for current_group, idx in zip(list_groups, idxmax):
        current_df = current_group[1]
        select_series = current_df.iloc[idx]
        select_series.name = current_group[0]
        list_series.append(select_series.transpose())
    df_events = pd.concat(list_series, axis=1).T.drop(
        ["pr", "longitude", "latitude"], axis=1
    )
    df_events.index.names = ["dates"]
    # df_events = df_events.reset_index()

    # Order by standar
    return df_events


def prepare_eq(
    df_events,
    dist_start,
    dist_delta,
    mag_start,
    mag_delta,
    filter,
):
    m1 = mag_start
    m2 = m1 + mag_delta
    d2 = dist_start + dist_delta
    d1 = dist_start
    print("m1: " + str(m1) + " m2: " + str(m2) + " d1: " + str(d1) + " d2: " + str(d2))
    df_events["pr"] = get_pr(df_events, d1, d2, m1, m2)
    if filter == 1:
        df_events = df_events.loc[df_events["pr"] > 1]
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(df_events)
    df_scaled = pd.DataFrame(
        arr_scaled, columns=df_events.columns, index=df_events.index
    )
    return df_scaled, scaler


def get_pr(df, d1, d2, m1, m2):
    (A, B) = get_coef(d1, d2, m1, m2)
    f_percetibility_radius = lambda x: np.divide(
        np.power(10, (x["mag"] * A) + B), x["dist"]
    )
    return df.apply(f_percetibility_radius, axis=1)


def get_coef(d1, d2, m1, m2):

    A = np.log10(d2 / d1) / (m2 - m1)
    B = np.log10(d1) - A * m1
    return (A, B)

def remove_outliers(df, max_z):
    z_score = stats.zscore(df, nan_policy = "omit")
    df[(np.abs(z_score) > max_z)] = np.NaN

def prepare_ion_data(site, freq):
    df_stations = pd.read_csv(station_list_path)
    station_names = list((df_stations[df_stations["site"] == site])["station_id"])
    df_ion = read_iono_data(station_names, freq)
    print(eq_path)
    df_eq = read_EQ_data(eq_path)
    df_eq = process_eq(df_eq, station_names=station_names, freq=freq)
    return df_ion, df_eq


# read_iono_data(["noa1"])
