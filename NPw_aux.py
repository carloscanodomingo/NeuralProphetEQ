from datetime import timedelta, datetime
import pandas as pd
from pathlib import Path
import numpy as np
import dateutil.parser
import pyproj
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from dataclasses import dataclass, asdict
start_day = "2016-01-01"
end_day = "2021-01-01"
freq = timedelta(hours=1)
station_path = "stations/"
GNSSTEC_path = "GNSSTEC/"
station_list_path = "station_list.csv"
irradiance_path = "irradiance.csv"
eq_path = "EQ/"
stations_path = "stations/"
ion_path = "ion/ionosphere_parameters_data.txt"
# ToDO the problem is that the currrent algorithm to reduce de number of EQ is based on PR and last too much,
# it would be better to process in the first step usign a base pr criteria and them calculate the real PR
# For each config event.
m1_base = 5.5
m2_base = 7.5
d1_base = 2000
d2_base = 2500


def process_GNSSTEC_files(path_to_raw_files):
    station_names = Path(path_to_raw_files).glob("*/")
    for station in station_names:
        p = station.rglob("*")
        files = [x for x in p if x.is_file()]
        current_df = pd.DataFrame()
        number = 0
        for file in files:
            current_df = pd.concat([current_df, pd.read_csv((file), compression="xz")])
            number = number + 1
        current_df.rename(columns={"tec": station.name}).set_index("dates").to_csv(
            str(station) + ".csv.xz", compression="xz"
        )


def get_station_names(path_to_stations, site):
    df_stations = pd.read_csv(path_to_stations + "/" + "station_list.csv")
    station_names = list((df_stations[df_stations["site"] == site])["station_id"])
    # Read all the station available:
    df_stations = df_stations.set_index("station_id")
    # Compute the mean lat and long of th eselected stations ToDo: Improve this aspect
    mean_location = df_stations.loc[station_names].describe().loc["mean"]
    return station_names, mean_location


def read_GNSSTEC_data(path_to_tec, station_names):
    stations_files = Path(path_to_tec).glob("*.csv.xz")
    list_stations = [
        station
        for station in stations_files
        if any(x in station.name for x in station_names)
    ]
    ds = pd.date_range(start=start_day, end=end_day, freq="30s")
    df = pd.DataFrame({"dates": ds}).set_index("dates")
    for station in list_stations:
        current_df = pd.read_csv(station, compression="xz")
        current_df["dates"] = pd.to_datetime(current_df["dates"])
        current_df = current_df.set_index("dates").sort_index()
        df = pd.merge(df, current_df, how="outer", left_index=True, right_index=True)
    df.index.names = ["ds"]
    return df


def process_input_data(df, freq, type=None):
    df = df.resample(rule=freq).mean()
    df.drop(df.index[-1], inplace=True)

    return df


def read_EQ_data(dir_path):
    p = Path(dir_path).glob("*.csv")
    all_pd = pd.DataFrame()

    valid_colums = ["time", "longitude", "latitude", "depth", "mag", "type"]
    for file in p:
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
    # Remove NaN interpolate
    all_pd = all_pd.interpolate(method="from_derivatives")
    return all_pd


""


def process_eq(df_events, station_names, mean_location, freq):
    # Create geodesic object to compute dist and azimuth
    geodesic = pyproj.Geod(ellps="WGS84")

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
    df_events["arc_cos"] = np.abs(
        np.cos(np.deg2rad(90 + df_eq_results["azimuth"].values))
    )
    df_events["arc_sin"] = np.abs(
        np.sin(np.deg2rad(90 + df_eq_results["azimuth"].values))
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


"""
class CustomMinMax(MinMaxScaler):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
    def transform(self, X, y=None):
        X["dist"] = 
        transformed_X = super().transform(X)
"""


def prepare_eq(
    df_events,
    dist_start,
    dist_delta,
    mag_start,
    mag_delta,
    filter,
    drop,
):
    m1 = mag_start
    m2 = m1 + mag_delta
    d2 = dist_start + dist_delta
    d1 = dist_start
    print("m1: " + str(m1) + " m2: " + str(m2) + " d1: " + str(d1) + " d2: " + str(d2))
    df_events["pr"] = get_pr(df_events, d1, d2, m1, m2)
    if filter == 1:
        df_events = df_events.loc[df_events["pr"] > 1]
    df_events_copy = df_events.copy()
    df_events_copy["depth"] = -df_events["depth"]
    df_events_copy["dist"] = -df_events["dist"]
    scaler = MinMaxScaler()
    df_scaled = drop_scale(df_events_copy, scaler, drop)
    return (df_scaled, scaler, df_events)

@dataclass
class ConfigEQ:
    dist_start: int
    dist_delta: int
    mag_start: float
    mag_delta: float
    filter: bool
    drop: list

def prepare_EQ(
    df_events,
    config_eq
):
    m1 = config_eq.mag_start
    m2 = m1 + config_eq.mag_delta
    d2 = config_eq.dist_start + config_eq.dist_delta
    d1 = config_eq.dist_start
    df_events["pr"] = get_pr(df_events, d1, d2, m1, m2)
    if filter == 1:
        df_events = df_events.loc[df_events["pr"] > 1]
    df_events_copy = df_events.copy()
    df_events_copy["depth"] = -df_events["depth"]
    df_events_copy["dist"] = -df_events["dist"]
    if config_eq.filter:
        df_events_copy = df_events_copy.drop(config_eq.drop, axis=1)
    return (df_events_copy)

def drop_scale(df, scaler, drop):
    if drop:
        df = df.drop(drop, axis=1)
    arr_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(arr_scaled, columns=df.columns, index=df.index)
    return df


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
    z_score = stats.zscore(df, nan_policy="omit")
    df[(np.abs(z_score) > max_z)] = np.NaN


def read_ion_data(path_ion):
    widths = [
        4,
        4,
        3,
        3,
        6,
        6,
        6,
        6,
        6,
        6,
        9,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        5,
        7,
        3,
        4,
        6,
        4,
        6,
        5,
        6,
        6,
        6,
        9,
        9,
        9,
        9,
        3,
    ]
    ion_data = pd.read_fwf(path_ion, widths=widths, header=None)
    id_ion = [
        "year",
        "doy",
        "hour",
        "id imf",
        "B scalar",
        "B vector",
        "Lat B",
        "Long B",
        "BY",
        "Bz",
        "SW Plasma Ta",
        "SW Proton",
        "SW Plasma Speed",
        "SW Plasma flow long",
        "SW Plasma Speed lat",
        "Alpha ratio",
        "Flow pressure",
        "Alfen",
        "Magneto",
        "Quasy",
        "Kp",
        "N sunspot",
        "Dst-index",
        "Ap index",
        "f107",
        "AE",
        "AL",
        "AU",
        "pc",
        "lyman",
        "Proton10",
        "Proton30",
        "Proton60",
        "Flux",
    ]
    ion_data.columns = id_ion
    strfmt = "{year}-{doy:0=3d}T{hour:0=2d}:00:00"

    ion_data["datetime"] = ion_data.apply(
        lambda x: datetime.strptime(
            strfmt.format(year=int(x["year"]), doy=int(x["doy"]), hour=int(x["hour"])),
            "%Y-%jT%H:%M:%S",
        ),
        axis=1,
    )

    df_ion = ion_data.drop(["year", "doy", "hour"], axis=1).set_index("datetime")
    return df_ion


def read_irradiance_data(path_data, site):
    df_irradiance = pd.read_csv(
        path_data + stations_path + irradiance_path,
        parse_dates=["time"],
        date_parser=dateutil.parser.parse,
    ).set_index("time")
    df_irradiance = pd.DataFrame(df_irradiance[site])
    df_irradiance.index = df_irradiance.index.tz_convert(None)

    return df_irradiance


def prepare_ion_data(path_data, site, freq, type="complete"):
    station_names, mean_location = get_station_names(path_data + stations_path, site)
    df_GNSSTEC = read_GNSSTEC_data(path_data + GNSSTEC_path, station_names)
    # Not very useful since it has to be related to the hour of the sun
    if type == "hourly":
        if pd.Timedelta(freq) < pd.Timedelta(days=1):
            raise KeyError("Freq has to be greater than 1d")
        df_GNSSTEC["hour"] = df_GNSSTEC.index.hour
        pd_df_ion_hour = pd.DataFrame()
        query_hour = "hour == {hour}"
        for hour in df_GNSSTEC.index.hour.unique():
            current_df_ion_hour = (
                df_GNSSTEC.query(query_hour.format(hour=hour))
                .drop("hour", axis=1)
                .resample(rule="1d")
                .mean()
            )
            current_df_ion_hour.columns = current_df_ion_hour.columns + "_" + str(hour)
            pd_df_ion_hour = pd.concat([pd_df_ion_hour, current_df_ion_hour], axis=1)
        df_GNSSTEC = pd_df_ion_hour

    df_GNSSTEC = process_input_data(df_GNSSTEC, freq)
    remove_outliers(df_GNSSTEC, 4)
    remove_outliers(df_GNSSTEC, 4)
    df_GNSSTEC = df_GNSSTEC.interpolate(metrod="from_derivatives")

    df_ion = read_ion_data(path_data + ion_path)
    df_ion = df_ion[["Kp", "f107", "N sunspot"]]
    df_ion = df_ion.resample(rule=freq).ffill()
    df_ion.index.names = ["ds"]

    df_irradiance = read_irradiance_data(path_data, site)
    df_irradiance.index.names = ["ds"]
    df_eq = read_EQ_data(path_data + eq_path)
    df_eq = process_eq(
        df_eq, station_names=station_names, mean_location=mean_location, freq=freq
    )
    df_covariate = pd.merge(
        df_irradiance, df_ion, left_index=True, right_index=True, how="outer"
    )
    df_covariate = df_covariate.interpolate(metrod="from_derivatives")

    return df_GNSSTEC, df_covariate, df_eq


# read_iono_data(["noa1"])

"""
import pvlib
import pandas as pd

data, inputs, meta = pvlib.iotools.get_pvgis_hourly(
    latitude=39.5, # North is positive
    longitude=23, # East is positive
    start=pd.Timestamp('2016-01-01'), # First available year is 2005
    end=pd.Timestamp('2020-12-31'), # Last available year is 2020 (depends on database)
    raddatabase='PVGIS-SARAH2',
    surface_tilt=0, # surface tilt angle
    surface_azimuth=0, # 0 degrees corresponds to south
    components=True, # Whether you want the individual components or just the total
    url='https://re.jrc.ec.europa.eu/api/v5_2/', # URL for version 5.2
    )

data[['poa_direct','poa_sky_diffuse','poa_ground_diffuse']].plot(
    figsize=(6,4), subplots=True, sharex=True)
"""
