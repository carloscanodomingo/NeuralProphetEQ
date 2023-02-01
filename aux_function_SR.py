import h5py
import pandas as pd
import numpy as np


class SR_SENSORS:
    NS = 1
    EW = 2
    BOTH = 3


def get_eq_filtered(
    df_events,
    dist_start,
    dist_delta,
    dist_max,
    lat_max,
    arc_max,
    mag_start,
    mag_delta,
    dist_perct,
    num_classes,
    eq_window,
    SR_sensor,
):
    mag = df_events["mag"].copy()
    dist = df_events["dist"].copy()
    arc = df_events["arc"].copy()
    lat = df_events["lat"].copy()
    m1 = mag_start
    m2 = m1 + mag_delta
    arc_max_cos = np.abs(np.cos(np.deg2rad(arc_max)))
    discarted = np.logical_or(mag > 100, dist > dist_max, np.abs(lat) > lat_max)
    # 90 is due to changing the axis to the horizontal one

    arc_max_cos = np.abs(np.cos(np.deg2rad(arc_max)))
    selected_ns = np.abs(np.cos(np.deg2rad(90 + arc))) >= arc_max_cos

    arc_max_sin = np.abs(np.sin(np.deg2rad(90 - arc_max)))
    selected_ew = np.abs(np.sin(np.deg2rad(90 + arc))) >= arc_max_sin

    if SR_sensor == SR_SENSORS.NS:
        arc_selected = selected_ns
    elif SR_sensor == SR_SENSORS.EW:
        arc_selected = selected_ew
    elif SR_sensor == SR_SENSORS.BOTH:
        arc_selected = np.logical_or(selected_ens, selected_ew)

    discarted = np.logical_or(discarted, arc_selected == False)
    mag[discarted] = np.nan
    dist[discarted] = np.nan
    n_eq = sum(~np.isnan(mag))
    vector_filter = np.ones(eq_window)
    out = np.zeros(len(dist))
    out_raw = np.zeros(len(dist))

    # Creating Figure
    x = np.linspace(4.5, 8, 40)
    y_dict = dict(x=x)
    names = list()
    y_last = np.zeros(len(x))
    dist_dif = (dist_max * dist_perct) / (num_classes)
    for index_classes in range(num_classes, 0, -1):

        d2 = dist_start + (index_classes) * dist_delta
        d1 = dist_start + dist_dif * (index_classes - 1)
        (A, B) = get_coef(d1, d2, m1, m2)
        cat_eq = np.zeros(len(dist))
        result_cat_eq = np.divide(np.power(10, (mag * A) + B), dist) >= 1
        cat_eq[result_cat_eq] = 1.0
        cat_eq_conv = np.convolve(cat_eq, vector_filter, mode="same")
        out[cat_eq_conv > 0.0] = index_classes
        out_raw[cat_eq > 0.0] = index_classes

        y = np.power(10, (x * A) + B)

        name = str("y" + str(index_classes))
        names.append(name)
        y_dict[name] = y - y_last
        y_last = y

    return (
        out,
        out_raw,
        n_eq,
    )


def get_coef(d1, d2, m1, m2):
    A = np.log10(d2 / d1) / (m2 - m1)
    B = np.log10(d1) - A * m1
    return (A, B)


def read_data(filepath):
    file = h5py.File(filepath, "r")

    NS_mean = file.get("NS_mean")[()]
    NS_mean = NS_mean.transpose()

    NS_std = file.get("NS_std")[()]
    NS_std = NS_std.transpose()

    EW_mean = file.get("EW_mean")[()]
    EW_mean = EW_mean.transpose()

    EW_std = file.get("EW_std")[()]
    EW_std = EW_std.transpose()

    mag = file.get("mag")[()]
    mag = mag.transpose()

    dist = file.get("dist")[()]
    dist = dist.transpose()

    depth = file.get("depth")[()]
    depth = depth.transpose()

    arc = file.get("arc")[()]
    arc = arc.transpose()

    lat = file.get("lat")[()]
    lat = lat.transpose()

    data = {
        "lat": lat,
        "NS_mean": NS_mean,
        "NS_std": NS_std,
        "EW_mean": EW_mean,
        "EW_std": EW_std,
        "arc": arc,
        "mag": mag,
        "dist": dist,
        "depth": depth,
    }
    d = {
        "NS_mean": pd.Series(tuple(data["NS_mean"])),
        "NS_std": pd.Series(tuple(data["NS_std"])),
        "EW_mean": pd.Series(tuple(data["EW_mean"])),
        "EW_std": pd.Series(tuple(data["EW_std"])),
        "lat": pd.Series(np.array(data["lat"]).squeeze()),
        "arc": pd.Series(np.array(data["arc"]).squeeze()),
        "mag": pd.Series(np.array(data["mag"]).squeeze()),
        "dist": pd.Series(np.array(data["dist"]).squeeze()),
        "depth": pd.Series(np.array(data["depth"]).squeeze()),
    }
    df = pd.DataFrame(d)
    return df
