import pandas as pd
import re as r
import math as m


# converter functions

def convert_rating(x):
    y = float(x)
    if m.isnan(y):
        return 0
    return y


def convert_type(x):
    ret_val = x
    if (x == '0') | (x == 'NaN'):
        ret_val = 'Free'
    return ret_val


def convert_reviews(x):
    ret_val = 0
    if r.match('[1-9].[0-9]M', x):
        ret_val = int(float(x[0:2]) * 1000000)
    else:
        ret_val = int(x)
    return ret_val


def convert_size(x):
    sz = -1
    if x == "Varies with device":
        return sz
    s = x[-1:]
    if s == "M":
        sz = int(float(x[:-1]) * 1000000)
    elif s == "k":
        sz = int(float(x[:-1]) * 1000)
    return sz


def convert_installs(s):
    if s == "Free":
        return 0
    r = [e for e in s if ((e != ",") & (e != "+"))]
    return int("".join(r))


def convert_price(s):
    p = 0.0
    if s[0:1] == "$":
        p = float(s[1:])
    return p


def load_data(path):
    # Load csv into dataframe. Do necessary conversions
    data_playstore = pd.read_csv(path, converters={
        'Rating': convert_rating, 'Reviews': convert_reviews,
        'Size': convert_size, 'Type': convert_type, 'Installs': convert_installs, 'Price': convert_price
    })
    # Data cleaning
    df = data_playstore.fillna(0).drop_duplicates()
    # Remove non english apps from the data frame
    df_eng_apps_only = df[df.App.str.contains(r'^[a-zA-Z0-9_]+')]
    return df_eng_apps_only
