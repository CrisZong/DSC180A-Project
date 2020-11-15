import urllib
import pandas as pd
import numpy as np
def fetch_and_save(link,save_name):
    urllib.request.urlretrieve(link,save_name)
def transform_data(path):
    col_names = ["Date","cases_specimen","percent","admits","cases_reported"]
    astypes = {"cases_specimen": "float", 'case_reported': "float"}
    covid_series = pd.read_excel(path, names=col_names, astype=astypes)
    covid_series = covid_series[5:].dropna().drop(["percent","admits"],axis=1).reset_index(drop=True) # remove the shitty data
    covid_series["Date"] = pd.to_datetime(covid_series["Date"] + '-2020')
    return covid_series
