from urllib import request
import pandas as pd
import numpy as np
def fetch_and_save(link,save_name):
    request.urlretrieve(link,save_name)
def transform_data(path,save_name):
    col_names = ["Date","cases_specimen","percent","admits","cases_reported"]
    covid_series = pd.read_excel(path, names=col_names)
    covid_series = covid_series[5:].dropna().drop(["percent","admits"],axis=1).reset_index(drop=True) # remove the shitty data
    covid_series["Date"] = pd.to_datetime(covid_series["Date"] + '-2020')
    covid_series.to_csv(save_name,index=False)
