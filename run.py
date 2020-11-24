#!/usr/bin/env python
from src.data.etl import fetch_and_save,transform_data
from src.model.model import loss_p,loss_s,loss_ps
from src.analysis.analysis import showLoss
import pandas as pd
import sys
import json


def main(targets):

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        fetch_and_save(data_cfg["links"][0],data_cfg["save_name"][0])
        transform_data("data/covid_time_series.xls",'data/cleaned_series.csv')
    if 'correlations' in targets: 
        with open('config/model-params.json') as fh:
            corr_cfg = json.load(fh)
        covid_series = pd.read_csv('data/cleaned_series.csv')
        showLoss(covid_series["cases_specimen"],
                covid_series["cases_reported"],
                corr_cfg["correlations"][0],
                corr_cfg["correlations"][1],
                [loss_p,loss_s,loss_ps])
    if 'test' in targets: 
        with open('config/model-params.json') as fh:
            corr_cfg = json.load(fh)
        covid_series = pd.read_csv('test/testdata/test_series.csv')
        showLoss(covid_series["cases_specimen"],
                covid_series["cases_reported"],
                corr_cfg["correlations"][0],
                corr_cfg["correlations"][1],
                [loss_p,loss_s,loss_ps])
    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
