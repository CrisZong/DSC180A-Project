#!/usr/bin/env python
from src.data.etl import fetch_and_save
import sys
import json


def main(targets):

    if 'data' in targets:
        with open('data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        fetch_and_save(data_cfg["links"][0],data_cfg["save_name"][0])
    if 'correlations' in targets: 
        
    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
