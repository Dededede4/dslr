import numpy as np
import pandas as pd
from Ft_array import *
import describe as descr
import histogram as histo
import scatter_plot as scatter
import pair_plot as pair
import argparse
from pandas.api.types import is_string_dtype


def main():
    parser = argparse.ArgumentParser(description='j\'suis un choixpeau magic')
    parser.add_argument('-s', '--scrypt', type=str, dest="scrypt", choices=['describe', 'histogram', 'scatter_plot', 'pair_plot'],
    help="function scrypt")
    parser.add_argument(type=str, dest="dataset",
                             help="describe dataset")

    opt = parser.parse_args()
    if (opt.scrypt == 'describe'):
        descr.describe(opt.dataset)
    elif (opt.scrypt == 'histogram'):
        histo.histogram(opt.dataset)
    elif (opt.scrypt == 'scatter_plot'):
        scatter.scatter(opt.dataset)
    elif (opt.scrypt == 'pair_plot'):
        pair.pair_plot(opt.dataset)

if __name__ == '__main__':
    main()
