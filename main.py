import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import utils
from pandas import read_csv




if __name__ == '__main__':
    file_name = 'Mission 2 - Breast Cancer/train.feats.csv'
    file_name2 = 'Mission 2 - Breast Cancer/train.labels.0.csv'
    df = utils.load_data(filename=file_name)
    df_labels = pd.read_csv(filepath_or_buffer=file_name2)
    utils.preprocess_label(df, df_labels)


    a = 1
    utils.preprocess1(df)
