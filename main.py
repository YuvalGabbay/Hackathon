import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    file_name0 = 'Mission 2 - Breast Cancer/train.labels.0.csv'
    labels0 = utils.load_labels(filename=file_name0)

    file_name1 = 'Mission 2 - Breast Cancer/train.labels.1.csv'
    labels1 = utils.load_labels(filename=file_name1)

    file_name = 'Mission 2 - Breast Cancer/train.feats.csv'

    data = utils.load_data(filename=file_name)
    data['labels0'] = labels0['labels']
    data['labels1'] = labels1['labels']
    df_after = utils.preprocess1(data)
    df_after = utils.preprocess2(data)
    df_after = utils.preprocess3(data)
    utils.preprocess4(data)
    a = 1
    vals=df_after['labels0'].value_counts()
    print(vals)
