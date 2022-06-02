import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from Model2 import Estimator2
import utils


def bar_plot(y_true, y_pred):
    X = y_true.unique()
    plt.bar(X, y_true, 0.4, label='true labels')
    plt.bar(X, y_pred, 0.4, label='pred labels')
    print(plt)


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
    df_after = utils.preprocess2(df_after)
    df_after = utils.preprocess3(df_after)
    new_labels = utils.preprocess4(df_after)
    vals = df_after['labels0'].value_counts()
    labels_1 = df_after['labels1']
    relevant_features = ["Age", "KI67_protein", "Surgery_sum", "Tumor_depth", "Tumor_width", "Margin_Type"]
    df_after = df_after[relevant_features]
    cols = new_labels.columns
    # weights = np.zeros(df_after.shape[0])
    # weights = new_labels.loc[(new_labels >= 1)]
    # a = 1

    est=Estimator2()
    est.fit(df_after, labels_1)
    predicted_val = est.predict(df_after)
    print(predicted_val)
    print("LOS",est.loss(df_after, labels_1))

