import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from Model import Estimator
from Model2 import Estimator2
import utils
from pathlib import Path


def bar_plot(y_true, y_pred):
    X = y_true.columns.values
    fig, ax = plt.subplots(figsize=(8, 6))
    X_axis = np.arange(len(X))
    rects1 = ax.bar(X_axis - 0.2, y_true.sum(axis=0), 0.4, label='True labels')
    rects2 = ax.bar(X_axis + 0.2, y_pred.sum(axis=0), 0.4, label='Pred labels')
    ax.set_xticks(X_axis, X)
    ax.set_xlabel("Metastases Sites")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Number of Patients with each metastases site group")
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        # train the model
        file_name = 'Mission 2 - Breast Cancer/train.feats.csv'
        data = utils.load_data(filename=file_name)
        file_name0 = 'Mission 2 - Breast Cancer/train.labels.0.csv'
        labels0 = utils.load_labels(filename=file_name0)
        file_name1 = 'Mission 2 - Breast Cancer/train.labels.1.csv'
        labels1 = utils.load_labels(filename=file_name1)

        data['labels0'] = labels0['labels']
        data['labels1'] = labels1['labels']
        df_after = utils.preprocess(data)
        new_labels = utils.preprocess_labels_part_1(df_after)
        labels_1 = df_after['labels1']
        relevant_features = ["Age", "KI67_protein", "Surgery_sum", "Tumor_depth", "Tumor_width", "Margin_Type"]
        df_after = df_after[relevant_features]
        weights = np.where((new_labels >= 1).any(axis=1), 0.9, 0.05)
        model_1 = Estimator(weights=weights)
        model_1.fit(df_after, new_labels)
        y_pred = model_1.predict(df_after)
        bar_plot(new_labels, y_pred)

        # Get labels for test
        file_name = 'Mission 2 - Breast Cancer/test.feats.csv'
        test_data = utils.load_data(filename=file_name)
        test_df = utils.preprocess(test_data)
        test_df = test_df[relevant_features]
        test_y_pred = model_1.predict(test_df)
        part_1_path = 'part1/predictions.csv'
        filepath = Path(part_1_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = pd.DataFrame(test_y_pred)
        df_to_save.to_csv(filepath)
    except ValueError:
        raise ValueError("Oh No - something went wrong in part 1")




    # est = Estimator2
    # est.fit(X=df_after, y=data['labels1'])
    # print(est.loss(df_after, data['labels1']))
