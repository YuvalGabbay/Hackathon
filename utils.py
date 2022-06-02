#################################################################
# FILE : utils.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577
# DESCRIPTION: utils for IML classes - implement split_train_test for models
#################################################################

from typing import Tuple
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def load_labels(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=filename, names=["labels"])
    return df


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filepath_or_buffer=filename)
    df.columns = ['Form_Name',
                  'Hospital',
                  'User_Name',
                  'Age',
                  'Basic_stage',
                  'Diagnosis_date',
                  'Her2',
                  'Histological_diagnosis',
                  'Histopatological_degree',
                  'Ivi -Lymphovascular_invasion',
                  'KI67_protein',
                  'Lymphatic_penetration',
                  'M-metastases_mark_(TNM)',
                  'Margin_Type',
                  'N-lymph_nodes_mark_(TNM)',
                  'Nodes_exam',
                  'Positive_nodes',
                  'Side',
                  'Stage',
                  'Surgery_date1',
                  'Surgery_date2',
                  'Surgery_date3',
                  'Surgery_name1',
                  'Surgery_name2',
                  'Surgery_name3',
                  'Surgery_sum',
                  'T-Tumor_mark_(TNM)',
                  'Tumor_depth',
                  'Tumor_width',
                  'er',
                  'pr',
                  'surgery_before_or_after-Activity_date',
                  'surgery_before_or_after-Actual_activity',
                  'id']
    # @TODO  - לשנות את התאריך לכמה זמן מאובחנת מהיום
    fields_to_drop = ["Hospital"]
    df = df.drop(columns=fields_to_drop)


    return df


def preprocess1(df: pd.DataFrame):
    histological_diagnosis = ["INFILTRATING DUCT CARCINOMA", "LOBULAR INFILTRATING CARCINOMA", "INTRADUCTAL CARCINOMA", ]
    print(df["Histological_diagnosis"].unique())
    df.drop_duplicates()


def preprocess2(df: pd.DataFrame):
    # preprocess the KI67_protein field
    field_name = 'KI67_protein'
    unique = df[field_name].value_counts()
    df[field_name] = df[field_name].str.rstrip('%')
    df[field_name] = df[field_name].str.replace(pat='(/w+)', repl='-1', regex=True)
    df[field_name] = df[field_name].fillna(value=-1)
    df[field_name] = df[field_name].str.findall('(\d+)')
    df[field_name] = df[field_name].apply(lambda x: np.float(np.array(x, dtype=float).mean()))
    df[field_name] = df[field_name].astype(float)
    mean_value = df[field_name].mean()
    df[field_name].fillna(value=mean_value, inplace=True)

    a = 1
    # from datetime import datetime
    # m = df['test'].apply(lambda v: isinstance(v, datetime))



def preprocess3(df: pd.DataFrame):
    #drop nan from Surgery_sum,Tumor_depth,Tumor_width
    df["Surgery_sum"] = df["Surgery_sum"].fillna(0)
    df["Tumor_depth"] = df["Tumor_depth"].fillna(0)
    df["Tumor_width"] = df["Tumor_width"].fillna(0)

    #change margin type to binary values
    df['Margin_Type'] = df['Margin_Type'].replace(['נקיים'], 0)
    df['Margin_Type'] = df['Margin_Type'].replace(['ללא'], 0)
    df['Margin_Type'] = df['Margin_Type'].replace(['נגועים'], 1)
    #print(df["Margin_Type"].unique())

    #Drop duplicate columns in which the user name and the day are the same
    datetimes = pd.to_datetime(df['Diagnosis_date'])


    df['date'] = datetimes.dt.date
    df = df.drop_duplicates(subset=['id', 'date'], keep='first')
    print(df.shape)

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # Shuffle your dataset
    shuffle_df = X.sample(frac=1)
    y = y.reindex_like(shuffle_df)
    train_size = int(len(shuffle_df) * train_proportion)
    train_X = shuffle_df[:train_size]
    train_y = y[:train_size]
    test_X = shuffle_df[train_size:]
    test_Y = y[train_size:]

    # train_X, test_X, train_y, test_Y  = train_test_split(X, y, test_size=train_proportion, random_state=0)
    return train_X, train_y, test_X, test_Y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
