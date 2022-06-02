import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import utils



if __name__ == '__main__':
    file_name = 'Mission 2 - Breast Cancer/train.feats.csv'
    df = utils.load_data(filename=file_name)
    a = 1
