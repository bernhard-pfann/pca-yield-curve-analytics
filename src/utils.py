import os
import urllib.request 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import adfuller


def download(target_path:str, end_date: str, overwrite: bool, start_date: str ="2004-09-06"):
    """Download rates from ECB"""

    if os.path.isfile(target_path) and not overwrite:
        print("Download skipped")
        return
    else:
        print("Download started")
        url = "https://sdw-wsrest.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.?startPeriod="+start_date+"&endPeriod="+end_date+"&format=csvdata"
        urllib.request.urlretrieve(url, target_path)
        print("Download finished")


def create_folders():
    if not os.path.exists("assets"):
        os.makedirs("assets")
    if not os.path.exists("assets/stress"):
        os.makedirs("assets/stress")
    if not os.path.exists("assets/stress/rates"):
        os.makedirs("assets/stress/rates")
    if not os.path.exists("assets/stress/eig_scores"):
        os.makedirs("assets/stress/eig_scores")


def calc_rmse(a,b):
    se = (a-b)**2
    rmse = np.sqrt(se.mean(axis=1))
    return rmse

def std_scale_pandas(df):
    """Performs standard scaling and retains index & columns of dataframe."""

    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    scaler = StandardScaler()
    df = pd.DataFrame(
        data=scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    return df

def rainbow(categories):
    """Generates a dictionary of color codes for each category."""
    c_scale = plt.cm.rainbow(np.linspace(0,1,len(categories)))
    c_dict = {}

    for i,c in zip(categories,c_scale):
        c_dict[i] = c
        
    return c_dict

def adf_test(df, col, alpha):
    """
    Applies the Augemented Dickey Fuller test for a respective time series at a 
    certain confidence interval.
    
    Args:
        df (pd.DataFrame): Observations
        col (str): Name of column to apply the statistical test on
        alpha (float): Possible values are [0.01, 0.05, 0.1]

    Returns:
        Dictionary with test statistic, p-value, critical value 
    """
    model  = adfuller(df[col])
    alpha = str(int(alpha*100))+"%"
    result = dict()
    
    result["adf_stat"]  = round(model[0],4)
    result["p_val"]     = round(model[1],4)
    result["threshold"] = round(model[4][alpha], 4)
    
    if result["adf_stat"] < result["threshold"]:
        result["stationary"] = "yes"
    else:
        result["stationary"] = "no"
    
    return result