# Libraries
import datetime as dt
import numpy as np
import pandas as pd
import warnings

# Disable depreciated warnings
pd.set_option("mode.chained_assignment",None)
warnings.simplefilter(action='ignore', category=FutureWarning)


def import_rates(filename, filepath):
    """
    This function imports data from selected path.
    Required parameters:
     - filepath: str
     - filename: str
    """
    return pd.read_csv(filepath+filename)


def clean_rates(df, start, end, maturities, freq):
    """
    This function filters the data for the selected time frame, relevant maturities, frequency
    Required parameters:
     - df: pd.DataFrame
     - start: date ("YY-MM-DD")
     - end: date ("YY-MM-DD")
     - maturities: list
     - frequency: category (day, week, month)
    """

    # Filter for time frame
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m-%d")
    df = df[(df["TIME_PERIOD"]>=start) & (df["TIME_PERIOD"]<=end)]


    # Exclude spot rate spreads
    df["DATA_TYPE_FM"] = df["DATA_TYPE_FM"].apply(lambda x: "SRS" if x[:3] == "SRS" else x)
    df = df[df["DATA_TYPE_FM"] != "SRS"]


    # Filter for spot rates
    df[["TYPE","MATURITY_STR"]] = df["DATA_TYPE_FM"].str.split(pat = "_", expand = True)
    df = df[df["TYPE"] == "SR"]
    df.reset_index(drop = True, inplace = True)


    # Extract maturity year & month
    df[["YEAR","MONTH"]] = df["MATURITY_STR"].str.split(pat="Y",expand = True)
    month, year = list(), list()

    for idx, val in df["MONTH"].iteritems():
        if val == None:
            month.append(df["YEAR"].loc[idx])
            year.append(0)

        else:
            month.append(val)
            year.append(df["YEAR"].loc[idx])


    # Clean maturity month
    df["MONTH"] = pd.Series(month)
    df["MONTH"].replace("","0M", inplace = True)
    df["MONTH"] = df["MONTH"].apply(lambda x: x[:-1])
    df["MONTH"] = df["MONTH"].apply(lambda x: 0 if x == "" else x)


    # Clean maturity year
    df["YEAR"]  = pd.Series(year)
    df["MONTH"] = pd.to_numeric(df["MONTH"])
    df["YEAR"]  = pd.to_numeric(df["YEAR"])
    df["MATURITY_NUM"] = df["YEAR"]+df["MONTH"]/12


    # Keep only relevant features
    df = df[["TYPE","TIME_PERIOD","MATURITY_STR","MATURITY_NUM","OBS_VALUE"]]


    # Filter df for spot rates with selected maturities
    df = df[df["MATURITY_NUM"].isin(maturities)]


    # Pivot df for date & maturity
    df = df.sort_values(by="TIME_PERIOD", ascending = True)
    df = df.pivot_table(columns = ["MATURITY_NUM"],
                        index   = ["TIME_PERIOD"],
                        values  = ["OBS_VALUE"], aggfunc=np.sum)

    df.columns = df.columns.get_level_values(1)
    df.index.name = None
    
    
    # Rename columns to string format maturities
    cols = [maturity_transform(i) for i in maturities]
    df.columns = cols


    # Filter df for selected frequency
    df["DATE"] = df.index

    if freq == "week":    
        df["FREQ"] = df["DATE"].dt.week
    elif freq == "month": 
        df["FREQ"] = df["DATE"].dt.month
    else:
        df["FREQ"] = df["DATE"].dt.day


    df["COUNT"]  = np.where(df["FREQ"].shift() != df["FREQ"],1,0).cumsum()
    df = df.groupby("COUNT").last()

    df.index = df["DATE"]
    df.index.name = None
    df = df.drop(["FREQ","DATE"], axis = 1)
    
    return df


def maturity_transform(mat_int):
    """
    This function transforms years as decimals into strings according to %Y%M format.
    Required parameter: float (decimal divisor of 12)
    """
    
    if mat_int*12%1 == 0:
        month = int(mat_int%1*12)
        year  = int(mat_int-month/12)

        if year == 0: 
            mat_str = str(month)+"M"
            
        elif month == 0:
            mat_str = str(year)+"Y"
            
        else:         
            mat_str = str(year)+"Y"+str(month)+"M"
            
        return mat_str
    
    else:
        print ("Maturity needs to be a divisor of 12!")
        return None