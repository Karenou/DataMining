import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

def load_data(confirm_data_path, death_data_path):
    confirmed_df = pd.read_csv(confirm_data_path)
    deaths_df = pd.read_csv(death_data_path)

    # only filter US data
    cols = confirmed_df.keys()
    confirmed = confirmed_df[confirmed_df["Country/Region"] == "US"].loc[:, cols[4]:cols[-1]]
    deaths = deaths_df[deaths_df["Country/Region"] == "US"].loc[:, cols[4]:cols[-1]]

    # get the dates
    dates = confirmed.keys()
    return confirmed, deaths, dates


def prepare_data(data, dates):
    X_train,  X_test, y_train, y_test = train_test_split(
        np.array([i for i in range(len(dates))]).reshape(-1, 1) , 
        data.T.values, 
        test_size=0.1, shuffle=False
    ) 
    return X_train,  X_test, y_train, y_test

def get_forecast_dates(dates, start = "1/22/2020", n_days=7):
    future_forcast = np.array([i for i in range(len(dates) + n_days)]).reshape(-1, 1)
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    return future_forcast, future_forcast_dates

def get_daily_pred(future_forcast_dates, pred_df, name, n_days=8):
    pred_df = pred_df.reshape(1,-1)[0]
    pred_df = pd.DataFrame({
        'date': future_forcast_dates[-n_days:], 
        'cumulative_pred': np.round(pred_df[-n_days:])
    })
    pred_df[name] = pred_df["cumulative_pred"].diff(1)
    return pred_df.iloc[1:][["date", name]].set_index("date")

def get_daily_data(data):
    daily_df = data.T.diff(periods=1)
    daily_df.columns = ["cnt"]
    daily_df.iloc[0, 0] = data.iloc[0, 0]
    daily_df = daily_df.reset_index(drop=False)
    daily_df.columns = ["date", "cnt"]
    return daily_df

