import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.layers import Flatten, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from utils import load_data, get_daily_data, get_forecast_dates

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_test_split(data, test_size=100, time_step=7):
    train_data = data["cnt"].iloc[:-test_size]
    test_data = data["cnt"].iloc[-test_size:]

    X_train, y_train = create_sequences(train_data.to_numpy().reshape(-1,1), time_step)
    X_test, y_test = create_sequences(test_data.to_numpy().reshape(-1,1), time_step)
    return X_train, y_train, X_test, y_test


def train_lstm(X_train, y_train, model_save_name, train=True):
    model = keras.Sequential(
    [
        BatchNormalization(),
        LSTM(128, activation= 'relu', input_shape=(X_train.shape[1], X_train.shape[2]), 
                return_sequences = False
                ),
        Dropout(0.5),
        Dense(32, activation = 'relu', kernel_initializer = 'random_uniform'),
        Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation = 'relu')
        ]
    ) 

    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer)

    if train:
        history = model.fit(x=X_train, y=y_train,
                        epochs=10, batch_size=16, verbose=2,
                        callbacks = [EarlyStopping(monitor='loss', patience=3, verbose=2, mode='min')])

        model.save('covid_prediction/output/%s.h5' % model_save_name)
    else:
        model = keras.models.load_model("covid_prediction/output/%s.h5" % model_save_name)
        # model.summary()

    return model

def predict(model, X_test, forecast_days=7):
    preds = []
    test_seq = X_test[-1].reshape((1, 7, 1))

    for i in range(forecast_days + 1):
        y_test_pred = model(test_seq)
        pred = y_test_pred.numpy()[0][0]
        preds.append(pred)
        new_seq = test_seq.flatten()
        new_seq = np.append(new_seq, [pred])
        test_seq = new_seq[1:].reshape((1,7,1))

    _, forecast_dates = get_forecast_dates(np.arange(7), start = "11/30/2021", n_days=0)

    output = {"dates": forecast_dates, "lstm_pred": preds[1:]}
    output_df = pd.DataFrame.from_dict(output, orient="columns")
    return output_df


if __name__ == "__main__":
    confirmed, deaths, dates = load_data(
        "covid_prediction/dataset/covid19_confirmed_global.txt",
        "covid_prediction/datasetcovid19_deaths_global.txt"
    )

    confirmed_daily = get_daily_data(confirmed)
    death_daily = get_daily_data(deaths)

    # predict confirm daily
    X_train, y_train, X_test, y_test = train_test_split(confirmed_daily)
    confirm_model = train_lstm(X_train, y_train, "lstm_model_us_confirm", train=False)
    confirm_pred = predict(confirm_model, X_test, forecast_days=7)
    confirm_pred.to_csv("covid_prediction/output/lstm_confirm_pred.csv", index=False, header=True)

    # predict confirm daily
    X_train, y_train, X_test, y_test = train_test_split(death_daily)
    death_model = train_lstm(X_train, y_train, "lstm_model_us_deaths", train=False)
    death_pred = predict(death_model, X_test, forecast_days=7)
    death_pred.to_csv("covid_prediction/output/lstm_death_pred.csv", index=False, header=True)