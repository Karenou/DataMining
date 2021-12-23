from model import polynomial, bayesian
from utils import load_data, prepare_data, get_forecast_dates, get_daily_pred


def predict(data, dates, save_path):
    future_forcast, future_forcast_dates = get_forecast_dates(dates, n_days=7)

    X_train,  X_test, y_train, y_test = prepare_data(data, dates)  
    poly_pred = polynomial(future_forcast, X_train, X_test, y_train, y_test)
    poly_pred = get_daily_pred(future_forcast_dates, poly_pred, "linear_pred")

    bayesian_pred = bayesian(future_forcast, X_train, X_test, y_train, y_test)
    bayesian_pred = get_daily_pred(future_forcast_dates, bayesian_pred, "bayesian_pred")

    # ensemble
    final_pred = poly_pred.join(bayesian_pred, on="date")
    final_pred.to_csv(save_path, index=True, header=True)


if __name__ == "__main__":
    confirmed, deaths, dates = load_data(
        "covid_prediction/dataset/covid19_confirmed_global.txt",
        "covid_prediction/datasetcovid19_deaths_global.txt"
    )

    predict(confirmed, dates, "covid_prediction/output/linear_confirm_pred.csv")
    predict(deaths, dates, "covid_prediction/output/linear_death_pred.csv")