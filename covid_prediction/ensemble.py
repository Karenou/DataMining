import pandas as pd


def ensemble(linear_pred_path, lstm_pred_path, save_path):
    linear_pred = pd.read_csv(linear_pred_path)
    # linear_pred = linear_pred.set_index("date")

    lstm_pred = pd.read_csv(lstm_pred_path)
    # lstm_pred = lstm_pred.set_index("date")

    final_pred = linear_pred.join(lstm_pred)
    final_pred["ensembled_pred"] = round((final_pred["linear_pred"] + final_pred["bayesian_pred"] + final_pred["lstm_pred"]) / 3)
    final_pred[["date", "ensembled_pred"]].to_csv(save_path, index=False, header=True)

if __name__ == "__main__":
    ensemble("covid_prediction/output/linear_confirm_pred.csv",
             "covid_prediction/output/lstm_confirm_pred.csv",
             "covid_prediction/output/ensemble_confirm_pred.csv")

    ensemble("covid_prediction/output/linear_death_pred.csv",
            "covid_prediction/output/lstm_death_pred.csv",
            "covid_prediction/output/ensemble_death_pred.csv")