import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def prepara_data(start_date, end_date, save_path):
    confirmed_df = pd.read_csv("covid_prediction/dataset/covid19_confirmed_global.txt")
    confirmed_df["confirm_cases_oct_21"] = confirmed_df[end_date] - confirmed_df[start_date]
    data = confirmed_df[["Country/Region", "Lat", "Long", "confirm_cases_oct_21"]]
    data = data[~data["Lat"].isnull()]
    data.to_csv(save_path, index=False)


start_date, end_date = "9/30/21", "10/31/21"
save_path = "covid_prediction/output/world_oct_21_confirm_cases.csv"
prepara_data(start_date, end_date,save_path)
data = pd.read_csv(save_path)

fig = px.scatter_geo(data, lat="Lat", lon="Long", size = "confirm_cases_oct_21", width=1200, height=1200)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
fig.write_image("covid_prediction/output/heat_map.png")