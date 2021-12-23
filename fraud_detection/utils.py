import pandas as pd
import numpy as np
import datetime
import geopy.distance
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def get_weekend(day):
    if day.weekday() > 4:
        return 1
    else:
        return 0
    
def get_distance(lat, long, merch_lat, merch_long):
    p1 = (lat, long)
    p2 = (merch_lat, merch_long)
    dist = geopy.distance.distance(p1, p2).km
    return dist

def get_age(dob):
    curr = datetime.datetime.strptime("2021-12-12", "%Y-%m-%d")
    age = round(abs((curr - dob).days / 365))
    return age

def get_city_level(x):
    if x > 500000:
        return "city_pop_larger_than_500000"
    elif x > 100000 and x <= 500000:
        return "city_pop_100000_to_500000"
    elif x > 50000 and x <= 100000:
        return "city_pop_50000_to_100000"
    elif x > 10000 and x <= 50000:
        return "city_pop_10000_to_50000"
    elif x > 5000 and x <= 10000:
        return "city_pop_5000_to_10000"
    elif x > 1000 and x <= 5000:
        return "city_pop_1000_to_5000"
    else:
        return "city_pop_smaller_than_1000"


def convert_to_categorical(df, cate_col_list):
    for cate_col in cate_col_list:   
        df[cate_col] = df[cate_col].astype('category')
    return df


def feature_engineering(data, feature_cols, label_col):
    data["cc_num_prefix"] = data["cc_num"].apply(lambda x: str(x)[:1])
    data["trans_month"] = data["trans_date_trans_time"].apply(lambda x:  x.month)
    data["trans_hour"] = data["trans_date_trans_time"].apply(lambda x: x.hour)
    data["trans_weekend"] = data["trans_date_trans_time"].apply(lambda x: get_weekend(x))
    data["age"] = data["dob"].apply(lambda x: get_age(x))
    data["city_pop_level"] = data["city_pop"].apply(lambda x: get_city_level(x))

    clean_data = data[feature_cols + label_col]
    clean_data = convert_to_categorical(clean_data, feature_cols[-7:])

    return clean_data


def evaluate(model, X, y):
    y_pred = model.predict(X, num_iteration=model.best_iteration_)
    
    print('Precision: %.4f' % precision_score(y, y_pred))
    print('Recall: %.4f' % recall_score(y, y_pred))
    print('F1：', f1_score(y, y_pred))
    print('AUC：', roc_auc_score(y, y_pred))
    
    return y_pred