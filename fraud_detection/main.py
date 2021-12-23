from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import pandas as pd
from utils import feature_engineering, evaluate


# if need to evaluate on test data, please put the correct test file path and read the data
data = pd.read_csv('fraudTrain.csv', index_col = 0, parse_dates=['trans_date_trans_time', 'dob'])

# if evaluate on test, change it to False
is_train = True

# the last 7 columns are categorical features
feature_cols = ["trans_month", "trans_weekend", "trans_hour", "amt",
                "age", "lat", "long", "merch_lat", "merch_long", 
                "cc_num_prefix", "gender", "category", "job",  "state", "zip", "city_pop_level"]

label_col = ["is_fraud"]

data = feature_engineering(data, feature_cols, label_col)

X = data.drop(["is_fraud"], axis=1)
y = data["is_fraud"]

# train or evaluate
if is_train:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gbm = LGBMClassifier(learning_rate=0.05, n_estimators=100, max_depth=10, scale_pos_weight=120,
                     min_child_samples=300, subsample=0.6, colsample_bytree=0.6, reg_lambda=1e-3,
                     random_state=100, silence=True)

    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
            early_stopping_rounds=5, categorical_feature=feature_cols[-7:])

    joblib.dump(gbm, 'Q4_output/lightgbm_model.pkl')

    y_pred = evaluate(gbm, X_test, y_test)
else:
    gbm = joblib.load('Q4_output/lightgbm_model.pkl')
    y_pred = evaluate(gbm, X, y)


y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("Q4_output/Q4_predicted_results.csv", index=False, header=False)


