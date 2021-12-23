from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


def polynomial(future_forcast, X_train, X_test, y_train, y_test):
    # transform our data for polynomial regression
    poly = PolynomialFeatures(degree=2)
    poly_X_train_confirmed = poly.fit_transform(X_train)
    poly_X_test_confirmed = poly.fit_transform(X_test)
    poly_future_forcast = poly.fit_transform(future_forcast)

    # polynomial regression
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)
    # print('MAE: %.0f' % mean_absolute_error(test_linear_pred, y_test))
    # print('MSE: %.0f' % mean_squared_error(test_linear_pred, y_test))

    return linear_pred


def bayesian(future_forcast, X_train, X_test, y_train, y_test):
    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 'normalize' : normalize}
    
    bayesian_poly = PolynomialFeatures(degree=2)
    bayesian_poly_X_train = bayesian_poly.fit_transform(X_train)
    bayesian_poly_X_test = bayesian_poly.fit_transform(X_test)
    bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(
        bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, 
        return_train_score=True, n_jobs=-1, n_iter=40, verbose=1
        )
    bayesian_search.fit(bayesian_poly_X_train, y_train)
    bayesian_confirmed = bayesian_search.best_estimator_
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    # print('MAE: %.0f' % mean_absolute_error(test_bayesian_pred, y_test))
    # print('MSE: %.0f' % mean_squared_error(test_bayesian_pred, y_test))

    return bayesian_pred