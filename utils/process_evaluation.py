import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

#from utils.u19_visualize_utils import plot_points
from utils.plot_func import plot_points


def evaluate_regression_on_train_test(Z_train, Y_train, Z_test, Y_test, rng, save_path, epoch):
    Z_train = np.asarray(Z_train)
    Z_test = np.asarray(Z_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.array(Y_test)
    scaler = StandardScaler()
    scaler.fit(Z_train)
    Z_train_transformed = scaler.transform(Z_train)
    Z_test_transfored = scaler.transform(Z_test)
    if np.count_nonzero(np.isnan(Z_train_transformed)) > 0 or np.count_nonzero(np.isnan(Z_test_transfored)) > 0:
        return {"NAN": {"r2": np.inf, "mae": np.inf, "rmse": np.inf, "mape": np.inf}}, None
    else:

        eva_dict = {}
        Y_test_transformed = Y_test*(rng[1]-rng[0]) + rng[0]
        # for linear_model, model_name in zip([LinearRegression(), SVR(kernel='rbf'), SVR(kernel="poly"), Lasso(), ElasticNet(0.1, l1_ratio=0.7), MLPRegressor(hidden_layer_sizes=(32))],
        #                                     ["LinearRegression", "SVR_rbf", "SVR_poly", "Lasso", "ElasticNet", "MLP"]):
        # for linear_model, model_name in zip([LinearRegression()], ["LinearRegression"]):
        
        model_name = "LinearRegression"
        linear_model = LinearRegression()

        linear_model.fit(Z_train_transformed, Y_train)
        Y_pred = linear_model.predict(Z_test_transfored)
        Y_pred_transformed = Y_pred*(rng[1]-rng[0]) + rng[0]
        Y_pred_transformed = np.clip(Y_pred_transformed, rng[0], rng[1])
        
        interval = 1
        plot_points(Y_test_transformed, Y_pred_transformed, 'test', 
                    save_path=os.path.join(save_path, f"test_{epoch}_{model_name}.png"),
                    interval=interval)

        r2 = r2_score(Y_test_transformed, Y_pred_transformed)
        mae = mean_absolute_error(Y_test_transformed, Y_pred_transformed)
        rmse = mean_squared_error(Y_test_transformed, Y_pred_transformed, squared=False)
        mape = mean_absolute_percentage_error(Y_test_transformed, Y_pred_transformed)
        eva_dict.update({model_name: {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}})

        return eva_dict, linear_model
