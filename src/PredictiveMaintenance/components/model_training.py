import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from PredictiveMaintenance.utils import common
from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.entity import (ModelTrainingConfig, ModelTrainingParams)


class ModelTraining:
    def __init__(self, 
                 config:ModelTrainingConfig,
                 params:ModelTrainingParams) -> None:
        self.config = config
        self.params = params

        common.create_directories([self.config.root_dir,
                                   self.config.models_dir,
                                   self.config.figures_dir])
    
    def evaluate(self, y_true, y_pred, label="train"):
        """
            This method is used for evaluation of the model.

            Parameters
            ----------
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.
            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
            label: str
                'train', 'test' or 'valid'


            Returns
            -------
            None

            Raises
            ------
            Exception
        
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            variance = r2_score(y_true, y_pred)
            logger.info(f"""{str(label)} set RMSE: {str(rmse)}, R2: {str(variance)}""")

            return rmse, variance
        
        except Exception as e:
            logger.exception(f"""Exception during computing evaluation metrics.
                             Exception message: {str(e)}""")
            raise e
    
    def load_data(self, ):
        """
            This method is used to load the pre-processed data for training.

            Parameters
            ----------
            None

            Returns
            -------
            X : pandas DataFrame like
                Ground truth (correct) input features
            y : pandas DataFrame like
                Ground truth (correct) target values

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            logger.info(f"""Loading pre-processed data for model training started...""")

            X = pd.read_csv(self.config.preprocessed_X)
            y = pd.read_csv(self.config.preprocessed_y)

            logger.info(f"""Loading pre-processed data for model training successful.""")
            return X, y

        except Exception as e:
            logger.exception(f"""Exception in data loading.
                             Exception message: {str(e)}""")
            raise e

    def plot_pred_actual(self, y_test, y_pred, model_name):
        """
            This method is used to plot real data and the predicted one to make some comparison.

            Parameters
            ----------
            y_test : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.
            y_test_hat : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
            model_name : str
                The name of the model that is used to predict y_test_hat

            Returns
            -------
            None

            Raises
            ------
            Exception

            Examples
            ------
            plot_pred_actual(y_test, y_test_hat, "random_forest");

            Notes
            ------
            The figure will be saved at 'artifacts/model_training/figures' directory 
            for further reference and evaluation.
        
        """
        try:
            logger.info(f"""Plotting Actual data Vs Predicted data started...""")
        
            indices = np.arange(len(y_test))
            wth= 0.6
            plt.figure(figsize=(70,30))
            true_values = [int(x) for x in y_test.values]
            predicted_values = y_pred.flatten()

            plt.bar(indices, true_values, width=wth, color='b', label='True RUL')
            plt.bar([i for i in indices], predicted_values, width=0.5*wth, color='r', alpha=0.7, label='Predicted RUL')

            plt.legend(prop={'size': 40})
            plt.tick_params(labelsize=40)
            plt.ylabel("Values")
            plt.title(f"""Comparison of y_test and y_pred for {str(model_name)}""")

            # plt.show()
            plt.savefig(os.path.join(self.config.figures_dir, str(model_name)+"_results.png"))

            logger.info(f"""Plotted Actual data Vs Predicted data for model '{str(model_name)}'.""")
        
        except Exception as e:
            logger.exception(f"""Exception while plotting Actual data Vs Predicted data.""")
            raise e
    
    def train_test_split(self, X, y):
        """
            This method is used split the training data into train set and dev set.

            Parameters
            ----------
            X : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) feature values.
            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            Returns
            -------
            X_train : pandas DataFrame type
                Ground truth (correct) training feature values.
            X_val : pandas DataFrame type
                Ground truth (correct) validation feature values.
            y_train : pandas DataFrame type
                Ground truth (correct) training target values.
            y_test :
                Ground truth (correct) validation target values.

            Raises
            ------
            Exception
        
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.params.test_size, random_state=42
            )
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.exception(f""""Exception at Train test split.
                             Exception message: {str(e)}""")
            raise e
    
    def get_best_params_for_linear_regression(self, X_train, y_train):
        """
            This method is used to get best parameters for Linear Regression algorithm 
            by performing hyper parameter tuning.

            Parameters
            ----------
            X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) feature values.
            y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            Returns
            -------
            best_linear_regression_model: 
                Returns best linear regression model by performing hyper parameter tuning.

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f"""Hyper parameter tuning for linear regression model started...""")

            params_grid = self.params.linear_regression_params

            grid = RandomizedSearchCV(estimator=LinearRegression(),
                                    param_distributions=params_grid,
                                    cv=3,
                                    scoring="neg_mean_squared_error",
                                    n_iter=len(params_grid))
            grid.fit(X_train, y_train)

            # extracting the best parameters
            fit_intercept = grid.best_params_["fit_intercept"]
            copy_X = grid.best_params_["copy_X"]

            # best linear regression model
            best_linear_regression_model = LinearRegression(
                fit_intercept=fit_intercept,
                copy_X=copy_X,
            )
            best_linear_regression_model.fit(X_train, y_train)

            logger.info(f"""Linear Regression best parameters found: {str(grid.best_params_)}""")

            return best_linear_regression_model
    
        except Exception as e:
            logger.exception(f"""" Exception occured in 'get_best_params_for_linear_regression' 
                             method of 'ModelTraining' class.Exception message: {str(e)}""")
            raise e
    
    def linear_regression(self, X_train, X_val, y_train, y_val):
        """
            This method performs hyper parameter tuning for linear regression, evaluate and 
            return rmse, r2_score respectively

            Parameters
            ----------
            X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) train feature values.
            X_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) validation feature values.
            y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) train target values.
            y_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) validation target values.

            Returns
            -------
            rmse_train : float
                train rmse score
            r2_score_train : float
                train r2_score
            rmse_val : float
                validation rmse score
            r2_score_val : float
                validation r2_score

            Raises
            ------
            Exception

            Notes
            ------
            This method plots real data and the predicted one to make some comparison. 
            Refer 'artifacts/model_training/figures/linear_regression_results.png' directory.
        
        """
        best_linear_regression_model = self.get_best_params_for_linear_regression(X_train, y_train)
        
        y_pred_valid = best_linear_regression_model.predict(X_val)
        rmse_val, r2_score_val = self.evaluate(y_val, y_pred_valid, "valid")

        y_pred_train = best_linear_regression_model.predict(X_train)
        rmse_train, r2_score_train = self.evaluate(y_train, y_pred_train, "train")

        logger.info(f"""Linear Regression: train set RMSE: {str(rmse_train)}, R2: {str(r2_score_train)}""")
        logger.info(f"""Linear Regression: valid set RMSE: {str(rmse_val)}, R2: {str(r2_score_val)}""")

        self.plot_pred_actual(y_val, y_pred_valid, "linear_regression")

        return rmse_train, r2_score_train, rmse_val, r2_score_val, best_linear_regression_model
    
    def get_best_params_for_svr(self, X_train, y_train):
        """
            This method is used to get best parameters for Support Vector Regression algorithm 
            by performing hyper parameter tuning.

            Parameters
            ----------
            X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) feature values.
            y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            Returns
            -------
            best_svr_model: 
                Returns best Support Vector Regression model by performing hyper parameter tuning.

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f"""Hyper parameter tuning for Support Vector Regression model started...""")

            params_grid = self.params.svr_params

            grid = RandomizedSearchCV(estimator=SVR(),
                                    param_distributions=params_grid,
                                    cv=3,
                                    scoring="neg_mean_squared_error",
                                    n_iter=len(params_grid))
            grid.fit(X_train, y_train.values.ravel())

            # extracting the best parameters
            kernel = grid.best_params_["kernel"]
            C = grid.best_params_["C"]

            # best linear regression model
            best_svr_model = SVR(
                kernel=kernel,
                C=C,
            )
            best_svr_model.fit(X_train, y_train.values.ravel())

            logger.info(f"""Support Vector Regression best parameters found: {str(grid.best_params_)}""")

            return best_svr_model
    
        except Exception as e:
            logger.exception(f"""" Exception occured in 'get_best_params_for_svr' 
                             method of 'ModelTraining' class.Exception message: {str(e)}""")
            raise e
    
    def svr(self, X_train, X_val, y_train, y_val):
        """
            This method performs hyper parameter tuning for support vector regression, evaluate and 
            return rmse, r2_score respectively

            Parameters
            ----------
            X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) train feature values.
            X_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) validation feature values.
            y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) train target values.
            y_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) validation target values.

            Returns
            -------
            rmse_train : float
                train rmse score
            r2_score_train : float
                train r2_score
            rmse_val : float
                validation rmse score
            r2_score_val : float
                validation r2_score

            Raises
            ------
            Exception

            Notes
            ------
            This method plots real data and the predicted one to make some comparison. 
            Refer 'artifacts/model_training/figures/linear_regression_results.png' directory.
        
        """
        best_svr_model = self.get_best_params_for_svr(X_train, y_train)
        
        y_pred_valid = best_svr_model.predict(X_val)
        rmse_val, r2_score_val = self.evaluate(y_val, y_pred_valid, "valid")

        y_pred_train = best_svr_model.predict(X_train)
        rmse_train, r2_score_train = self.evaluate(y_train, y_pred_train, "train")

        logger.info(f"""SVR: train set RMSE: {str(rmse_train)}, R2: {str(r2_score_train)}""")
        logger.info(f"""SVR: valid set RMSE: {str(rmse_val)}, R2: {str(r2_score_val)}""")

        self.plot_pred_actual(y_val, y_pred_valid, "svr")

        return rmse_train, r2_score_train, rmse_val, r2_score_val, best_svr_model
    
    def get_best_params_for_random_forest(self, X_train, y_train):
        """
            This method is used to get best parameters for Random Forest Regression algorithm 
            by performing hyper parameter tuning.

            Parameters
            ----------
            X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) feature values.
            y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            Returns
            -------
            best_svr_model: 
                Returns best Random Forest Regression model by performing hyper parameter tuning.

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f"""Hyper parameter tuning for Random Forest Regression model started...""")

            params_grid = self.params.random_forest_params

            grid = RandomizedSearchCV(estimator=RandomForestRegressor(),
                                    param_distributions=params_grid,
                                    cv=3,
                                    scoring="neg_mean_squared_error",
                                    n_iter=len(params_grid))
            grid.fit(X_train, y_train.values.ravel())

            # extracting the best parameters
            n_estimators = grid.best_params_["n_estimators"]
            max_features = grid.best_params_["max_features"]
            max_depth = grid.best_params_["max_depth"]
            criterion = grid.best_params_["criterion"]

            # best linear regression model
            best_rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                criterion=criterion,
            )
            best_rf_model.fit(X_train, y_train.values.ravel())

            logger.info(f"""Random Forest Regression best parameters found: {str(grid.best_params_)}""")

            return best_rf_model
    
        except Exception as e:
            logger.exception(f"""" Exception occured in 'get_best_params_for_random_forest' 
                             method of 'ModelTraining' class.Exception message: {str(e)}""")
            raise e
    
    def random_forest(self, X_train, X_val, y_train, y_val):
        """
            This method performs hyper parameter tuning for random forest regression, evaluate and 
            return rmse, r2_score respectively

            Parameters
            ----------
            X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) train feature values.
            X_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) validation feature values.
            y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) train target values.
            y_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) validation target values.

            Returns
            -------
            rmse_train : float
                train rmse score
            r2_score_train : float
                train r2_score
            rmse_val : float
                validation rmse score
            r2_score_val : float
                validation r2_score

            Raises
            ------
            Exception

            Notes
            ------
            This method plots real data and the predicted one to make some comparison. 
            Refer 'artifacts/model_training/figures/linear_regression_results.png' directory.
        
        """
        best_rf_model = self.get_best_params_for_random_forest(X_train, y_train)
        
        y_pred_valid = best_rf_model.predict(X_val)
        rmse_val, r2_score_val = self.evaluate(y_val, y_pred_valid, "valid")

        y_pred_train = best_rf_model.predict(X_train)
        rmse_train, r2_score_train = self.evaluate(y_train, y_pred_train, "train")

        logger.info(f"""Random Forest: train set RMSE: {str(rmse_train)}, R2: {str(r2_score_train)}""")
        logger.info(f"""Random Forest: valid set RMSE: {str(rmse_val)}, R2: {str(r2_score_val)}""")

        self.plot_pred_actual(y_val, y_pred_valid, "random_forest")

        return rmse_train, r2_score_train, rmse_val, r2_score_val, best_rf_model
    
    def train(self, X_train, X_val, y_train, y_val):

        (rmse_train_lr, r2_score_train_lr, rmse_val_lr, 
         r2_score_val_lr, best_linear_regression_model) = self.linear_regression(X_train, X_val, y_train, y_val)
        
        (rmse_train_svr, r2_score_train_rf, rmse_val_rf, 
         r2_score_val_svr, best_svr_model) = self.svr(X_train, X_val, y_train, y_val)
        
        (rmse_train_rf, r2_score_train_rf, rmse_val_rf, 
         r2_score_val_rf, best_rf_model) = self.random_forest(X_train, X_val, y_train, y_val)
        
        result_tuple = ((rmse_train_lr, r2_score_train_lr, rmse_val_lr, r2_score_val_lr, best_linear_regression_model),
                        (rmse_train_svr, r2_score_train_rf, rmse_val_rf, r2_score_val_svr, best_svr_model),
                        (rmse_train_rf, r2_score_train_rf, rmse_val_rf, r2_score_val_rf, best_rf_model))

        # Sort the tuple based on r2_score_val in descending order
        sorted_result_tuple = sorted(result_tuple, key=lambda x: x[3], reverse=True)

        best_result_tuple = self.return_best_model(sorted_result_tuple)
        self.save_best_model_and_corresponding_metrics(best_result_tuple)


    def return_best_model(self, sorted_result_tuple):

        # returns the tuple with best r2 validation score
        # that do not overfits, rmse_val <= rmse_train
        # if all the models overfits, then return the tuple with highest r2 validation score
        for a_tuple in sorted_result_tuple:
            if a_tuple[2] <= a_tuple[0]:
                return a_tuple
        return sorted_result_tuple[0]
    
    def save_best_model_and_corresponding_metrics(self, best_result_tuple):
        
        best_model = best_result_tuple[-1]
        # save best model 
        with open(os.path.join(self.config.models_dir, "best_model"+".pkl"), "wb") as file:
            pickle.dump(best_model, file)
        
        best_metric = f"""Train: RMSE: {str(best_result_tuple[0])}, r2_score: {str(best_result_tuple[1])}
Validation: RMSE: {str(best_result_tuple[2])}, r2_score: {str(best_result_tuple[3])}"""
        with open(os.path.join(self.config.models_dir, "best_model_metrics.txt"), "w") as file:
            file.write(best_metric)

        logger.info(f"""The best model and its corresponding metrics are saved.""")
        

