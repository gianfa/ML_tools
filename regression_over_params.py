import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score



# Regression
def fit_model_over_params(coeffs, m_name, m, X_train, X_test, y_train):
    ''' Fit a model over a parameters lists
    Arguments:
        coeffs (dict): coefficients or params to feed to the estimator.
            { PARAM: PARAM_VALUES }
        m_name (srt): model name.
        m : model object.
    Returns:
        fitted (dict): { PARAM: {PARAM_VALUE: PREDICTIONS} }
    Example:
        m_name = 'RF'
        m = models[m_name]
        fit_model_over_params(coeffs[m_name], m_name, m)
    '''
    fitted = coeffs.copy()
    for c, val_list in coeffs.items():  # 'coef_name' : coeff_list
        fitted[c] = {}
        for val in val_list:
            setattr(m, c, val)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            fitted[c][val]= y_pred
    return fitted

def fit_all_models_over_params(coeffs, models, X_train, X_test, y_train):
    '''Fit models over parameters lists
    Arguments:
        coeffs (dict): coefficients or params to feed to the estimator.
            { ESTIMATOR_NAME: { PARAM: PARAM_VALUES }}
    Returns:
        fitted (dict): { ESTIMATOR_NAME: { PARAM: {PARAM_VALUE: PREDICTIONS} }} 
    Example:
        models = {
            'Ada': AdaBoostRegressor(random_state=rs, n_estimators=100),
            'RF': RandomForestRegressor( n_jobs=-1, random_state=rs)
        }

        coeffs = {
            'Ada': {
                'n_estimators': [5, 20, 50],
            },
            'RF': {
                'n_estimators': [1, 2, 8, 16],
                'max_depth': [2, 4, 6] 
            }
        }

        fit_all_model_over_params(coeffs)
    '''
    fitted = {}
    for m_name, c_dict in coeffs.items():
        m = models[m_name]
        fitted[m_name] = fit_model_over_params(coeffs[m_name], m_name, m, X_train, X_test, y_train)
    return fitted

def plot_scores_vs_c(fitted, m_name, y_test):
    ''' Plot regression scores of a fitted model over parameters list
    Arguments:
        fitted (dict): { ESTIMATOR_NAME: { PARAM: {PARAM_VALUE: PREDICTIONS} }} 
    Returns:
        m_name (srt): model name.
    Example:
        m_name = 'RF'
        fitted = fit_all_model_over_params(coeffs, m_name)
        plot_scores_vs_c(fitted, m_name)
    '''
    scores = ['evar', 'maxer', 'r2']

    scores2name = {
        'evar' : 'Explained Variance',
        'maxer': 'Max Absolute Error',
        'r2' : 'R2'
    }
    c_count = 1
    sc_count= 1
    fig = plt.figure(figsize=(13,10))
    for c, c_list in fitted[m_name].items():

        for scorer in scores:
            n_coeffs = len(fitted[m_name])
            ax = plt.subplot(len(scores), n_coeffs, c_count)
            sc_count += 1; c_count += 1
            xt = []
            yt = []
            # compute score #
            score = None
            for par, y_pred in c_list.items():
                #
                if scorer == 'evar': score = explained_variance_score(y_test, y_pred)
                if scorer == 'maxer': score = max_error(y_test, y_pred)
                if scorer == 'r2': score = r2_score(y_test, y_pred)
                #
                xt.append(par)
                yt.append(score)
            ax.plot(xt, yt, marker='o', alpha=0.4)
        #         ax.set_title(models[mi][0])
            ax.set_title(m_name + ' ' + scores2name[scorer])
            ax.set_xlabel(c)
            ax.legend(c_list.keys());
            ax.axhline(0, color='gray', alpha=0.5)
    plt.subplots_adjust(top=1.7)
    return

def plot_all_scores_vs_c(fitted, y_test):
    '''
    Arguments:
        fitted (dict): { ESTIMATOR_NAME: { PARAM: {PARAM_VALUE: PREDICTIONS} }}
    Returns:
        m_name (srt): model name.
    Example:
        fitted = fit_model_over_params(m_name, m)
        plot_scores_vs_c(fitted, 'RF')
    '''
    for m_name, c_dict in fitted.items():
        plot_scores_vs_c(fitted, m_name, y_test)
    return
