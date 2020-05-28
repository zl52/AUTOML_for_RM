import pandas as pd;
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.multiclass import type_of_target
import statsmodels.api as sm
import xgboost as xgb
import xgboost.sklearn
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from tools import *
from model_evaluation import model_summary


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   12. MODEL TRAINING   ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def cv_check(x, y, model_name='xgb', xgb_params=XGB_PARAMS, x_test=None, y_test=None, cv=5,
             random_state=2019, silent=True, **kwargs):
    """
    Cross validation check.

    : params x: x
    : params y: y
    : params model_name: model type
    : params xgb_params: xgb parameters
    : params x_test: x_test
    : params y_test: y_test
    : params cv: number of folds
    : params random_state: seed
    : params silent: If True, restrict prints of cv process

    : return df: dataframe recording cv process
    : return summary: statistical summary of df
    """
    skf = StratifiedKFold(n_splits=cv, random_state=random_state)
    df = pd.DataFrame()
    i = 0
    for train_index, val_index in skf.split(x, y):
        x_train, x_val = x.iloc[train_index,], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        i += 1
        if not silent:
            print('>>>>>>>>> {i} out of {j} folds'.format(i=i, j=cv))
        if x_test is not None and y_test is not None:
            model, pred_train_value, pred_val_value, pred_test_value = \
                model_toolkit(model_name, x_train, y_train, x_val=x_val, y_val=y_val, x_test=None,
                              xgb_params=xgb_params, make_prediction=True, **kwargs)

            res = model_summary(pred_train_value, y_train, pred_val_value, y_val,
                                pred_test_value=pred_test_value, real_test_label=y_test,
                                ax=None, pos_label=1, use_formater=False, plot=False)
        else:
            model, pred_train_value, pred_val_value = \
                model_toolkit(model_name, x_train, y_train, x_val=x_val, y_val=y_val, x_test=x_test,
                              xgb_params=xgb_params, make_prediction=True, **kwargs)
            res = model_summary(pred_train_value, y_train, pred_val_value, y_val,
                                pred_test_value=None, real_test_label=None,
                                ax=None, pos_label=1, use_formater=False, plot=False)
        df = pd.concat([df, res], axis=0)
    summary = df.reset_index(drop=False).groupby('index').agg([np.mean, np.std])
    return df, summary


def model_toolkit(model_name, x_train, y_train, x_val=None, y_val=None, x_test=None,
                  xgb_params=XGB_PARAMS, make_prediction=False, **kwargs):
    """
    Toolkit integrating training and predicting process for all model types.

    : params model_name: model type
    : params x_train: independent variable of train set
    : params y_train: dependent variable of train set
    : params x_val: independent variables of validation set
    : params y_val: dependent variable of validation set
    : params x_test: independent variables of test set
    : params y_test: dependent variable of test set
    : params xgb_params: xgb parameters
    : params make_prediction: If True, make predictions for each set

    : return model: (predictions for train, validation and test sets)
    """
    # try:
    if model_name == 'xgb':
        if (x_val is not None) & (y_val is not None):
            if x_test is None:
                model, pred_train_value, pred_val_value = xgbt(x_train
                                                             , y_train
                                                             , x_val
                                                             , y_val
                                                             , None
                                                             , params=xgb_params
                                                             , make_prediction=True
                                                             , **kwargs)
            else:
                model, pred_train_value, pred_val_value, pred_test_value = xgbt(x_train
                                                                              , y_train
                                                                              , x_val
                                                                              , y_val
                                                                              , x_test
                                                                              , params=xgb_params
                                                                              , make_prediction=True
                                                                              , **kwargs)
        else:
            model, pred_train_value = xgbt(x_train
                                         , y_train
                                         , x_val
                                         , y_val
                                         , x_test
                                         , params=xgb_params
                                         , make_prediction=True
                                         , **kwargs)
    else:
        if model_name == 'rf':
            model = rf(x_train, y_train, **kwargs)
        elif model_name == 'ada':
            model = ada(x_train, y_train, **kwargs)
        elif model_name == 'gb':
            model = gb(x_train, y_train, **kwargs)
        elif model_name == 'et':
            model = et(x_train, y_train, **kwargs)
        elif model_name == 'ovr':
            model = ovr(x_train, y_train, **kwargs)
        elif model_name == 'gnb':
            model = gnb(x_train, y_train)
        elif model_name == 'lr':
            model = lr(x_train, y_train, **kwargs)
            if (x_val is not None) & (y_val is not None):
                x_val['intercept'] = 1
            if x_test is not None:
                x_test['intercept'] = 1
        elif model_name == 'lsvm':
            model = lsvm(x_train, y_train, **kwargs)
        elif model_name == 'knn':
            model = knn(x_train, y_train, **kwargs)

        try:
            pred_train_value = model.predict_proba(x_train)[:, 1]
            if (x_val is not None) & (y_val is not None):
                pred_val_value = model.predict_proba(x_val)[:, 1]
            if x_test is not None:
                pred_test_value = model.predict_proba(x_test)[:, 1]
        except:
            pred_train_value = model.predict(x_train)
            if (x_val is not None) & (y_val is not None):
                pred_val_value = model.predict(x_val)
            if x_test is not None:
                pred_test_value = model.predict(x_test)

    if make_prediction == True:
        if (x_val is not None) & (y_val is not None) & (x_test is not None):
            print('Average of predictive values for train set:', np.round(np.mean(pred_train_value), 4))
            print('Average of predictive values for validation set:', np.round(np.mean(pred_val_value), 4))
            print('Average of predictive values for test set:', np.round(np.mean(pred_test_value), 4))
            return model, pred_train_value, pred_val_value, pred_test_value
        elif (x_val is not None) & (y_val is not None) & (x_test is None):
            print('Average of predictive values for train set:', np.round(np.mean(pred_train_value), 4))
            print('Average of predictive values for validation set:', np.round(np.mean(pred_val_value), 4))
            return model, pred_train_value, pred_val_value
        elif ((x_val is None) | (y_val is not None)) & (x_test is not None):
            print('Average of predictive values for train set:', np.round(np.mean(pred_train_value), 4))
            print('Average of predictive values for test set:', np.round(np.mean(pred_test_value), 4))
            return model, pred_train_value, pred_test_value
        else:
            print('Average of predictive values for train set:', np.round(np.mean(pred_train_value), 4))
            return model, pred_train_value
    else:
        return model


def xgbt(x_train, y_train, x_val=None, y_val=None, x_test=None, params=XGB_PARAMS, num_boost_round=10000,
         early_stopping_rounds=50, make_prediction=False, **kwargs):
    """
    Train xgb model and make predictions.

    : params model_name: model type
    : params x_train: independent variable of train set
    : params y_train: dependent variable of train set
    : params x_val: independent variables of validation set
    : params y_val: dependent variable of validation set
    : params x_test: independent variables of test set
    : params y_test: dependent variable of test set
    : params params: xgb parameters
    : params num_boost_round: num_boost_round for xgb model
    : params early_stopping_rounds: early_stopping_rounds for xgb model
    : params make_prediction: If True, make predictions for each set

    : return model: (predictions for train, validation and test sets)
    """
    if ud_type_of_target(y_train)=='binary':
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
    elif ud_type_of_target(y_train)=='continuous':
        params['objective'] = 'reg:linear'
        params['eval_metric'] = 'rmse'
    # elif ud_type_of_target(y_train)=='multiclass':
    #     params['objective'] = 'multi:softmax'
    #     params['eval_metric'] = 'mlogloss'
    #     params['num_class'] = len(np.unique(y_train))
    else:
        raise ValueError('Type of target is not supported')
    if (x_val is not None) & (y_val is not None):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        model_ = xgb.train(params
                          , dtrain
                          , num_boost_round=num_boost_round
                          , evals=watchlist
                          , verbose_eval=False
                          , early_stopping_rounds=early_stopping_rounds
                          , **kwargs)
        model = xgb.train(params
                          , dtrain
                          , num_boost_round=model_.best_iteration + 1
                          , evals=watchlist
                          , verbose_eval=True
                          , **kwargs)        
        if make_prediction == True:
            pred_train_value = model.predict(dtrain)
            pred_val_value = model.predict(dval)
            if x_test is not None:
                dtest = xgb.DMatrix(x_test)
                pred_test_value = model.predict(dtest)

                return model, pred_train_value, pred_val_value, pred_test_value
            else:
                return model, pred_train_value, pred_val_value
        else:
            return model
    else:
        dtrain = xgb.DMatrix(x_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, **kwargs)
        if make_prediction == True:
            pred_train_value = model.predict(dtrain)
            return model, pred_train_value
        else:
            return model


def rf(x_train, y_train, **kwargs):
    """
    Train random forest model.
    """
    params = {'n_estimators':200,
              'max_depth':None,
              'max_leaf_nodes':5,
              'min_samples_leaf':40,
              'min_samples_split':2,
              'min_weight_fraction_leaf':0.0,
              'random_state':2019,
              'max_features':"auto",
              'n_jobs':-1,
              'verbose':0,
              'bootstrap':True,
              'oob_score':True,
              'warm_start':False}
    if kwargs != {}:
        params.update(kwargs)
    if ud_type_of_target(y_train)=='binary':
        params.update({'class_weight':'balanced', 'criterion':'gini'})
        randomforest = RandomForestClassifier(**params)
    elif ud_type_of_target(y_train)=='continuous':
        params.update({'criterion':'mse'})
        randomforest = RandomForestRegressor(**params)
    else:
        raise ValueError('Type of target is not supported') 
    model = randomforest.fit(x_train, y_train)
    return model


def ada(x_train, y_train, **kwargs):
    """
    Train adaboost forest model.
    """
    params = {'n_estimators':400,
              'learning_rate':0.01,
              'base_estimator':None,
              'random_state':2019}
    if kwargs != {}:
        params.update(kwargs)
    if ud_type_of_target(y_train)=='binary':
        params.update({'algorithm':'SAMME.R'})
        adaboost = AdaBoostClassifier(**params)
    elif ud_type_of_target(y_train)=='continuous':
        adaboost = AdaBoostRegressor(**params)
    else:
        raise ValueError('Type of target is not supported') 
    model = adaboost.fit(x_train, y_train)
    return model


def gb(x_train, y_train, **kwargs):
    """
    Train gradient boosting model.
    """
    params = {'learning_rate':0.01, 
              'n_estimators':400, 
              'max_depth':5, 
              'subsample':0.8, 
              'min_samples_split':2, 
              'min_samples_leaf':1, 
              'min_impurity_split':1e-7, 
              'verbose':0, 
              'random_state':2019, 
              'max_features':None, 
              'max_leaf_nodes':None}
    if kwargs != {}:
        params.update(kwargs)
    if ud_type_of_target(y_train)=='binary':
        params.update({'loss':'deviance'})
        gbdt = GradientBoostingClassifier(**params)
    elif ud_type_of_target(y_train)=='continuous':
        params.update({'loss':'ls'})
        gbdt = GradientBoostingRegressor(**params)
    else:
        raise ValueError('Type of target is not supported') 
    model = gbdt.fit(x_train, y_train)
    return model


def et(x_train, y_train, **kwargs):
    """
    Train extra tree model.
    """
    params = {'n_estimators':400, 
              'max_depth':8, 
              'max_features':"auto", 
              'n_jobs':-1, 
              'random_state':2019, 
              'verbose':0}
    if kwargs != {}:
        params.update(kwargs)
    if ud_type_of_target(y_train)=='binary':
        params.update({'class_weight':'balanced', 'criterion':'gini'})
        extratree = ExtraTreesClassifier(**params)
    elif ud_type_of_target(y_train)=='continuous':
        params.update({'criterion':'mse'})
        extratree = ExtraTreesRegressor(**params)
    else:
        raise ValueError('Type of target is not supported') 
    model = extratree.fit(x_train, y_train)
    return model


def ovr(x_train, y_train, **kwargs):
    """
    Train oneVSrest model.
    """
    params = {'n_estimators':400, 
              'max_depth':8, 
              'max_features':"auto", 
              'n_jobs':-1, 
              'random_state':2019}
    if kwargs != {}:
        params.update(kwargs)
    if type_of_target(y_train)!='binary':
        raise ValueError('Type of target is not supported')
    else:
        est = RandomForestClassifier(**params)
        ovr = OneVsRestClassifier(est, n_jobs=-1)
    model = ovr.fit(x_train, y_train)
    return model


def gnb(x_train, y_train):
    """
    Train gaussian naive bayesian model.
    """
    if type_of_target(y_train)!='binary':
        raise ValueError('Type of target is not supported')
    else:
        gnb = GaussianNB()

    model = gnb.fit(x_train, y_train)
    return model


def lr(x_train, y_train, **kwargs):
    """
    Train logistics regression / linear regression model.

    params **kwargs['print']: If True, output model summary
    params **kwargs['equation_file']: equation_file's filename
    params **kwargs['model_summary_file']: model_summary_file's filename
    """
    params = {'C':0.1,
              'verbose':0,
              'dual':False,
              'max_iter':-1,
              'random_state':2019}
    x_train['intercept'] = 1
    if ud_type_of_target(y_train)=='binary':
        logisticregression = sm.Logit(y_train, x_train)
        model = logisticregression.fit(disp=False)
        model_summary = model.summary2(alpha=0.05)
        print(model_summary)
        if kwargs != {}:
            if kwargs['print'] == True:
                output_equation_txt = "df.loc[:,'intercept'] = 1\n"
                output_equation_txt += "df['logit'] = "

                for idx in model.params.index:
                    output_equation_txt += "\\\n" + "(".rjust(15) + str(model.params[idx]) + ") * df[" \
                                        + "'"+idx+"'" + "] + "
                output_equation_txt += "0\n\ndf['score'] = [1/(1+exp(-logit)) for logit in df['logit']]"
                write_txt(output_equation_txt, kwargs['equation_file'], encoding="utf-8")

                model_summary_output = pd.DataFrame()
                model_summary_output['Estimate'] = model.params
                model_summary_output['SE'] = model.bse
                model_summary_output['T_value'] = model.tvalues
                model_summary_output['P_value'] = model.pvalues
                try:
                    model_summary_output = model_summary_output.join(kwargs['vif'])
                except:
                    pass
                model_summary_output.to_excel(kwargs['model_summary_file'])

    elif ud_type_of_target(y_train)=='continuous':
        logisticregression = sm.OLS(y_train, x_train)
        model = logisticregression.fit(disp=False)
        model_summary = model.summary2(alpha=0.05)
        print(model_summary)
    else:
        raise ValueError('Type of target is not supported') 
    return model


def lsvm(x_train, y_train, **kwargs):
    """
    Train linear SVM model.
    """
    params = {'C':0.1,
              'verbose':0,
              'dual':False,
              'max_iter':-1,
              'random_state':2019}
    if kwargs != {}:
        params.update(kwargs)
    if ud_type_of_target(y_train)=='binary':
        params.update({'penalty':'l2', 'loss':'squared_hinge'})
        lsvm = LinearSVC(**params)
    elif ud_type_of_target(y_train)=='continuous':
        params.update({'loss':'epsilon_insensitive'})
        lsvm = LinearSVR(**params)
    else:
        raise ValueError('Type of target is not supported') 
    model = lsvm.fit(x_train, y_train)
    return model


def knn(x_train, y_train, **kwargs):
    """
    Train k-nearest neighbors model
    """
    x_train = np.log10(x_train + 1)
    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0

    params = {'n_neighbors':20, 
              'weights':'uniform', 
              'algorithm':'auto', 
              'n_jobs':-1}
    if kwargs != {}:
        params.update(kwargs)
    if ud_type_of_target(y_train)=='continuous':
        kneighbors = KNeighborsRegressor(**params)
    else:
        kneighbors = KNeighborsClassifier(**params)
    model = kneighbors.fit(x_train, y_train)
    return model


class BAYESIAN_OPTIMIZATION():
    """
    Apply bayesian optimization to tune model's parameters.
    """
    def __init__(self,
                 x,
                 y,
                 init_points=20,
                 acq='ei',
                 n_iter=40,
                 clf_metric='auc',
                 reg_metric='rmse'):
        """
        : params x: independent variables
        : params y: dependent variable
        : params init_points: number of parameter combinations when optimazation initializes
        : params n_iter: number of iterations
        : params acq: aquisition function
        : params clf_metric: metrics for classifiers
        : params reg_metric: metrics for regressors
        """
        self.x = x
        self.y = y
        self.tot = ud_type_of_target(y)
        self.init_points = init_points
        self.n_iter = n_iter
        self.acq = acq
        self.clf_metric = clf_metric
        self.reg_metric = reg_metric
        def get_eval_metric():
            if self.tot=='binary':
                return self.clf_metric 
            elif 'continuous':
                return self.reg_metric
            else:
                raise ValueError('Type of target is not supported')
        self.eval_metric = get_eval_metric()

    def xgb_bo(self):
        self.xgb_bo_optimizer = BayesianOptimization(
            self.xgb_cvscore_4bo,
            {'min_child_weight': (1, 19),
             'eta': (0.001, 0.5),
             'max_depth': (1, 15),
             'subsample': (0, 0.9),
             'colsample_bytree': (0, 0.9),
             'gamma': (0, 1),
             'lambdA': (0, 100),
             'alpha': (0, 100)}
        )
        self.xgb_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.xgb_bo_optimizer.max
        XGB_PARAMS = \
            {'eval_metric': self.eval_metric,
             'min_child_weight': int(res['params']['min_child_weight']),
             'eta': res['params']['eta'],
             'max_depth': int(res['params']['max_depth']),
             'subsample': res['params']['subsample'],
             'colsample_bytree': res['params']['colsample_bytree'],
             'gamma': res['params']['gamma'],
             'lambdA': res['params']['lambdA'],
             'alpha': res['params']['alpha']
             }
        if self.tot=='binary':
            XGB_PARAMS.update({'objective': 'binary:logistic'})
        elif self.tot=='continuous':
            XGB_PARAMS.update({'objective': 'reg:linear'})
        # elif self.tot=='multiclass':
            # XGB_PARAMS.update({'objective': 'multi:softmax', 'num_class': len(np.unique(self.y))})
        print('Best combination of parameters are shownn as below\n', XGB_PARAMS)
        return XGB_PARAMS

    def xgb_cvscore_4bo(self, eta, min_child_weight, max_depth, subsample, colsample_bytree, gamma, lambdA, alpha, cv=5):
        XGB_PARAMS = \
            {'eval_metric': self.eval_metric,
             'min_child_weight': int(min_child_weight),
               'eta': eta,
               'max_depth': int(max_depth),
               'subsample': subsample,
               'colsample_bytree': colsample_bytree,
               'gamma': gamma,
               'lambdA': lambdA,
               'alpha': alpha
             }
        if self.tot=='binary':
            XGB_PARAMS.update({'objective': 'binary:logistic'})
        elif self.tot=='continuous':
            XGB_PARAMS.update({'objective': 'reg:linear'})
        # elif self.tot=='multiclass':
            # XGB_PARAMS.update({'objective': 'multi:softmax', 'num_class': len(np.unique(self.y))})

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='xgb', xgb_params=XGB_PARAMS, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True)
        return summary.loc['val', self.eval_metric]['mean']

    def rf_bo(self):
        self.rf_bo_optimizer = BayesianOptimization(
            self.rf_cvscore_4bo,
            {'n_estimators': (10, 1000),
             'max_depth': (1, 40),
             'min_samples_split': (2, 40),
             'min_samples_leaf': (0.01, 0.5),
             'max_features': (0.1, 0.999)}
        )
        self.rf_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.rf_bo_optimizer.max
        RF_PARAMS = \
            {'n_jobs': -1,
             'bootstrap':True,
             'oob_score':True,
             'warm_start':False,
             'verbose': 0,
             'n_estimators': int(res['params']['n_estimators']),
             'max_depth': int(res['params']['max_depth']),
             'min_samples_split': int(res['params']['min_samples_split']),
             'min_samples_leaf': res['params']['min_samples_leaf'],
             'max_features': res['params']['max_features']
             }
        if self.tot=='binary':
            RF_PARAMS.update({'criterion':'gini', 'class_weight':'balanced'})
        elif self.tot=='continuous':
            RF_PARAMS.update({'criterion':'mse'})
        print('Best combination of parameters are shownn as below\n', RF_PARAMS)
        return RF_PARAMS

    def rf_cvscore_4bo(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                       max_features, cv=5):
        RF_PARAMS = \
            {'n_jobs': -1,
             'bootstrap':True,
             'oob_score':True,
             'warm_start':False,
             'verbose': 0,
             'n_estimators': int(n_estimators),
             'max_depth': int(max_depth),
             'min_samples_split': int(min_samples_split),
             'min_samples_leaf': min_samples_leaf,
             'max_features': max_features
             }
        if self.tot=='binary':
            RF_PARAMS.update({'criterion':'gini', 'class_weight':'balanced'})
        elif self.tot=='continuous':
            RF_PARAMS.update({'criterion':'mse'})

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='rf', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **RF_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']

    def ada_bo(self):
        self.ada_bo_optimizer = BayesianOptimization(
            self.ada_cvscore_4bo,
            {'n_estimators': (10, 1000),
             'learning_rate': (0.001, 1)}
        )
        self.ada_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.ada_bo_optimizer.max
        ADA_PARAMS = \
            {'base_estimator': None,
             'n_estimators': int(res['params']['n_estimators']),
             'learning_rate': (res['params']['learning_rate']),
             }
        if self.tot=='binary':
            ADA_PARAMS.update({'algorithm':'SAMME.R'})

        print('Best combination of parameters are shownn as below\n', ADA_PARAMS)
        return ADA_PARAMS

    def ada_cvscore_4bo(self, n_estimators, learning_rate, cv=5):
        ADA_PARAMS = \
            {'base_estimator': None,
             'n_estimators': int(n_estimators),
             'learning_rate': learning_rate
             }
        if self.tot=='binary':
            ADA_PARAMS.update({'algorithm':'SAMME.R'})

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='ada', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **ADA_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']

    def gb_bo(self):
        self.gb_bo_optimizer = BayesianOptimization(
            self.gb_cvscore_4bo,
            {'n_estimators': (10, 1000),
             'learning_rate': (0.001, 1),
             'max_depth': (1, 40),
             'min_samples_split': (2, 40),
             'min_samples_leaf': (0.01, 0.5),
             'min_impurity_split': (1e-6, 1e-3),
             'subsample': (0.1, 0.999)}
        )
        self.gb_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.gb_bo_optimizer.max
        GB_PARAMS = \
            {'verbose': 0,
             'max_features': None,
             'max_leaf_nodes': None,
             'n_estimators': int(res['params']['n_estimators']),
             'max_depth': int(res['params']['max_depth']),
             'learning_rate': res['params']['learning_rate'],
             'min_samples_split': int(res['params']['min_samples_split']),
             'min_samples_leaf': res['params']['min_samples_leaf'],
             'min_impurity_split': res['params']['min_impurity_split'],
             'subsample': res['params']['subsample']
             }
        if self.tot=='binary':
            GB_PARAMS.update({'loss':'deviance'})
        elif self.tot=='continuous':
            GB_PARAMS.update({'loss':'ls'})

        print('Best combination of parameters are shownn as below\n', GB_PARAMS)
        return GB_PARAMS

    def gb_cvscore_4bo(self, n_estimators, max_depth, learning_rate, min_samples_split, min_samples_leaf,
                       min_impurity_split, subsample, cv=5):
        GB_PARAMS = \
            {'verbose': 0,
             'max_features': None,
             'max_leaf_nodes': None,
             'n_estimators': int(n_estimators),
             'max_depth': int(max_depth),
             'learning_rate': learning_rate,
             'min_samples_split': int(min_samples_split),
             'min_samples_leaf': min_samples_leaf,
             'min_impurity_split': min_impurity_split,
             'subsample': subsample,
             }
        if self.tot=='binary':
            GB_PARAMS.update({'loss':'deviance'})
        elif self.tot=='continuous':
            GB_PARAMS.update({'loss':'ls'})

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='gb', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **GB_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']

    def et_bo(self):
        self.et_bo_optimizer = BayesianOptimization(
            self.rf_cvscore_4bo,
            {'n_estimators': (10, 1000),
             'max_depth': (1, 40),
             'min_samples_split': (2, 40),
             'min_samples_leaf': (0.01, 0.5),
             'max_features': (0.1, 0.999)}
        )
        self.et_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.et_bo_optimizer.max
        ET_PARAMS = \
            {'max_features': "auto",
             'n_jobs': -1,
             'verbose': 0,
             'criterion': 'gini',
             'n_estimators': int(res['params']['n_estimators']),
             'max_depth': int(res['params']['max_depth']),
             'min_samples_split': int(res['params']['min_samples_split']),
             'min_samples_leaf': res['params']['min_samples_leaf'],
             'max_features': res['params']['max_features']
             }
        if self.tot=='binary':
            ET_PARAMS.update({'class_weight':'balanced', 'criterion':'gini'})
        elif self.tot=='continuous':
            ET_PARAMS.update({'criterion':'mse'})

        print('Best combination of parameters are shownn as below\n', ET_PARAMS)
        return ET_PARAMS

    def et_cvscore_4bo(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                       max_features, cv=5):
        ET_PARAMS = \
            {'max_features': "auto",
             'n_jobs': -1,
             'verbose': 0,
             'criterion': 'gini',
             'n_estimators': int(n_estimators),
             'max_depth': int(max_depth),
             'min_samples_split': int(min_samples_split),
             'min_samples_leaf': min_samples_leaf,
             'max_features': max_features
             }
        if self.tot=='binary':
            ET_PARAMS.update({'class_weight':'balanced', 'criterion':'gini'})
        elif self.tot=='continuous':
            ET_PARAMS.update({'criterion':'mse'})

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='et', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **ET_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']

    def ovr_bo(self):
        self.ovr_bo_optimizer = BayesianOptimization(
            self.ovr_cvscore_4bo,
            {'n_estimators': (10, 1000),
             'max_depth': (1, 40),
             'min_samples_split': (2, 40),
             'min_samples_leaf': (0.01, 0.5),
             'max_features': (0.1, 0.999)}
        )
        self.ovr_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.ovr_bo_optimizer.max
        OVR_PARAMS = \
            {'n_jobs': -1,
             'verbose': 0,
             'criterion':'gini',
             'class_weight':'balanced',
             'n_estimators': int(res['params']['n_estimators']),
             'max_depth': int(res['params']['max_depth']),
             'min_samples_split': int(res['params']['min_samples_split']),
             'min_samples_leaf': res['params']['min_samples_leaf'],
             'max_features': res['params']['max_features']
             }
        print('Best combination of parameters are shownn as below\n', OVR_PARAMS)
        return OVR_PARAMS

    def ovr_cvscore_4bo(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                        max_features, cv=5):
        OVR_PARAMS = \
            {'n_jobs': -1,
             'verbose': 0,
             'criterion':'gini',
             'class_weight':'balanced',
             'n_estimators': int(n_estimators),
             'max_depth': int(max_depth),
             'min_samples_split': int(min_samples_split),
             'min_samples_leaf': min_samples_leaf,
             'max_features': max_features
             }
        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='ovr', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **OVR_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']

    def lsvm_bo(self):
        self.lsvm_bo_optimizer = BayesianOptimization(
            self.lsvm_cvscore_4bo,
            {'C': (0.001, 100)}
        )
        self.lsvm_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.lsvm_bo_optimizer.max
        LSVM_PARAMS = {'C': res['params']['C'],
                       'verbose': 0,
                       'dual':False,
                       'max_iter': -1}
        if self.tot=='binary':
            LSVM_PARAMS.update({'penalty':'l2', 'loss':'squared_hinge'})
        elif self.tot=='continuous':
            LSVM_PARAMS.update({'loss':'epsilon_insensitive'})

        print('Best combination of parameters are shownn as below\n', LSVM_PARAMS)
        return LSVM_PARAMS

    def lsvm_cvscore_4bo(self, C, cv=5):
        LSVM_PARAMS = {'C': C,
                       'verbose': 0,
                       'dual':False,
                       'max_iter': -1}
        if self.tot=='binary':
            LSVM_PARAMS.update({'penalty':'l2', 'loss':'squared_hinge'})
        elif self.tot=='continuous':
            LSVM_PARAMS.update({'loss':'epsilon_insensitive'})

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='lsvm', xgb_params=LSVM_PARAMS, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **LSVM_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']

    def knn_bo(self):
        self.knn_bo_optimizer = BayesianOptimization(
            self.knn_cvscore_4bo,
            {'n_neighbors': (3, 40),
             'leaf_size': (5, 40)}
        )
        self.knn_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.knn_bo_optimizer.max
        KNN_PARAMS = \
            {'n_jobs': -1,
             'weights':'uniform', 
             'algorithm':'auto', 
             'n_neighbors': int(res['params']['n_neighbors']),
             'leaf_size': int(res['params']['leaf_size'])
             }

        print('Best combination of parameters are shownn as below\n', KNN_PARAMS)
        return KNN_PARAMS

    def knn_cvscore_4bo(self, n_neighbors, leaf_size, cv=5):
        KNN_PARAMS = \
            {'n_neighbors': int(n_neighbors),
             'leaf_size': int(leaf_size),
             'n_jobs': -1,
             'weights':'uniform', 
             'algorithm':'auto', 
             }

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, model_name='knn', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **KNN_PARAMS)
        return summary.loc['val', self.eval_metric]['mean']


def model_selection(x, y, init_points=1, n_iter=1, acq='ei', cv=5, clf_metric='auc', reg_metric='rmse', random_state=2019,
                    model_list=['xgb', 'rf', 'ada', 'gb', 'et', 'ovr', 'knn', 'lr']):
    """
    Select best model.

    : params x: independent variables
    : params y: dependent variable
    : params init_points: number of parameter combinations when optimazation initializes
    : params n_iter: number of iterations
    : params acq: aquisition function
    : params cv: number of folds
    : params clf_metric: metrics for classifiers
    : params reg_metric: metrics for regressors
    : params random_state: seed
    : params model_list: list of models for training

    : return best_model: best model type
    : return best_model_params: parameter combination of best model type
    : return score_dict: dictionary recording each model type's best score
    : return params_dict: dictionary recording each model type's best parameter combination
    """
    if ud_type_of_target(y)=='binary':
        eval_metric = clf_metric
    elif ud_type_of_target(y)=='continuous':
        eval_metric = reg_metric
    else:
        raise ValueError('Type of target is not supported')

    score_dict, params_dict = {}, {}
    bo = BAYESIAN_OPTIMIZATION(x, y, init_points=init_points, n_iter=n_iter, acq=acq, clf_metric=clf_metric, reg_metric=reg_metric)
    if 'xgb' in model_list:
        try:
            print("Apply bayesian optimization to train XGBoost model\n")
            XGB_PARAMS = bo.xgb_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='xgb', xgb_params=XGB_PARAMS, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True)
                score_dict.update({'xgb': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'xgb': XGB_PARAMS})
        except Exception as e:
            print(e)


    if 'rf' in model_list:
        try:
            print("Apply bayesian optimization to train RandomForest model\n")
            RF_PARAMS = bo.rf_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='rf', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True, **RF_PARAMS)
                score_dict.update({'rf': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'rf': RF_PARAMS})
        except Exception as e:
            print(e)


    if 'ada' in model_list:
        try:
            print("Apply bayesian optimization to train Adaboost model\n")
            ADA_PARAMS = bo.ada_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='ada', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True, **ADA_PARAMS)
                score_dict.update({'ada': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'ada': ADA_PARAMS})
        except Exception as e:
            print(e)


    if 'gb' in model_list:
        try:
            print("Apply bayesian optimization to train GBDT model\n")
            GB_PARAMS = bo.gb_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='gb', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True, **GB_PARAMS)
                score_dict.update({'gb': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'gb': GB_PARAMS})
        except Exception as e:
            print(e)


    if 'et' in model_list:
        try:
            print("Apply bayesian optimization to train ExtraTrees model\n")
            ET_PARAMS = bo.et_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='et', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True, **ET_PARAMS)
                score_dict.update({'et': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'et': ET_PARAMS})
        except Exception as e:
            print(e)


    if 'ovr' in model_list:
        try:
            print("Apply bayesian optimization to train OneVsRest model\n")
            OVR_PARAMS = bo.ovr_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='ovr', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True, **OVR_PARAMS)
                score_dict.update({'ovr': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'ovr': OVR_PARAMS})
        except Exception as e:
            print(e)


    if 'lr' in model_list:
        try:
            with HiddenPrints():
                score = cv_check(x, y, model_name='lr', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True)
                score_dict.update({'lr': score[1].loc['val', eval_metric]['mean']})
        except Exception as e:
            print(e)


    if 'knn' in model_list:
        try:
            print("Apply bayesian optimization to train KNN model\n")
            KNN_PARAMS = bo.knn_bo()
            print('\n' + '_ ' * 60 + '\n')
            with HiddenPrints():
                score = cv_check(x, y, model_name='knn', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True)
                score_dict.update({'knn': score[1].loc['val', eval_metric]['mean']})
                params_dict.update({'knn': KNN_PARAMS})
        except Exception as e:
            print(e)


    if 'gnb' in model_list:
        try:
            with HiddenPrints():
                score = cv_check(x, y, model_name='gnb', xgb_params=None, x_test=None, y_test=None,
                                 cv=cv, random_state=random_state, silent=True)
                score_dict.update({'gnb': score[1].loc['val', eval_metric]['mean']})
        except Exception as e:
            print(e)


    print("CV performance of each model:", score_dict)
    best_model = sorted(score_dict.items(), key=lambda score_dict: score_dict[1], reverse=True)[0][0]

    if best_model not in ['lr', 'gnb']:
        best_model_params = params_dict[best_model]
        print("Best model is {i} with params: {j}".format(i=best_model, j=best_model_params))
        return best_model, best_model_params, score_dict, params_dict

    else:
        print("Best model is {i}".format(i=best_model))
        return best_model, None, score_dict, params_dict
