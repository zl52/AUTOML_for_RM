import pandas as pd;
import numpy as np

from tools import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

import xgboost as xgb
import xgboost.sklearn
from xgboost import XGBClassifier

from bayes_opt import BayesianOptimization

from model_evaluation import model_summary
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   12. MODEL TRAINING   ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def cv_check(x, y, clf_name='xgb', xgb_params=XGB_PARAMS, x_test=None, y_test=None, cv=5,
             random_state=2019, silent=True, **kwargs):
    """
    Cross validation check

    : params x: x
    : params y: y
    : params clf_name: model type
    : params xgb_params: xgb parameters
    : params x_test: x_test
    : params y_test: y_test
    : params cv: number of folds
    : params random_state: seed
    : params silent: whether to print cv process

    : params df: dataframe recording cv process
    : params summary: statistical summary of df
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
            clf, pred_train_value, pred_val_value, pred_test_value = \
                model_toolkit(clf_name, x_train, y_train, x_val=x_val, y_val=y_val, x_test=None,
                              xgb_params=xgb_params, make_prediction=True, **kwargs)

            res = model_summary(pred_train_value, y_train, pred_val_value, y_val,
                                pred_test_value=pred_test_value, real_test_label=y_test,
                                ax=None, pos_label=1, use_formater=False, plot=False)

        else:
            clf, pred_train_value, pred_val_value = \
                model_toolkit(clf_name, x_train, y_train, x_val=x_val, y_val=y_val, x_test=x_test,
                              xgb_params=xgb_params, make_prediction=True, **kwargs)

            res = model_summary(pred_train_value, y_train, pred_val_value, y_val,
                                pred_test_value=None, real_test_label=None,
                                ax=None, pos_label=1, use_formater=False, plot=False)

        df = pd.concat([df, res], axis=0)

    summary = df.reset_index(drop=False).groupby('index').agg([np.mean, np.std])

    return df, summary


def model_toolkit(clf_name, x_train, y_train, x_val=None, y_val=None, x_test=None,
                  xgb_params=XGB_PARAMS, make_prediction=False, **kwargs):
    """
    toolkit integrating training and predicting process for all model types

    : params clf_name: model type
    : params x_train: independent variable of train set
    : params y_train: dependent variable of train set
    : params x_val: independent variables of validation set
    : params y_val: dependent variable of validation set
    : params x_test: independent variables of test set
    : params y_test: dependent variable of test set
    : params xgb_params: xgb parameters
    : params make_prediction: whether to output predictions for each set

    : return: model (predictions for train, validation and test sets)
    """
    if clf_name == 'xgb':
        if (x_val is not None) & (y_val is not None):
            if x_test is None:
                clf, pred_train_value, pred_val_value = xgbt(x_train
                                                             , y_train
                                                             , x_val
                                                             , y_val
                                                             , None
                                                             , params=xgb_params
                                                             , make_prediction=True
                                                             , **kwargs)
            else:
                clf, pred_train_value, pred_val_value, pred_test_value = xgbt(x_train
                                                                              , y_train
                                                                              , x_val
                                                                              , y_val
                                                                              , x_test
                                                                              , params=xgb_params
                                                                              , make_prediction=True
                                                                              , **kwargs)
        else:
            clf, pred_train_value = xgbt(x_train
                                         , y_train
                                         , x_val
                                         , y_val
                                         , x_test
                                         , params=xgb_params
                                         , make_prediction=True
                                         , **kwargs)

    else:
        try:
            if clf_name == 'rf':
                clf = rf(x_train, y_train, **kwargs)

            elif clf_name == 'ada':
                clf = ada(x_train, y_train, **kwargs)

            elif clf_name == 'gb':
                clf = gb(x_train, y_train, **kwargs)

            elif clf_name == 'et':
                clf = et(x_train, y_train, **kwargs)

            elif clf_name == 'ovr':
                clf = ovr(x_train, y_train, **kwargs)

            elif clf_name == 'gnb':
                clf = gnb(x_train, y_train)

            elif clf_name == 'lr':
                clf = lr(x_train, y_train, **kwargs)
                x_val['intercept'] = 1
                if x_test is not None:
                    x_test['intercept'] = 1

            elif clf_name == 'lsvc':
                clf = lsvc(x_train, y_train, **kwargs)

            elif clf_name == 'knn':
                clf = knn(x_train, y_train, **kwargs)

            if clf_name not in ['lsvc', 'lr']:
                pred_train_value = clf.predict_proba(x_train)[:, 1]
                pred_val_value = clf.predict_proba(x_val)[:, 1]

            else:
                pred_train_value = clf.predict(x_train)
                pred_val_value = clf.predict(x_val)

            if x_test is not None:
                if clf_name not in ['lsvc', 'lr']:
                    pred_test_value = clf.predict_proba(x_test)[:, 1]

                else:
                    pred_test_value = clf.predict(x_test)

        except:
            print('Valid abbrs of classifiers are xgb, rf,ada,gb,et,ovr,gnb,lr,lsvc and knn\n')

    if make_prediction == True:
        if x_test is not None:
            print('Average of predictive values for train set:', '{:.1f}%'.format(np.mean(pred_train_value) * 100))
            print('Average of predictive values for validation set:', '{:.1f}%'.format(np.mean(pred_val_value) * 100))
            print('Average of predictive values for test set:', '{:.1f}%'.format(np.mean(pred_test_value) * 100))

            return clf, pred_train_value, pred_val_value, pred_test_value

        else:
            print('Average of predictive values for train set:', '{:.1f}%'.format(np.mean(pred_train_value) * 100))
            print('Average of predictive values for validation set:', '{:.1f}%'.format(np.mean(pred_val_value) * 100))

            return clf, pred_train_value, pred_val_value

    else:
        return clf


def xgbt(x_train, y_train, x_val=None, y_val=None, x_test=None, params=XGB_PARAMS, num_boost_round=10000,
         early_stopping_rounds=50, make_prediction=False, **kwargs):
    """
    toolkit of training and predicting process for xgb model

    : params clf_name: model type
    : params x_train: independent variable of train set
    : params y_train: dependent variable of train set
    : params x_val: independent variables of validation set
    : params y_val: dependent variable of validation set
    : params x_test: independent variables of test set
    : params y_test: dependent variable of test set
    : params params: xgb parameters
    : params num_boost_round: num_boost_round for xgb model
    : params early_stopping_rounds: early_stopping_rounds for xgb model
    : params make_prediction: whether to output predictions for each set

    : return: model (predictions for train, validation and test sets)
    """
    if (x_val is not None) & (y_val is not None):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(params
                          , dtrain
                          , num_boost_round=num_boost_round
                          , evals=watchlist
                          , verbose_eval=True
                          , early_stopping_rounds=early_stopping_rounds
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
    if kwargs != {}:
        randomforest = RandomForestClassifier(**kwargs)

    else:
        randomforest = RandomForestClassifier(n_estimators=200
                                              , max_depth=None
                                              , max_leaf_nodes=5
                                              , min_samples_leaf=40
                                              , min_samples_split=2
                                              , min_weight_fraction_leaf=0.0
                                              , random_state=2019
                                              , max_features="auto"
                                              , class_weight='balanced'
                                              , criterion='gini'
                                              , n_jobs=-1
                                              , verbose=0
                                              , bootstrap=True
                                              , oob_score=True
                                              , warm_start=False)

    clf = randomforest.fit(x_train, y_train)

    return clf


def ada(x_train, y_train, **kwargs):
    if kwargs != {}:
        adaboost = AdaBoostClassifier(**kwargs)

    else:
        adaboost = AdaBoostClassifier(n_estimators=400
                                      , learning_rate=0.01
                                      , base_estimator=None
                                      , algorithm='SAMME.R'
                                      , random_state=2019)

    clf = adaboost.fit(x_train, y_train)

    return clf


def gb(x_train, y_train, **kwargs):
    if kwargs != {}:
        gbdt = GradientBoostingClassifier(**kwargs)

    else:
        gbdt = GradientBoostingClassifier(learning_rate=0.01
                                          , n_estimators=400
                                          , max_depth=5
                                          , subsample=0.8
                                          , min_samples_split=2
                                          , min_samples_leaf=1
                                          , min_impurity_split=1e-7
                                          , verbose=0
                                          , random_state=2019
                                          , loss='deviance'
                                          , max_features=None
                                          , max_leaf_nodes=None)

    clf = gbdt.fit(x_train, y_train)

    return clf


def et(x_train, y_train, **kwargs):
    if kwargs != {}:
        extratree = ExtraTreesClassifier(**kwargs)

    else:
        extratree = ExtraTreesClassifier(n_estimators=400
                                         , max_depth=8
                                         , max_features="auto"
                                         , n_jobs=-1
                                         , random_state=2019
                                         , verbose=0
                                         , criterion='gini')

    clf = extratree.fit(x_train, y_train)

    return clf


def ovr(x_train, y_train, **kwargs):
    if kwargs != {}:
        est = RandomForestClassifier(**kwargs)

    else:
        est = RandomForestClassifier(n_estimators=400
                                     , max_depth=8
                                     , n_jobs=-1
                                     , random_state=2019
                                     , max_features="auto"
                                     , verbose=0)

    ovr = OneVsRestClassifier(est, n_jobs=-1)
    clf = ovr.fit(x_train, y_train)

    return clf


def gnb(x_train, y_train):
    gnb = GaussianNB()
    clf = gnb.fit(x_train, y_train)

    return clf


def lr(x_train, y_train, **kwargs):
    x_train['intercept'] = 1
    logisticregression = sm.Logit(y_train, x_train)
  
    clf = logisticregression.fit(disp=False)
    model_summary = clf.summary2(alpha=0.05)
    print(model_summary)

    if kwargs != {}:
        try:
            if kwargs['print'] == True:
                output_equation_txt = "df.loc[:,'intercept'] = 1\n"
                output_equation_txt += "df['logit'] = "

                for idx in clf.params.index:
                    output_equation_txt += "\\\n" + "(".rjust(15) + str(clf.params[idx]) + ") * df[" \
                                        + "'"+idx+"'" + "] + "
                output_equation_txt += "0\n\ndf['score'] = [1/(1+exp(-logit)) for logit in df['logit']]"
                write_recoding_txt(output_equation_txt, kwargs['equation_file'], encoding="utf-8")

                model_summary_output = pd.DataFrame()
                model_summary_output['Estimate'] = clf.params
                model_summary_output['SE'] = clf.bse
                model_summary_output['T_value'] = clf.tvalues
                model_summary_output['P_value'] = clf.pvalues
                model_summary_output = model_summary_output.join(kwargs['vif'])
                model_summary_output.to_excel(kwargs['model_summary_file'], index = None)
        except:
            raise ValueError('Params \'print\', \'equation_file\' and \'model_summary_file\' are needed')

    return clf


def lsvc(x_train, y_train, **kwargs):
    if kwargs != {}:
        linearsvc = LinearSVC(**kwargs)

    else:
        linearsvc = LinearSVC(C=0.1
                              , penalty='l2'
                              , loss='squared_hinge'
                              , verbose=0
                              , dual=False
                              , max_iter=-1
                              , random_state=2019)

    clf = linearsvc.fit(x_train, y_train)

    return clf


def knn(x_train, y_train, **kwargs):
    x_train = np.log10(x_train + 1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0

    if kwargs != {}:
        kneighbors = KNeighborsClassifier(**kwargs)

    else:
        kneighbors = KNeighborsClassifier(n_neighbors=20
                                          , weights='uniform'
                                          , algorithm='auto'
                                          , n_jobs=-1)

    clf = kneighbors.fit(x_train, y_train)

    return clf


class BAYSIAN_OPTIMIZATION():
    def __init__(self, x, y, init_points=20, n_iter=40, acq='ei'):
        self.x = x
        self.y = y
        self.init_points = init_points
        self.n_iter = n_iter
        self.acq = acq

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
            {'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'min_child_weight': int(res['params']['min_child_weight']),
             'eta': res['params']['eta'],
             'max_depth': int(res['params']['max_depth']),
             'subsample': res['params']['subsample'],
             'colsample_bytree': res['params']['colsample_bytree'],
             'gamma': res['params']['gamma'],
             'lambdA': res['params']['lambdA'],
             'alpha': res['params']['alpha']
             }
        print('Best combination of parameters are shownn as below\n', XGB_PARAMS)

        return XGB_PARAMS

    def xgb_cvscore_4bo(self, eta, min_child_weight, max_depth, subsample, colsample_bytree, gamma, lambdA, alpha,
                        score='auc', cv=5):
        XGB_PARAMS = \
            {'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'min_child_weight': int(min_child_weight),
             'eta': eta,
             'max_depth': int(max_depth),
             'subsample': subsample,
             'colsample_bytree': colsample_bytree,
             'gamma': gamma,
             'lambdA': lambdA,
             'alpha': alpha
             }

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='xgb', xgb_params=XGB_PARAMS, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True)

        return summary.loc['val', score]['mean']

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
             'verbose': 0,
             'n_estimators': int(res['params']['n_estimators']),
             'max_depth': int(res['params']['max_depth']),
             'min_samples_split': int(res['params']['min_samples_split']),
             'min_samples_leaf': res['params']['min_samples_leaf'],
             'max_features': res['params']['max_features']
             }
        print('Best combination of parameters are shownn as below\n', RF_PARAMS)

        return RF_PARAMS

    def rf_cvscore_4bo(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                       max_features, score='auc', cv=5):
        RF_PARAMS = \
            {'n_jobs': -1,
             'verbose': 0,
             'n_estimators': int(n_estimators),
             'max_depth': int(max_depth),
             'min_samples_split': int(min_samples_split),
             'min_samples_leaf': min_samples_leaf,
             'max_features': max_features
             }

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='rf', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **RF_PARAMS)

        return summary.loc['val', score]['mean']

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
             'algorithm': 'SAMME.R',
             'n_estimators': int(res['params']['n_estimators']),
             'learning_rate': (res['params']['learning_rate']),
             }
        print('Best combination of parameters are shownn as below\n', ADA_PARAMS)

        return ADA_PARAMS

    def ada_cvscore_4bo(self, n_estimators, learning_rate, score='auc', cv=5):
        ADA_PARAMS = \
            {'base_estimator': None,
             'algorithm': 'SAMME.R',
             'n_estimators': int(n_estimators),
             'learning_rate': learning_rate
             }
        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='ada', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **ADA_PARAMS)

        return summary.loc['val', score]['mean']

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
             'loss': 'deviance',
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
        print('Best combination of parameters are shownn as below\n', GB_PARAMS)

        return GB_PARAMS

    def gb_cvscore_4bo(self, n_estimators, max_depth, learning_rate, min_samples_split, min_samples_leaf,
                       min_impurity_split, subsample, score='auc', cv=5):
        GB_PARAMS = \
            {'verbose': 0,
             'loss': 'deviance',
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

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='gb', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **GB_PARAMS)

        return summary.loc['val', score]['mean']

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

        print('Best combination of parameters are shownn as below\n', ET_PARAMS)

        return ET_PARAMS

    def et_cvscore_4bo(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                       max_features, score='auc', cv=5):
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
        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='et', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **ET_PARAMS)

        return summary.loc['val', score]['mean']

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
             'n_estimators': int(res['params']['n_estimators']),
             'max_depth': int(res['params']['max_depth']),
             'min_samples_split': int(res['params']['min_samples_split']),
             'min_samples_leaf': res['params']['min_samples_leaf'],
             'max_features': res['params']['max_features']
             }
        print('Best combination of parameters are shownn as below\n', OVR_PARAMS)

        return OVR_PARAMS

    def ovr_cvscore_4bo(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                        max_features, score='auc', cv=5):
        OVR_PARAMS = \
            {'n_jobs': -1,
             'verbose': 0,
             'n_estimators': int(n_estimators),
             'max_depth': int(max_depth),
             'min_samples_split': int(min_samples_split),
             'min_samples_leaf': min_samples_leaf,
             'max_features': max_features
             }
        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='ovr', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **OVR_PARAMS)

        return summary.loc['val', score]['mean']

    def lsvc_bo(self):
        self.lsvc_bo_optimizer = BayesianOptimization(
            self.lsvc_cvscore_4bo,
            {'C': (0.001, 100)}
        )
        self.lsvc_bo_optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, acq=self.acq)
        res = self.lsvc_bo_optimizer.max
        LSVC_PARAMS = {'C': res['params']['C'],
                       'verbose': 0,
                       'max_iter': -1}
        print('Best combination of parameters are shownn as below\n', LSVC_PARAMS)

        return LSVC_PARAMS

    def lsvc_cvscore_4bo(self, C, score='auc', cv=5):
        LSVC_PARAMS = {'C': C,
                       'verbose': 0,
                       'max_iter': -1}
        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='lsvc', xgb_params=LSVC_PARAMS, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **LSVC_PARAMS)

        return summary.loc['val', score]['mean']

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
             'n_neighbors': int(res['params']['n_neighbors']),
             'leaf_size': int(res['params']['leaf_size'])
             }
        print('Best combination of parameters are shownn as below\n', KNN_PARAMS)

        return KNN_PARAMS

    def knn_cvscore_4bo(self, n_neighbors, leaf_size, score='auc', cv=5):
        KNN_PARAMS = \
            {'n_neighbors': int(n_neighbors),
             'leaf_size': int(leaf_size)
             }

        with HiddenPrints():
            _, summary = cv_check(self.x, self.y, clf_name='knn', xgb_params=None, x_test=None, y_test=None,
                                  cv=cv, random_state=2019, silent=True, **KNN_PARAMS)

        return summary.loc['val', score]['mean']


def model_selection(x, y, init_points=1, n_iter=1, acq='ei', cv=5, random_state=2019):
    bo = BAYSIAN_OPTIMIZATION(x, y, init_points=init_points, n_iter=n_iter, acq=acq)
    print("Apply bayesian optimization to XGBoost model\n")
    XGB_PARAMS = bo.xgb_bo()
    print('\n' + '_ ' * 60 + '\n')

    print("Apply bayesian optimization to RandomForest model\n")
    RF_PARAMS = bo.rf_bo()
    print('\n' + '_ ' * 60 + '\n')

    print("Apply bayesian optimization to Adaboost model\n")
    ADA_PARAMS = bo.ada_bo()
    print('\n' + '_ ' * 60 + '\n')

    print("Apply bayesian optimization to GBDT model\n")
    GB_PARAMS = bo.gb_bo()
    print('\n' + '_ ' * 60 + '\n')

    print("Apply bayesian optimization to ExtraTrees model\n")
    ET_PARAMS = bo.et_bo()
    print('\n' + '_ ' * 60 + '\n')

    print("Apply bayesian optimization to OneVsRest model\n")
    OVR_PARAMS = bo.ovr_bo()
    print('\n' + '_ ' * 60 + '\n')

    print("Apply bayesian optimization to KNN model\n")
    KNN_PARAMS = bo.knn_bo()
    print('\n' + '_ ' * 60 + '\n')

    score_dict = {}
    params_dict = {'xgb': XGB_PARAMS, 'rf': RF_PARAMS, 'ada': ADA_PARAMS, 'gb': GB_PARAMS, 'et': ET_PARAMS,
                   'ovr': OVR_PARAMS, 'knn': KNN_PARAMS}

    with HiddenPrints():
        score = cv_check(x, y, clf_name='xgb', xgb_params=XGB_PARAMS, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True)[1].loc['val', 'auc']['mean']
        score_dict.update({'xgb': score})

        score = cv_check(x, y, clf_name='rf', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True, **RF_PARAMS)[1].loc['val', 'auc']['mean']
        score_dict.update({'rf': score})

        score = cv_check(x, y, clf_name='ada', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True, **ADA_PARAMS)[1].loc['val', 'auc']['mean']
        score_dict.update({'ada': score})

        score = cv_check(x, y, clf_name='gb', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True, **GB_PARAMS)[1].loc['val', 'auc']['mean']
        score_dict.update({'gb': score})

        score = cv_check(x, y, clf_name='et', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True, **ET_PARAMS)[1].loc['val', 'auc']['mean']
        score_dict.update({'et': score})

        score = cv_check(x, y, clf_name='ovr', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True, **OVR_PARAMS)[1].loc['val', 'auc']['mean']
        score_dict.update({'ovr': score})

        score = cv_check(x, y, clf_name='lr', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True)[1].loc['val', 'auc']['mean']
        score_dict.update({'lr': score})

        score = cv_check(x, y, clf_name='gnb', xgb_params=None, x_test=None, y_test=None,
                         cv=cv, random_state=random_state, silent=True)[1].loc['val', 'auc']['mean']
        score_dict.update({'gnb': score})

    print("CV performance of each model:", score_dict)
    best_model = sorted(score_dict.items(), key=lambda score_dict: score_dict[1], reverse=True)[0][0]
    if best_model not in ['lr', 'gnb']:
        best_model_params = params_dict[best_model]
        print("Best model is {i} with params: {j}".format(i=best_model, j=best_model_params))
        return best_model, best_model_params, score_dict, params_dict

    else:
        print("Best model is {i}".format(i=best_model))
        return best_model, None, score_dict, params_dict
