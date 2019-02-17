import pandas as pd;
import numpy as np
import statsmodels.api as sm

from tools import *
from feature_evaluation import get_iv
from sample_splitter import SAMPLE_SPLITTER
from model_training import xgbt, rf
from model_evaluation import get_xgb_fi


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 10. FEATURE SELECTION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class FEATURE_FILTER(object):
    """
    Filter features by IV and model's feature importance
    """

    def __init__(self, target=TARGET, exclude_list=[], drop_weak=True, drop_suspicous=True,
                 silent=True):
        """
        : params target: list of default boolean targets
        : params exclude_list: columns to be excluded
        : params drop_weak: whether to drop features with iv below iv_floor
        : params drop_suspicous: whether to drop features with iv beyond iv_cap
        : params silent: whether to print details of filtering process
        """
        self.target = target
        self.exclude_list = exclude_list
        self.drop_weak = drop_weak
        self.drop_suspicous = drop_suspicous
        self.silent = silent

    def ud_iv_filter_fit(self, df, cov_list_appd=[], cav_list_appd=[], iv_floor=0.02, iv_cap=1,
                         trimmr=None, bins=10, woe_min=-20, woe_max=20, **kwargs):
        """
        Calculate iv for each feature and get the list of features to drop

        : params df: the input dataframe
        : params iv_floor: minimum of acceptable iv
        : params iv_cap: maximum of acceptable iv
        : params trimmr: trimmer used to trim continuous variables
        : params cav_list_appd: extra categorical variables to be added
        : params cov_list_appd: extra cotinuous variables to be added
        : params bins: number of bins
        : param woe_min: minimum of woe value
        : param woe_max: maximum of woe value
        """
        self.df_iv = get_iv(df, target=self.target, trimmr=None, cav_list_appd=cav_list_appd,
                            cov_list_appd=cov_list_appd, exclude_list=self.exclude_list,
                            bins=bins, woe_min=woe_min, woe_max=woe_max, **kwargs)

        self.weak_feat = self.df_iv[self.df_iv[self.df_iv.columns[0]] < iv_floor].index.tolist()
        self.suspicous_feat = self.df_iv[self.df_iv[self.df_iv.columns[0]] > iv_cap].index.tolist()
        drop_feats = []

        if self.drop_weak:
            drop_feats += self.weak_feat

        if self.drop_suspicous:
            drop_feats += self.suspicous_feat

        self.df_iv_new = self.df_iv.drop(drop_feats, axis=0)
        if not self.silent:
            print('Features shown as below should be dropped\n')

            print(self.df_iv.loc[self.suspicous_feat + self.weak_feat])
            print('\n' + '_ ' * 60 + ' \n')

            print('These features are weak, cuz their IVs do not exceed defined floor: {thr}\n\n' \
                  .format(thr=iv_floor), '\n '.join(self.weak_feat))
            print('\n' + '_ ' * 60 + ' \n')

            print('These features are suspicous, cuz their IVs exceed defined cap: {thr}\n\n' \
                  .format(thr=iv_cap), '\n '.join(self.suspicous_feat))
            print('\n' + '_ ' * 60 + ' \n')

            print('Remaining important features are shown as below:\n\n', self.df_iv_new.head(10))

    def ud_iv_filter_transform(self, df):
        """
         Drop features

        : params df: the input dataframe

        : return df: the output dataframe with filtered features
        """
        try:
            if self.drop_weak:
                df.drop(list(set(self.weak_feat) & set(df.columns)), axis=1, inplace=True)

            if self.drop_suspicous:
                df.drop(list(set(self.suspicous_feat) & set(df.columns)), axis=1, inplace=True)

            return df.apply(pd.to_numeric, errors='ignore')

        except:
            raise Exception("Failed to filter features")

    def ud_iv_filter_fit_transform(self, df, cov_list_appd=[], cav_list_appd=[], iv_floor=0.02, iv_cap=1,
                                   trimmr=None, bins=10, woe_min=-20, woe_max=20, **kwargs):
        """
        Calculate iv for each feature and drop features

        : params df: the input dataframe
        : params iv_floor: minimum of acceptable iv
        : params iv_cap: maximum of acceptable iv
        : params trimmr: trimmer used to trim continuous variables
        : params cav_list_appd: extra categorical variables to be added
        : params cov_list_appd: extra cotinuous variables to be added
        : params bins: number of bins
        : param woe_min: minimum of woe value
        : param woe_max: maximum of woe value

        : return df: the output dataframe with filtered features
        """
        self.ud_iv_filter_fit(df, cov_list_appd=cov_list_appd, cav_list_appd=cav_list_appd,
                              iv_floor=iv_floor, iv_cap=iv_cap, trimmr=trimmr, bins=bins,
                              woe_min=woe_min, woe_max=woe_max, **kwargs)

        return self.ud_iv_filter_transform(df)

    def ud_xgbfi_filter_fit(self, df, label, df_va=None, xgb_params=XGB_PARAMS, alpha=0.7, top=20,
                            random_state=2019):
        """
        Train xgb model, order features by their performances in the model and get the list of features to keep

        : params df: the input dataframe
        : params label: boolean label
        : params xgb_params: parameters for training xgb model
        : params alpha: weight parameter for importance_type 
                        (larger the alpha, larger the weight for importance_type = gain)
        : params top: number of features from each importance_type to keep when method is interaction and 
                      number of features to keep in total when method is rank
        : params random_state: seed
        """
        feat = list(set(df.columns.tolist()) - set(self.exclude_list))

        with HiddenPrints():
            if df_va is None:
                x_train, y_train, x_val, y_val = SAMPLE_SPLITTER(df[feat], label, dt_col=None, uid=None,
                                                                 method='random', random_state=random_state,
                                                                 drop_dt_col=False)

            else:
                x_train, y_train, x_val, y_val = df[feat].drop([label], axis=1), df[label], \
                                                 df_va[feat].drop([label], axis=1), df_va[label]

            model = xgbt(x_train, y_train, x_val, y_val, x_test=None, params=xgb_params, make_prediction=False)

        self.xgb_fi, self.xgb_important_feat = get_xgb_fi(model, method='rank', alpha=alpha, top=top)
        if not self.silent:
            print('Features shown as are important according to XGBoost model\'s feature importance ranking\n')

    def ud_xgbfi_filter_transform(self, df, label):
        """
         Drop features

        : params df: the input dataframe

        : return df: the output dataframe with filtered features
        """
        try:
            df = df[list(set(self.xgb_important_feat) & set(df.columns)) + [label] + self.exclude_list]
            for i in set(self.xgb_important_feat) - set(df.columns):
                df.loc[:, i] = 0

            return df

        except:
            raise Exception("Failed to filter features")

    def ud_xgbfi_filter_fit_transform(self, df, label, df_va=None, xgb_params=XGB_PARAMS, alpha=0.7, top=20,
                                      random_state=2019):
        """
        Train xgb model, order features by their performances in the model and return the dataframe
        with important features

        : params df: the input dataframe
        : params label: boolean label
        : params xgb_params: parameters for training xgb model
        : params alpha: weight parameter for importance_type 
                        (larger the alpha, larger the weight for importance_type = gain)
        : params top: number of features from each importance_type to keep when method is interaction and 
                      number of features to keep in total when method is rank
        : params random_state: seed

        : return df: the output dataframe with filtered features
        """
        self.ud_xgbfi_filter_fit(df, label, df_va=df_va, xgb_params=xgb_params, alpha=alpha, top=top,
                                 random_state=random_state)

        return self.ud_xgbfi_filter_transform(df, label)

    def ud_rffi_filter_fit(self, df, label, top=20, **kwargs):
        """
        Train rf model, order features by their performances in the model and get the list of features to keep

        : params df: the input dataframe
        : params label: boolean label
        : params top: number of features to keep
        """
        feat = list(set(df.columns.tolist()) - set(self.exclude_list)) + [label]
        x_train, y_train = df[feat].drop([label], axis=1), df[label]

        with HiddenPrints():
            model = rf(x_train, y_train, **kwargs)

        self.rf_fi = pd.DataFrame(model.feature_importances_, index=x_train.columns,
                                  columns=['feature importance']).sort_values(by='feature importance', ascending=False)
        self.rf_important_feat = self.rf_fi.head(top).index.tolist()

        if not self.silent:
            print('Features shown as are important according to RandomForest model\'s feature importance ranking\n')

    def ud_rffi_filter_transform(self, df, label):
        """
         Drop features

        : params df: the input dataframe

        : return df: the output dataframe with filtered features
        """
        try:
            df = df[list(set(self.rf_important_feat) & set(df.columns)) + [label] + self.exclude_list]
            for i in set(self.rf_important_feat) - set(df.columns):
                df.loc[:, i] = 0

            return df

        except:
            raise Exception("Failed to filter features")

    def ud_rffi_filter_fit_transform(self, df, label, top=20, **kwargs):
        """
        Train rf model, order features by their performances in the model and return the dataframe
        with important features

        : params df: the input dataframe
        : params label: boolean label
        : params top: number of features to keep

        : return df: the output dataframe with filtered features
        """
        self.ud_rffi_filter_fit(df, label, top=top, **kwargs)

        return self.ud_rffi_filter_transform(df, label)

    def ud_lr_filter_fit(self, df, label, feat_cnt, alpha=0.05, stepwise=True):
        """
        Train Logistic Regression Model designed by forward selection / bidirectional selection

        : params df: the input dataframe
        : params label: boolean label
        : params feat_cnt: number of features to keep at most
        : params alpha: significant level (p value) for model selection
        : params stepwise: whether to use bidirectional stepwise selection        
        """
        feat_left = list(set(df.columns.tolist()) - set(self.exclude_list + [label]))
        if sum(df[feat_left].dtypes == object) != 0:
            raise Exception("LR model can't deal with categorical features")

        self.lr_important_feat = []
        score, best_score = 0, 0

        while feat_left and best_score <= alpha and len(self.lr_important_feat) < feat_cnt:
            new_score_list = []

            for i in feat_left:
                new_feat = self.lr_important_feat + [i]
                model_forward = sm.Logit(df[label], df[new_feat]).fit(disp=False)
                score = model_forward.pvalues[i]
                new_score_list.append((score, i))

            new_score_list.sort(reverse=True)
            best_score, feat_to_add = new_score_list.pop()

            if best_score <= alpha:
                feat_left.remove(feat_to_add)
                self.lr_important_feat.append(feat_to_add)
                if not self.silent:
                    print(feat_to_add + ' enters: p-value = ' + str(np.round(best_score, 4)))

            if stepwise:
                model_backword = sm.Logit(df[label], df[self.lr_important_feat]).fit(disp=False)
                for i in self.lr_important_feat:
                    if model_backword.pvalues[i] > alpha:
                        self.lr_important_feat.remove(i)
                        print(i + ' removed: p-value = ' + str(np.round(model_backword.pvalues[i], 4)))

        print('\n' + '_ ' * 60 + ' \n')
        print('Features selected by Logistic Regression Model are shown as below:\n\n',
              '\n '.join(self.lr_important_feat))

    def ud_lr_filter_transform(self, df, label):
        """
         Drop features

        : params df: the input dataframe
        : params label: boolean label

        : return df: the output dataframe with filtered features
        """
        try:
            df = df[list(set(self.lr_important_feat) & set(df.columns)) + [label] + self.exclude_list]
            for i in set(self.lr_important_feat) - set(df.columns):
                df.loc[:, i] = 0

            return df

        except:
            raise Exception("Failed to filter features")

    def ud_lr_filter_fit_transform(self, df, label, feat_cnt, alpha=0.05, stepwise=True):
        """
        Train LR model, order features by their performances in the model and return the dataframe
        with important features

        : params df: the input dataframe
        : params label: boolean label
        : params feat_cnt: number of features to keep at most
        : params alpha: significant level (p value) for model selection
        : params stepwise: whether to use bidirectional stepwise selection        

        : return df: the output dataframe with filtered features
        """
        self.ud_lr_filter_fit(df, label, feat_cnt=feat_cnt, alpha=alpha, stepwise=stepwise)

        return self.ud_lr_filter_transform(df, label)
