import pandas as pd;
import numpy as np
import statsmodels.api as sm
from sklearn.utils.multiclass import type_of_target

from tools import *
from feature_evaluation import get_iv, get_vif_cor
from sample_splitter import sample_splitter
from model_training import xgbt, rf
from model_evaluation import get_xgb_fi


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 11. FEATURE SELECTION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class FeatureFilter(object):
    """
    Filter features based on information value and model's feature importance.
    """

    def __init__(self,
                 target=[],
                 exclude_list=[],
                 silent=True):
        """
        Parameters
        ----------
        target: list
                List of default boolean targets
        exclude_list: list
                Feature excluded from selection
        silent: boolean
                Restrict prints of filtering process
        """
        self.target = target
        self.exclude_list = exclude_list
        self.silent = silent

    def iv_filter_fit(self, df, label, drop_suspicous=True, cov_appd_list=[], cav_appd_list=[], iv_floor=0.02, iv_cap=1,
                      trimmr=None, bins=10, woe_min=-20, woe_max=20, **kwargs):
        """
        Calculate information value for each feature and get a list of features to drop.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        drop_suspicous: boolean
                Drop features having iv beyond iv_cap
        cav_appd_list: list
                Extra categorical variables to be added
        cov_appd_list: list
                extra cotinuous variables to be added
        iv_floor: int or float
                Minimum of acceptable iv
        iv_cap: int or float
                Maximum of acceptable iv
        trimmr: <feature_trimming.FeatureTrimmer>
                Trimmer used to trim continuous variables
        bins: int
                Number of bins
        woe_min: int or float
                Minimum of woe value
        woe_max: int or float
                Maximum of woe value
        kwargs: dict
                Dictionary of params for decision tree
        """
        self.df_iv = get_iv(df, target=label, trimmr=None, cav_appd_list=cav_appd_list,
                            cov_appd_list=cov_appd_list, exclude_list=self.exclude_list,
                            bins=bins, woe_min=woe_min, woe_max=woe_max, **kwargs)
        self.weak_feat = self.df_iv[self.df_iv[self.df_iv.columns[0]] < iv_floor].index.tolist()
        self.suspicous_feat = self.df_iv[self.df_iv[self.df_iv.columns[0]] > iv_cap].index.tolist()
        drop_feats = self.weak_feat
        self.drop_suspicous = drop_suspicous
        if self.drop_suspicous:
            drop_feats += self.suspicous_feat
        self.df_iv_new = self.df_iv.drop(drop_feats, axis=0)
        if not self.silent:
            print('Features shown as below should be dropped\n')
            print(self.df_iv.loc[self.suspicous_feat + self.weak_feat])
            print('\n%s\n'%('_ ' * 60))
            print('These features are weak, cuz their IVs do not exceed defined floor: {thr}\n\n' \
                  .format(thr=iv_floor), '\n '.join(self.weak_feat))
            print('\n%s\n'%('_ ' * 60))
            if self.drop_suspicous:
                print('These features are suspicous, cuz their IVs exceed defined cap: {thr}\n\n' \
                      .format(thr=iv_cap), '\n '.join(self.suspicous_feat))
                print('\n%s\n'%('_ ' * 60))
            print('Remaining important features are shown as below:\n\n', self.df_iv_new.head(10))

    def iv_filter_transform(self, df):
        """
        Drop features based on their information value.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        df.drop(list(set(self.weak_feat) & set(df.columns)), axis=1, inplace=True)
        if self.drop_suspicous:
            df.drop(list(set(self.suspicous_feat) & set(df.columns)), axis=1, inplace=True)
        return df.apply(pd.to_numeric, errors='ignore')

    def iv_filter_fit_transform(self, df, label, drop_suspicous=True, cov_appd_list=[], cav_appd_list=[], iv_floor=0.02, iv_cap=1,
                                   trimmr=None, bins=10, woe_min=-20, woe_max=20, **kwargs):
        """
        Calculate information value for each feature and drop some features.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        cav_appd_list: list
                Extra categorical variables to be added
        cov_appd_list: list
                Extra cotinuous variables to be added
        iv_floor: int or float
                Minimum of acceptable iv
        iv_cap: int or float
                Maximum of acceptable iv
        trimmr: <feature_trimming.FeatureTrimmer>
                Trimmer used to trim continuous variables
        bins: int
                Number of bins
        woe_min: int or float
                Minimum of woe value
        woe_max: int or float
                Maximum of woe value
        kwargs: dict
                Dictionary of params for decision tree

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        self.iv_filter_fit(df, label, drop_suspicous=drop_suspicous, cov_appd_list=cov_appd_list,
                           cav_appd_list=cav_appd_list, iv_floor=iv_floor, iv_cap=iv_cap, trimmr=trimmr,
                           bins=bins, woe_min=woe_min, woe_max=woe_max, **kwargs)
        return self.iv_filter_transform(df)

    def xgbFI_filter_fit(self, df, label, df_va=None, xgb_params=XGB_PARAMS, alpha=0.7, top=20,
                         random_state=2019):
        """
        Train xgb model, select features based their performances in the model and get a list of features to keep.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        xgb_params: dict
                Parameters to train xgb model
        alpha: float
                Weight importance_type 
                (larger the alpha, larger the weight for importance_type = gain)
        top: int
                Number of features from each importance_type to keep when method is interaction and 
                number of features to keep in total when method is rank
        random_state: int
                Random seed
        """
        feat = list(set(df) - set(self.exclude_list + self.target)) + [label]
        with HiddenPrints():
            if df_va is None:
                x_train, y_train, x_val, y_val = sample_splitter (df[feat], label, val_size=0, dt_col=None, 
                                                                  method='random', random_state=random_state,
                                                                  drop_dt_col=False)
            else:
                x_train, y_train, x_val, y_val = df[feat].drop([label], axis=1), df[label], \
                                                 df_va[feat].drop([label], axis=1), df_va[label]
            model = xgbt(x_train, y_train, x_val, y_val, x_test=None, params=xgb_params, make_prediction=False)
        self.xgb_fi, self.xgb_important_feat = get_xgb_fi(model, method='rank', alpha=alpha, top=top)
        if not self.silent:
            print('Features shown as are important according to XGBoost model\'s feature importance ranking\n')

    def xgbFI_filter_transform(self, df, label):
        """
        return the dataframe with important features based on xgb feature importance.

        Parameters
        ----------
        df: the input dataframe

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        df = df[list((set(self.xgb_important_feat) & set(df.columns)) | set(self.exclude_list + self.target + [label]))]
        for i in set(self.xgb_important_feat) - set(df.columns):
            df.loc[:, i] = 0
        return df

    def xgbFI_filter_fit_transform(self, df, label, df_va=None, xgb_params=XGB_PARAMS, alpha=0.7, top=20,
                                      random_state=2019):
        """
        Train Xgboost model, select features based on their performances in the model and return the dataframe
        with important features.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        xgb_params: dict
                Parameters to train xgb model
        alpha: float
                Weight importance_type 
                (larger the alpha, larger the weight for importance_type = gain)
        top: int
                Number of features from each importance_type to keep when method is interaction and 
                number of features to keep in total when method is rank
        random_state: int
                Random seed

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        self.xgbFI_filter_fit(df, label, df_va=df_va, xgb_params=xgb_params, alpha=alpha, top=top,
                                 random_state=random_state)
        return self.xgbFI_filter_transform(df, label)

    def rfFI_filter_fit(self, df, label, top=20, **kwargs):
        """
        Train randomForest model, select features by their performances in the model and get the list of features to keep.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        top: int
                Number of features to keep
        """
        feat = list(set(df) - set(self.exclude_list + self.target)) + [label]
        x_train, y_train = df[feat].drop([label], axis=1), df[label]
        with HiddenPrints():
            model = rf(x_train, y_train, **kwargs)
        self.rf_fi = pd.DataFrame(model.feature_importances_, index=x_train.columns,
                                  columns=['feature importance']).sort_values(by='feature importance', ascending=False)
        self.rf_important_feat = self.rf_fi.head(top).index.tolist()
        if not self.silent:
            print('Features shown as are important according to RandomForest model\'s feature importance ranking\n')

    def rfFI_filter_transform(self, df, label):
        """
        return the dataframe with important features.

        Parameters
        ----------
        df: the input dataframe

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        df = df[list((set(self.rf_important_feat) & set(df.columns)) | set(self.exclude_list + self.target + [label]))]
        for i in set(self.rf_important_feat) - set(df.columns):
            df.loc[:, i] = 0
        return df

    def rfFI_filter_fit_transform(self, df, label, top=20, **kwargs):
        """
        Train randomForest model, select features based their performances in the model and return the dataframe
        with important features.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        top: int
                Number of features to keep

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        self.rfFI_filter_fit(df, label, top=top, **kwargs)
        return self.rfFI_filter_transform(df, label)

    def lr_pvalue_filter_fit(self, df, label, model_type='classification', bidirectional=True, feat_cnt=15, alpha=0.05):
        """
        Train logisticRegression model, select features based their performances in the model.
        When measuring their performances, p-value is applied.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        bidirectional: boolean
                Use bidirectional stepwise selection           
        feat_cnt: int
                Number of features to keep at most, active when score_based is False
        alpha: float
                Significant level (p value) for model selection, active when score_based is False
        """
        feat_left = list(set(df) - set(self.exclude_list + self.target + [label]))
        if sum(df[feat_left].dtypes == object) != 0:
            raise ValueError("LR model can't deal with categorical features")
        self.lr_important_feat = []
        score, best_score = 0, 0
        while feat_left and best_score <= alpha and len(self.lr_important_feat) < feat_cnt:
            new_score_list = []
            for i in feat_left:
                new_feat = self.lr_important_feat + [i]
                if model_type == 'classification':
                    model_forward = sm.Logit(df[label], df[new_feat]).fit(disp=False)
                elif model_type == 'regression':
                    model_forward = sm.OLS(df[label], df[new_feat]).fit(disp=False)
                else:
                    raise ValueError("Model type must be classification or regression")
                score = model_forward.pvalues[i]
                new_score_list.append((score, i))
            new_score_list.sort(reverse=True)
            best_score, feat_to_add = new_score_list.pop()
            if best_score <= alpha:
                feat_left.remove(feat_to_add)
                self.lr_important_feat.append(feat_to_add)
                if not self.silent:
                    print(feat_to_add + ' enters: p-value = ' + str(np.round(best_score, 4)))
            if len(self.lr_important_feat)>1 and bidirectional:
                if model_type == 'classification':
                    model_backword = sm.Logit(df[label], df[self.lr_important_feat]).fit(disp=False)
                elif model_type == 'regression':
                    model_backword = sm.OLS(df[label], df[self.lr_important_feat]).fit(disp=False)
                else:
                    raise ValueError("Model type must be classification or regression")
                for i in self.lr_important_feat:
                    if model_backword.pvalues[i] > alpha:
                        self.lr_important_feat.remove(i)
                        if not self.silent:
                            print(i + ' removed: p-value = ' + str(np.round(model_backword.pvalues[i], 4)))

        print('\n' + '_ ' * 60 + ' \n')
        print('Features selected by Logistic Regression Model are shown as below:\n\n',
              '\n '.join(self.lr_important_feat))

    def lr_score_filter_fit(self, df, label, model_type='classification', bidirectional=True, standard='bic'):
        """
        Train logisticRegression model, select features based their performances in the model.
        When measuring their performances, information criterion is applied.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        bidirectional: boolean
                Use bidirectional stepwise selection           
        standard: str
                Standard used to select features ('AIC' or 'BIC')
        feat_cnt: int
                Number of features to keep at most, active when score_based is False
        alpha: float
                Significant level (p value) for model selection, active when score_based is False
        """
        feat_left = list(set(df) - set(self.exclude_list + self.target + [label]))
        if sum(df[feat_left].dtypes == object) != 0:
            raise ValueError("LR model can't deal with categorical features")
        feat = np.abs(df[feat_left+[label]].corr()[label]).sort_values(ascending=False).index[1]
        feat_left.remove(feat)
        self.lr_important_feat = [feat]
        if model_type == 'classification':
            prev_model = sm.Logit(df[label], df[self.lr_important_feat]).fit(disp=False)
        elif model_type == 'regression':
            prev_model = sm.OLS(df[label], df[self.lr_important_feat]).fit(disp=False)
        else:
            raise ValueError("Model type must be classification or regression")
        if standard=='bic':
            best_score_all, best_score = prev_model.bic, 0
        elif standard=='aic':
            best_score_all, best_score = prev_model.aic, 0
        else:
            raise ValueError("standard type must be aic or bic")
        while feat_left:
            new_score_list = []
            for feat in feat_left:
                new_feats = self.lr_important_feat + [feat]
                if model_type == 'classification':
                    model_forward = sm.Logit(df[label], df[new_feats]).fit(disp=False)
                elif model_type == 'regression':
                    model_forward = sm.OLS(df[label], df[new_feats]).fit(disp=False)
                if standard=='bic':
                    score = model_forward.bic
                if standard=='aic':
                    score = model_forward.aic
                new_score_list.append((score, 'add', feat))
            if len(self.lr_important_feat)>1 and bidirectional:
                for feat in self.lr_important_feat:
                    new_feats = [f for f in self.lr_important_feat if f!=feat]
                    if model_type == 'classification':
                        model_backward = sm.Logit(df[label], df[new_feats]).fit(disp=False)
                    elif model_type == 'regression':
                        model_backward = sm.OLS(df[label], df[new_feats]).fit(disp=False)
                    if standard=='bic':
                        score = model_backward.bic
                    if standard=='aic':
                        score = model_backward.aic
                    new_score_list.append((score, 'remove', feat))
            new_score_list.sort(reverse=True)
            best_score, how, feat = new_score_list.pop()
            if float(best_score) < best_score_all:
                best_score_all = best_score
                if how=='remove':
                    self.lr_important_feat.remove(feat)
                    feat_left.append(feat)
                    if not self.silent:
                        print('%s added: %s = %f'%(feat, standard, np.round(score, 4)))
                else:
                    self.lr_important_feat.append(feat)
                    feat_left.remove(feat)
                    if not self.silent:
                        print('%s removed: %s = %f'%(feat, standard, np.round(score, 4)))
            else:
                print('\n' + '_ ' * 60 + ' \n')
                print('Features selected by Logistic Regression Model are shown as below:\n\n',
                      '\n '.join(self.lr_important_feat))
                break

    def lr_filter_fit(self, df, label, score_based=True, model_type='classification',
                      bidirectional=True, standard='bic', feat_cnt=15, alpha=0.05):
        """
        Train logisticRegression model, select features by their performances in the model.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        score_based: boolean
                If score_based is True, use Information Criterion to select features, otherwise p-value method is applied
        bidirectional: boolean
                Use bidirectional stepwise selection           
        standard: str
                Standard used to select features ('AIC' or 'BIC')
        feat_cnt: int
                Number of features to keep at most, active when score_based is False
        alpha: float
                Significant level (p value) for model selection, active when score_based is False
        """
        if score_based:
            self.lr_score_filter_fit(df, label, model_type=model_type, bidirectional=bidirectional, standard=standard)
        else:
            self.lr_pvalue_filter_fit(df, label, model_type=model_type, bidirectional=bidirectional, feat_cnt=feat_cnt, alpha=alpha)

    def lr_filter_transform(self, df, label):
        """        return the dataframe with important features.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        df = df[list((set(self.lr_important_feat) & set(df.columns)) | set(self.exclude_list + self.target + [label]))]
        for i in set(self.lr_important_feat) - set(df.columns):
            df.loc[:, i] = 0
        return df

    def lr_filter_fit_transform(self, df, label, score_based=True, model_type='classification',
                                bidirectional=True, standard='bic', feat_cnt=15, alpha=0.05):
        """
        Train logisticRegression model, select features based on their performances in the model and return the dataframe.
        with important features.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        score_based: boolean
                If score_based is True, use Information Criterion to select features, otherwise p-value method is applied
        bidirectional: boolean
                Use bidirectional stepwise selection           
        standard: str
                Standard used to select features ('AIC' or 'BIC')
        feat_cnt: int
                Number of features to keep at most, active when score_based is False
        alpha: float
                Significant level (p value) for model selection, active when score_based is False        

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        self.lr_filter_fit(df, label, score_based=score_based, model_type=model_type,
                           bidirectional=bidirectional, standard=standard, feat_cnt=feat_cnt, alpha=alpha)
        return self.lr_filter_transform(df, label)

    def cor_vif_filter_fit(self, df, label, cor_thr=0.9, vif_thr=10):
        """
        Select features based on correlation matrix and VIFs.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        cor_thr: float
                correlation threshold for feature selection
        vif_thr: float
                VIF threshold for feature selection

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        if not hasattr(self, 'df_iv'):
            self.iv_filter_fit(df, label)
        feat = list(set(df) - set(self.exclude_list + self.target + [label]))
        self.cor_drop_col, self.vif_drop_col = [], []
        df_iv = self.df_iv[[self.df_iv.columns[0]]]
        df_iv.columns=['iv']
        df_iv = df_iv.sort_values(by='iv', ascending=False)
        cor_high = df[[i for i in df_iv.index if i in feat]].corr().applymap(lambda x: np.nan if abs(x)>cor_thr else x).isnull()
        feat_left = list(cor_high)
        i = 0
        while i < len(feat_left)-1:
            ex_index = cor_high.iloc[:,i][i+1:].index[np.where(cor_high.iloc[:,i][i+1:])].tolist()
            for feat in ex_index:
                feat_left.remove(feat)
                self.cor_drop_col.append(feat)
            cor_high = cor_high.loc[feat_left, feat_left]
            i += 1

        dropped = True
        while dropped:
            dropped = False
            vif, cor = get_vif_cor(df[feat_left], target=self.target)
            vif = vif.sort_values(by='vif')
            max_vif_feat, max_vif = vif.iloc[0].index, vif.iloc[0]['vif']
            if max_vif > vif_thr:
                feat_left.remove(max_vif_feat)
                self.vif_drop_col.append(max_vif_feat)
                dropped = True

        if not self.silent:
            print('Features shown as below are dropped according to the correlation matrix\n\n',
                  '\n '.join(self.cor_drop_col))
            print('\n' + '_ ' * 60 + ' \n')
            print('Features shown as below are dropped according to VIFs\n\n',
                  '\n '.join(self.vif_drop_col))

    def cor_vif_filter_transform(self, df, label):
        """
        Select features based on correlation matrix and VIFs.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        if self.cor_drop_col!=[]:
            df.drop(list(set(self.cor_drop_col) & set(df.columns)), axis=1, inplace=True)
        if self.vif_drop_col!=[]:
            df.drop(list(set(self.vif_drop_col) & set(df.columns)), axis=1, inplace=True)
        return df.apply(pd.to_numeric, errors='ignore')

    def cor_vif_filter_fit_transform(self, df, label, cor_thr=0.9, vif_thr=10):
        """
        Select features based on correlation matrix and VIFs.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used to train models
        cor_thr: float
                correlation threshold for feature selection
        vif_thr: float
                VIF threshold for feature selection

        Returns
        ----------
        df: pd.DataFrame
                The output dataframe with selected features
        """
        self.cor_vif_filter_fit(df, label, cor_thr=cor_thr, vif_thr=vif_thr)
        return self.cor_vif_filter_transform(df, label)
