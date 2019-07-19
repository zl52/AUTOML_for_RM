import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   9. FEATURE SCALING   ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class FeatureScaler():
    """
    User-defined scaler can rescale features.
    """

    def __init__(self,
                 scaler_type='StandardScaler',
                 target=TARGET,
                 silent=False,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        scaler_type: str
                Scaler's type, including StandardScaler and MinMaxScaler
        target: list
                List of default boolean targets
        silent: boolean
                Restrict the print of transformation process
        recoding_dict: dict
                Dictionary recording recoding statements for each feature
        feat_dict: dict
                Dictionary recording changes of feature names
        """
        self.target = target
        self.scaler_type = scaler_type
        self.silent = silent
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict

    def fit(self, df, label=None, action=True, exclude_list=[], write_recoding_statement=True):
        """
        Fit the scaler by checking the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        action: boolean
                Fit scaler when the param is True. Otherwise use "nothing" imputer 
        exclude_list: list
                List of features excluded from being scaled when action is True
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in exclude_list + self.target + [label]]

        if action:
            if self.scaler_type == 'StandardScaler':
                self.scalr = StandardScaler(copy=True, with_mean=True, with_std=True)
                self.scalr.fit(df[col])
                for i, ori_name in enumerate(col):
                    mean = self.scalr.mean_[i]
                    std = np.sqrt(self.scalr.var_[i])
                    if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                        recoding_statement = ""
                        recoding_statement += "\ndf['%s'] = (df['%s'] - %s) / %s"%(ori_name, ori_name, str(mean), str(std))
                        key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                        self.feat_dict.update({key: ori_name})
                        self.recoding_dict[key] += recoding_statement
                if not self.silent:
                    print("Use StandardScaler to scale the features")

            elif self.scaler_type == 'MinMaxScaler':
                self.scalr = MinMaxScaler(feature_range=(0, 1), copy=True)
                self.scalr.fit(df[col])
                for i, ori_name in enumerate(col):
                    maximum = self.scalr.data_max_[i]
                    minimum = self.scalr.data_min_[i]
                    if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                        recoding_statement = ""
                        recoding_statement += "\ndf['%s'] = (df['%s'] - %s) / %s"%(ori_name, ori_name, str(minimum), str(maximum - minimum))
                        key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                        self.feat_dict.update({key: ori_name})
                        self.recoding_dict[key] += recoding_statement
                if not self.silent:
                    print("Use MinMaxScaler to scale the features")

            else:
                raise ValueError("scale_type must chosen among \'StandardScaler\' and \'MinMaxScaler\'")

        else:
            self.scalr = None

    def transform(self, df, label=None, action=True, exclude_list=[]):
        """
        Rescale features in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        action: boolean
                Fit scaler when the param is True. Otherwise use "nothing" imputer 
        exclude_list: list
                List of features excluded from being scaled when action is True

        Returns
        ----------
        df: pd.DataFrame
                Scaled dataframe
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in exclude_list + self.target + [label]]
        if action and not self.scalr:
            try:
                df[col] = self.scalr.transform(df[col])
            except:
                raise ValueError("Scaling process incur error")
        return df

    def fit_transform(self, df, label=None, action=True, exclude_list=[], write_recoding_statement=True):
        """
        Fit the scaler and rrescale features in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        action: boolean
                Fit scaler when the param is True. Otherwise use "nothing" imputer 
        exclude_list: list
                List of features excluded from being scaled when action is True

        Returns
        ----------
        df: pd.DataFrame
                Scaled dataframe
        """
        self.fit(df, label=label, action=action, exclude_list=exclude_list, write_recoding_statement=write_recoding_statement)
        return self.transform(df, label=label, action=action, exclude_list=exclude_list)
