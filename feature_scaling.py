import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
                 target=[],
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

    def fit(self, df, label=None, exclude_list=[], with_mean=True, with_std=True, feature_min=0, feature_max=1):
        """
        Fit the scaler by checking the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        exclude_list: list
                List of features excluded from being rescaled when action is True
        with_mean: boolean
                If True, center the data before scaling
        with_std: boolean
                If True, scale the data to unit variance (or equivalently, unit standard deviation)
        feature_min: float
                Desired minimum of transformed data
        feature_max: float
                Desired maximum of transformed data
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = set(list(df.select_dtypes(exclude=[object]))) - set(exclude_list + self.target + [label])

        if self.scaler_type == 'StandardScaler':
            self.scalr = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
            self.scalr.fit(df[col])
            if not self.silent:
                print("Apply StandardScaler to scale features")

        elif self.scaler_type == 'MinMaxScaler':
            self.scalr = MinMaxScaler(feature_range=(feature_min, feature_max), copy=True)
            self.scalr.fit(df[col])
            if not self.silent:
                print("Apply MinMaxScaler to scale features")
        else:
            raise ValueError("Variable \'scale_type\' must chosen between \'StandardScaler\' and \'MinMaxScaler\'")

    def transform(self, df, label=None, action=True, exclude_list=[], write_recoding_statement=True):
        """
        Rescale features in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        action: boolean
                Fit and transform scaler when the param is True. Otherwise use "nothing" imputer 
        exclude_list: list
                List of features excluded from being rescaled when action is True
        recoding_dict: dict
                Dictionary recording recoding statements for each feature

        Returns
        ----------
        df: pd.DataFrame
                Dataframe with continuous features rescaled
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = set(list(df.select_dtypes(exclude=[object]))) - set(exclude_list + self.target + [label])
        if action:
            try:
                df[list(col)] = self.scalr.transform(df[list(col)])

                if self.scaler_type == 'StandardScaler':
                    for i, ori_name in enumerate(col):
                        mean = self.scalr.mean_[i]
                        std = np.sqrt(self.scalr.var_[i]) if self.scalr.var_ is not None else None
                        if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                            recoding_statement = ""
                            if mean is not None and std not in [None, 0]:
                                recoding_statement += "\ndf['%s'] = (df['%s'] - %s) / %s"%(ori_name, ori_name, str(mean), str(std))
                            elif mean is None and std not in [None, 0]:
                                recoding_statement += "\ndf['%s'] = df['%s'] / %s"%(ori_name, ori_name, str(std))
                            elif mean is not None and std in [None, 0]:
                                recoding_statement += "\ndf['%s'] = df['%s'] - %s"%(ori_name, ori_name, str(mean))
                            else:
                                pass
                            key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                            self.feat_dict.update({key: ori_name})
                            self.recoding_dict[key] += recoding_statement

                if self.scaler_type == 'MinMaxScaler':
                    for i, ori_name in enumerate(col):
                        maximum = self.scalr.data_max_[i]
                        minimum = self.scalr.data_min_[i]
                        if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                            recoding_statement = ""
                            recoding_statement += "\ndf['%s'] = (df['%s'] - %s) / %s"%(ori_name, ori_name, str(minimum), str(maximum - minimum))
                            key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                            self.feat_dict.update({key: ori_name})
                            self.recoding_dict[key] += recoding_statement
            except Exception as e:
                print(e)
                raise Exception("Scaling process encounters an error")

        return df

    def fit_transform(self, df, label=None, action=True, exclude_list=[], write_recoding_statement=True,
                      with_mean=True, with_std=True, feature_min=0, feature_max=1):
        """
        Fit the scaler and rescale features in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        action: boolean
                Fit scaler when the param is True. Otherwise use "nothing" imputer 
        exclude_list: list
                List of features excluded from being rescaled when action is True
        recoding_dict: dict
                Dictionary recording recoding statements for each feature
        with_mean: boolean
                If True, center the data before scaling (when StandardScaler is applied)
        with_std: boolean
                If True, scale the data to unit variance (or equivalently, unit standard deviation) (when StandardScaler is applied)
        feature_min: float
                Desired minimum of transformed data (when MinMaxScaler is applied)
        feature_max: float
                Desired maximum of transformed data (when MinMaxScaler is applied)

        Returns
        ----------
        df: pd.DataFrame
                Dataframe with continuous features rescaled
        """
        self.fit(df, label=label, exclude_list=exclude_list, with_mean=with_mean, with_std=with_std,
                 feature_min=feature_min, feature_max=feature_max)
        return self.transform(df, label=label, action=action, exclude_list=exclude_list, write_recoding_statement=write_recoding_statement)
