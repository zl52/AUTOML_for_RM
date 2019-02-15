import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   7. FEATURE SCALING   ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class UD_SCALER():
    """
    User-defined scaler which can normalize features
    """

    def __init__(self, recoding_dict=None, feat_dict=None, scaler_type='StandardScaler',
                 target=TARGET, silent=False):
        """
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : params scaler_type: choose scaler's type, including StandardScaler and MinMaxScaler
        : params target: list of default boolean targets
        : params silent: whether to print details of transformation process
        """
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict
        self.target = target
        self.scaler_type = scaler_type
        self.silent = silent

    def ud_fit(self, df, label=None, action=True, exclude_list=[]):
        """
        Fit the scaler using the input dataframe

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being scaled when action is True
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

                    if self.recoding_dict is not None and self.feat_dict is not None:
                        # recoding_statement = "######### Scale {i} using StandardScaler ########" \
                        #                      .format(i=ori_name)
                        recoding_statement = ""
                        recoding_statement += "\ndf['" + ori_name + "'] = (df['" + ori_name + "'] - " \
                                              + str(mean) + ") / " + str(std)
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

                    if self.recoding_dict is not None and self.feat_dict is not None:
                        # recoding_statement = "######### Scale {i} using MinMaxScaler ########" \
                        #                      .format(i=ori_name)
                        recoding_statement = ""
                        recoding_statement += "\ndf['" + ori_name + "'] = (df['" + ori_name + "'] - " \
                                              + str(minimum) + ") / " + str(maximum - minimum)
                        key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                        self.feat_dict.update({key: ori_name})
                        self.recoding_dict[key] += recoding_statement

                if not self.silent:
                    print("Use MinMaxScaler to scale the features")

            else:
                raise ValueError("scale_type must chosen among \'StandardScaler\' and \'MinMaxScaler\'")

        else:
            self.scalr == None

    def ud_transform(self, df, label=None, action=True, exclude_list=[]):
        """
        Scale features in the input dataframe

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being scaled when action is True
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in exclude_list + self.target + [label]]

        if action:
            if self.scalr is not None:
                try:
                    df[col] = self.scalr.transform(df[col])

                except:
                    raise ValueError("Scaling process incur error")
            else:
                pass

        else:
            pass

        return df

    def ud_fit_transform(self, df, label=None, action=True, exclude_list=[]):
        """
        Fit the scaler and scale features in the input dataframe

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being scaled when action is True
        """
        self.ud_fit(df, label=label, action=action, exclude_list=exclude_list)

        return self.ud_transform(df, label=label, action=action, exclude_list=exclude_list)
