import pandas as pd
import numpy as np

from tools import *
from sample_exploration import desc_stat


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   6. FEATURE TRIMMING  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class UD_TRIMMER():
    """
    User-defined trimmer which can trim continuous variables in the dataframe
    """

    def __init__(self, recoding_dict=None, feat_dict=None, target=TARGET, silent=True):
        """
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : params target: list of default boolean targets
        : params silent: whether to print details of transformation process
        """
        self.target = target
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict
        self.silent = silent

    def ud_fit(self, df, label=None, exclude_list=[]):
        """
        Fit trimmer by checking continuous variables in the the input dataframe

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param exclude_list: list of features excluded from being used to fit the transformer
        """
        feats_list = [i for i in df.columns if i not in [label] + self.target + exclude_list]
        formater = "{0:.02f}".format
        stat = desc_stat(df[feats_list], target=[], ratio_pct=False, use_formater=False)
        stat['qr'] = stat.apply(lambda x: max((x['75%'] - x['25%']),
                                              (x['99%'] - x['75%']),
                                              (x['25%'] - x['1%']))
                                , axis=1)
        stat['upper_bound'] = stat.apply(lambda x: max(min((x['75%'] + 1.5 * x['qr']), x['max']),
                                                       x['99%'])
                                         , axis=1)
        stat['lower_bound'] = stat.apply(lambda x: min(max((x['25%'] - 1.5 * x['qr']), x['min']),
                                                       x['1%'])
                                         , axis=1)
        stat['upper_bound'] = stat.apply(lambda x: x['max'] if x['upper_bound'] == x['lower_bound'] \
            else x['upper_bound'], axis=1)
        stat['lower_bound'] = stat.apply(lambda x: x['min'] if x['upper_bound'] == x['lower_bound'] \
            else x['lower_bound'], axis=1)
        self.ub_dict = np.round(stat['upper_bound'], 5).to_dict()
        self.lb_dict = np.round(stat['lower_bound'], 5).to_dict()
        if not self.silent:
            print("Trim continuous features in the dataframe")

    def ud_transform(self, df, label=None, action=False, exclude_list=[], write=False):
        """
        Trim continuous variables in the the input dataframe

        : param df: the dataframe where continuous variables will be trimmed
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being trimmed
        : param write: whether to write recoding statement

        : return df: trimmed dataframe
        """
        if action:
            for ori_name in self.ub_dict.keys():
                if ori_name not in exclude_list:

                    try:
                        recoding_statement = ''
                        v1 = self.ub_dict.get(ori_name)
                        v2 = self.lb_dict.get(ori_name)
                        df[ori_name] = df[ori_name].map(lambda x: x if x < v1 else v1)
                        df[ori_name] = df[ori_name].map(lambda x: x if x > v2 else v2)

                        if write is True and self.recoding_dict is not None and self.feat_dict is not None:
                            # recoding_statement = "######### Trmming {i} ########".format(i=ori_name)
                            recoding_statement = ""
                            recoding_statement += "\ndf.loc[" + "df['" + ori_name + "'] > " + str(v1) \
                                                  + ", '" + ori_name + "'] = " + str(v1)
                            recoding_statement += "\ndf.loc[" + "df['" + ori_name + "'] < " + str(v2) \
                                                  + ", '" + ori_name + "'] = " + str(v2)
                            key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                            self.feat_dict.update({key: ori_name})
                            self.recoding_dict[key] += recoding_statement
                    except:
                        print("Can\'t trim feature", ori_name)
                else:
                    pass
        else:
            pass

        return df

    def ud_fit_transform(self, df, label=None, action=False, exclude_list=[], write=True):
        """
        Fit trimmer and trim continuous variables in the the input dataframe

        : param df: dataframe where continuous variables are going to be trimmed
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being trimmed
        : param write: whether to write recoding statement

        : return df: trimmed dataframe
        """
        if action:
            self.ud_fit(df, label=label, exclude_list=exclude_list)
            df = self.ud_transform(df, label=label, action=action, exclude_list=exclude_list, write=write)

        return df
