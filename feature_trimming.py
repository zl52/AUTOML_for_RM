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


class FeatureTrimmer():
    """
    Feature trimmer can trim continuous variables in the dataframe.
    """

    def __init__(self,
                 target=TARGET,
                 silent=True,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        target: list
                List of default boolean targets
        silent: boolean
                Restrict the print of transformation process
        recoding_dict: dict
                Dictionary recording recoding statements for each feature.
        feat_dict: dict
                Dictionary recording changes of feature names
        """
        self.target = target
        self.silent = silent
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict

    def fit(self, df, label=None, exclude_list=[], write_recoding_statement=True):
        """
        Fit trimmer by checking continuous variables in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        exclude_list: list
                List of features excluded from being used to fit the transformer
        write_recoding_statement: boolean
                Write recoding statement
        """
        feats_list = [i for i in df.select_dtypes(exclude=[object]).columns if i not in [label] + \
                      self.target + exclude_list]
        formater = "{0:.02f}".format
        with HiddenPrints():
            stat = desc_stat(df[feats_list], target=[], is_ratio_pct=False, use_formater=False)
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
        for ori_name in self.ub_dict.keys():
            v1 = self.ub_dict.get(ori_name)
            v2 = self.lb_dict.get(ori_name)
            if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                recoding_statement = ''
                recoding_statement += "\ndf.loc[df['%s'] > %f, '%s'] = %f" % (ori_name, v1, ori_name, v1)
                recoding_statement += "\ndf.loc[df['%s'] < %f, '%s'] = %f" % (ori_name, v2, ori_name, v2)
                key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                self.feat_dict.update({key: ori_name})
                self.recoding_dict[key] += recoding_statement

    def transform(self, df, label=None, action=False, exclude_list=[]):
        """
        Trim continuous variables in the the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                Dataframe where continuous variables are being trimmed
        label: boolean
                Label will be used in the modeling process
        action: boolean
                only take action when the param is True. Otherwise return origin values
        exclude_list: list
                List of features excluded from being trimmed
        write_recoding_statement: boolean
                Write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Trimmed dataframe
        """
        if action:
            for ori_name in set(self.ub_dict.keys()) - set(exclude_list):
                try:
                    v1 = self.ub_dict.get(ori_name)
                    v2 = self.lb_dict.get(ori_name)
                    df[ori_name] = df[ori_name].map(lambda x: v1 if x > v1 else x)
                    df[ori_name] = df[ori_name].map(lambda x: v2 if x < v2 else x)
                except:
                    print("Can\'t trim feature %s" %ori_name)
        else:
            pass
        return df

    def fit_transform(self, df, label=None, action=True, exclude_list=[], write_recoding_statement=True):
        """
        Fit trimmer and trim continuous variables in the the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                Dataframe where continuous variables are being trimmed
        label: boolean
                Label will be used in the modeling process
        action: boolean
                Take action when the param is True. Otherwise return origin values
        exclude_list: list
                List of features excluded from being trimmed
        write_recoding_statement: boolean
                Write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Trimmed dataframe
        """
        self.fit(df, label=label, exclude_list=exclude_list, write_recoding_statement=write_recoding_statement)
        if action:
            df = self.transform(df, label=label, action=action, exclude_list=exclude_list)
        return df
