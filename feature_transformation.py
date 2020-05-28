import pandas as pd
import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################5. FEATURE TRANSFORMATION#####################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class FeatureTransformer():
    """
    Feature transformer can log-transform, sqrt-transform or square-transform continuous variables.
    """

    def __init__(self,
                 target=[],
                 drop_features=True,
                 silent=False,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        target: list
                List of default targets
        drop_features: boolean
                If True, drop original feature after transformation
        silent: boolean
                If True, restrict the print of transformation process
        recoding_dict: dict
                Dictionary recording recoding statements for each feature
        feat_dict: dict
                Dictionary recording changes of feature names
        """
        self.target = target
        self.drop_features = drop_features
        self.silent = silent
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict

    def square_fit_transform(self, x, y=None):
        """
        Transform feature x.

        Parameters
        ----------
        x: pd.Series
                Independent variable to be transformed, pd.series of shape (n_samples,)
        y: pd.Series
                Dependent variable

        Returns
        ----------
        x: pd.series of shape (n_samples,)
                Transformed feature
        """
        x = x.apply(np.square)
        self.__dict__.update({'%s_transformer'%x.name: 'square'})
        x.name = x.name + '_square'
        return np.round(x, 6)

    def sqrt_fit_transform(self, x, y=None):
        """
        Transform feature x.

        Parameters
        ----------
        x: pd.Series
                Independent variable to be transformed, pd.series of shape (n_samples,)
        y: pd.Series
                Dependent variable

        Returns
        ----------
        x: pd.series of shape (n_samples,)
                Transformed feature
        """
        x = x.apply(lambda x: np.sqrt(max(x, 0)))
        self.__dict__.update({'%s_transformer'%x.name: 'sqrt'})
        x.name = x.name + '_sqrt'
        return np.round(x, 6)

    def log_fit_transform(self, x, y=None):
        """
        Transform feature x.

        Parameters
        ----------
        x: pd.Series
                Independent variable to be transformed, pd.series of shape (n_samples,)
        y: pd.Series
                Dependent variable

        Returns
        ----------
        x: pd.series of shape (n_samples,)
                Transformed feature
        """
        x = x.apply(lambda x: np.log(max(x, 0.00001)))
        self.__dict__.update({'%s_transformer'%x.name: 'log'})
        x.name = x.name + '_log'
        return np.round(x, 6)

    def fit(self, df, label, exclude_list=[]):
        """
        Find the best transformation for each feature in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        exclude_list: list
                List of features excluded
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = set(list(df.select_dtypes(exclude=[object]))) - set(self.target + [label])
        for ori_name in col:
            if ori_name not in exclude_list:
                x = df[ori_name]
                y = df[label]
                x_square = self.square_fit_transform(x, y=None)
                x_sqrt = self.sqrt_fit_transform(x, y=None)
                x_log = self.log_fit_transform(x, y=None)
                df_trans = pd.DataFrame({'x_ori': x, 'x_square': x_square,
                                         'x_sqrt': x_sqrt, 'x_log': x_log, 'label': y}).dropna()
                f_stat, _ = f_regression(df_trans.drop('label', axis=1), df_trans['label'])
                if np.argmin(f_stat) == 0:
                    self.__dict__.update({'%s_transformer'%ori_name: 'without'})
                elif np.argmin(f_stat) == 1:
                    x = x_square
                    self.__dict__.update({'%s_transformer'%ori_name: 'square'})
                elif np.argmin(f_stat) == 2:
                    x = x_sqrt
                    self.__dict__.update({'%s_transformer'%ori_name: 'sqrt'})
                elif np.argmin(f_stat) == 3:
                    x = x_log
                    self.__dict__.update({'%s_transformer'%ori_name: 'log'})
                if not self.silent:
                    print('Best transformation for %s is %s transformation' %(ori_name, 
                          self.__dict__.get('%s_transformer' %ori_name)))
            else:
                self.__dict__.update({'%s_transformer'%ori_name: 'without'})

    def transform(self, df, label, action=True, exclude_list=[], write_recoding_statement=True):
        """
        Transform features in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        action: boolean
                If True, take action. Otherwise return origin values
        exclude_list: list
                List of features excluded from being transformed
        write_recoding_statement: boolean
                If True, write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Dataframe with some features transformed
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = set(list(df.select_dtypes(exclude=[object]))) - set(self.target + [label])
        if action:
            if not self.silent:
                print("Transform continuous features in the dataframe")
            for ori_name in col - set(exclude_list):
                try:
                    transformr = self.__dict__.get('%s_transformer'%ori_name)
                    new_name = ori_name
                    recoding_statement = ''
                    if transformr == 'square':
                        new_name = '%s_square'%ori_name
                        df[new_name] = self.square_fit_transform(x=df[ori_name], y=label)
                        recoding_statement += "\ndf['%s'] = np.round(df['%s'].apply(np.square), 6)" %(new_name, ori_name)
                    elif transformr == 'sqrt':
                        new_name = '%s_sqrt'%ori_name
                        df[new_name] = self.sqrt_fit_transform(x=df[ori_name], y=label)
                        recoding_statement += "\ndf['%s'] = np.round(df['%s'].apply(lambda x: np.sqrt(max(x, 0))), 6)" %(new_name, ori_name)
                    elif transformr == 'log':
                        new_name = '%s_log'%ori_name
                        df[new_name] = self.log_fit_transform(x=df[ori_name], y=label)
                        recoding_statement += "\ndf['%s'] = np.round(df['%s'].apply(lambda x: np.log(max(x, 0.00001))), 6)" %(new_name, ori_name)
                    if self.drop_features and transformr != 'without':
                        del df[ori_name]
                    if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                        self.recoding_dict[ori_name] += recoding_statement
                        self.feat_dict.update({ori_name: new_name})
                except Exception as e:
                    print('Failed to transform %s\''%ori_name)
                    print(e)

        return df

    def fit_transform(self, df, label, action=True, exclude_list=[], write_recoding_statement=True):
        """
        Find the best transformation for each feature in the input dataframe and Transform features in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        action: boolean
                If True, take action. Otherwise return origin values
        exclude_list: list
                List of features excluded from being transformed
        write_recoding_statement: boolean
                If True, write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Dataframe with some features transformed
        """
        self.fit(df, label=label, exclude_list=exclude_list)
        return self.transform(df, label=label, action=action, exclude_list=exclude_list,
                              write_recoding_statement=write_recoding_statement)