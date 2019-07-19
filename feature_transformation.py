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
                 target=TARGET,
                 drop_features=True,
                 silent=False,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        target: list
                List of default boolean targets
        drop_features: boolean
                Drop original feature
        silent: boolean
                Restrict the print of transformation process
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
                Feature to be transformed, pd.series of shape (n_samples,)
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
        return x

    def sqrt_fit_transform(self, x, y=None):
        """
        Transform feature x.

        Parameters
        ----------
        x: pd.Series
                Feature to be transformed, pd.series of shape (n_samples,)
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
        return x

    def log_fit_transform(self, x, y=None):
        """
        Transform feature x.

        Parameters
        ----------
        x: pd.Series
                Feature to be transformed, pd.series of shape (n_samples,)
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
        return x

    def best_single_fit_transform(self, x, y, fit_action=True, write_recoding_statement=True):
        """
        Find the best transformation for the feature.

        Parameters
        ----------
        x: pd.Series
                Feature to be transformed, pd.series of shape (n_samples,)
        y: pd.Series
                Dependent variable
        fit_action: boolean
                Take action when fit_action is True. Otherwise return origin values
        write_recoding_statement: boolean
                Write recoding statement

        Returns
        ----------
        x: pd.series of shape (n_samples,)
                Transformed feature
        """
        if y is None:
            raise Exception('Transformation selection needs valid y label.')
        ori_name = x.name
        recoding_statement = ""
        if fit_action:
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
                recoding_statement += "\ndf['%s'] = df['%s'].apply(np.square)" %(x.name, ori_name)
            elif np.argmin(f_stat) == 2:
                x = x_sqrt
                self.__dict__.update({'%s_transformer'%ori_name: 'sqrt'})
                recoding_statement += "\ndf['%s'] = df['%s'].apply(lambda x: np.sqrt(max(x, 0)))" %(x.name, ori_name)

            elif np.argmin(f_stat) == 3:
                x = x_log
                self.__dict__.update({'%s_transformer'%ori_name: 'log'})
                recoding_statement += "\ndf['%s'] = df['%s'].apply(lambda x: np.log(max(x, 0.00001)))" %(x.name, ori_name)

            if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                self.recoding_dict[ori_name] += recoding_statement
                self.feat_dict.update({ori_name: x.name})
        else:
            self.__dict__.update({'%s_transformer' %ori_name: 'without'})
        if not self.silent:
            print('Best transformation for %s is %s transformation' %(ori_name, 
                                                    self.__dict__.get('%s_transformer' %ori_name)))
        return x

    def fit(self, df, label, fit_action=True, exclude_list=[], write_recoding_statement=True):
        """
        Find the best transformation for each feature in the the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        fit_action: boolean
                Fit transformer when the param is True
        exclude_list: list
                List of features excluded from being fitted
        write_recoding_statement: boolean
                Write recoding statement
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in self.target + [label]]
        if fit_action:
            for i in set(col) - set(exclude_list):
                self.best_single_fit_transform(x=df[i], y=df[label], fit_action=True, 
                                               write_recoding_statement=write_recoding_statement)

            for i in set(col) & set(exclude_list):
                self.best_single_fit_transform(x=df[i], y=df[label], fit_action=False,
                                               write_recoding_statement=write_recoding_statement)

        else:
            for i in col:
                self.best_single_fit_transform(x=df[i], y=df[label], fit_action=False,
                                               write_recoding_statement=write_recoding_statement)

    def transform(self, df, label, transform_action=False, exclude_list=[]):
        """
        Find the best transformation for each feature in the the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        transform_action: boolean
                Take action when the param is True. Otherwise return origin values
        exclude_list: list
                List of features excluded from being transformed

        Returns
        ----------
        df: pd.DataFrame
                Transformed dataframe
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in self.target + [label]]

        if transform_action:
            for i in set(col) - set(exclude_list):
                try:
                    transformr = self.__dict__.get(i + '_transformer')
                    if transformr == 'square':
                        df['%s_square'%i] = self.square_fit_transform(df[i], y=None)
                    elif transformr == 'sqrt':
                        df['%s_sqrt'%i] = self.sqrt_fit_transform(df[i], y=None)
                    elif transformr == 'log':
                        df['%s_log'%i] = self.log_fit_transform(df[i], y=None)
                    elif transformr == 'without':
                        pass
                    if self.drop_features and transformr != 'without':
                        del df[i]
                except:
                    print('%s\'s transformer is not defined'%i)

        else:
            pass

        return df

    def fit_transform(self, df, label, fit_action=True, transform_action=True, exclude_list=[],
                      write_recoding_statement=True):
        """
        Find the best transformation for each feature in the the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        fit_action: boolean
                Fit transformer when the param is True
        transform_action: boolean
                Take action when the param is True. Otherwise return origin values
        exclude_list: list
                List of features excluded from being fitted and transformed
        write_recoding_statement: boolean
                Write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Transformed dataframe
        """
        self.fit(df, label=label, fit_action=fit_action, exclude_list=exclude_list,
                 write_recoding_statement=write_recoding_statement)
        return self.transform(df, label=label, transform_action=transform_action,
                                     exclude_list=exclude_list)