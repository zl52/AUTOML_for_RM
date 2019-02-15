import pandas as pd;
import numpy as np

from sklearn.preprocessing import Imputer

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 9. FEATURE IMPUTATION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class UD_IMPUTER(Imputer):
    """
    User-defined imputer which imputes features seperately using defined imputation strategy
    imputation strategies for each feature are stored in imputr
    """

    def __init__(self, recoding_dict=None, feat_dict=None, use_mean_method=True, target=TARGET,
                 silent=False):
        """
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : params use_mean_method: whether to use mean value as imputing value
        : params target: list of default boolean targets
        : params silent: whether to print details of imputing process
        """
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict
        self.use_mean_method = use_mean_method
        self.target = target
        self.silent = silent

    def ud_fit(self, x, y=None, strategy='mean', prefix=''):
        """
        Fit imputer by checking x according to defined strategy for the feature

        : param x: feature to be imputed, array-like of shape (n_samples,)
        : param y: label
        : param strategy: imputation strategy ('mean', 'median', 'most_frequent' and 'nothing')
        : param prefix: prefix (eg. original column name)
        """
        if strategy == 'nothing':
            self.__dict__.update({prefix + '_imputer': 'nothing'})

        else:
            imputr = Imputer(strategy=strategy)
            imputr.fit(np.array(x).reshape(-1, 1))
            imputr.feat_name = prefix
            self.__dict__.update({prefix + '_imputer': imputr})

    def ud_transform(self, x, prefix='', write=False):
        """
        Impute x

        : param x: feature to be imputed, array-like of shape (n_samples,)
        : param write: whether to write recoding statement

        : return x: imputed feature
        """
        imputr = self.__dict__.get(prefix + '_imputer')
        recoding_statement = ''
        ori_name = prefix

        if imputr == 'nothing':
            x = np.array(x).reshape(-1, 1)

        else:
            x = imputr.transform(np.array(x).reshape(-1, 1))

            if write is True and self.recoding_dict is not None and self.feat_dict is not None:
                # recoding_statement = "######### Impute {i} using {j} strategy {k} ########" \
                #                         .format(i=ori_name, j=type(imputr), k=imputr.strategy)
                recoding_statement = ""
                recoding_statement += "\n" + "df.loc[:,'" + ori_name + "'] = df['" + ori_name \
                                      + "'].fillna(" + str(imputr.statistics_[0]) + ")"
                self.recoding_dict.update({ori_name: recoding_statement})

        return x

    def ud_fit_transform(self, x, y=None, strategy='mean', prefix=''):
        """
        Fit imputer by checking x and impute x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: imputed feature
        """
        self.ud_fit(x, y=y, strategy=strategy, prefix=prefix)

        return self.ud_transform(x, prefix=prefix)

    def ud_fit_all(self, df, label, action=True, mean_list=[], median_list=[], most_freq_list=[],
                   exclude_list=[]):
        """
        Fit imputer for all features in the dataframe using designated methods

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise use "nothing" imputer 
        : param mean_list: list of features to be imputed using mean value
        : param median_list: list of features to be imputed using median value
        : param most_freq_list: list of features to be imputed using most frequent value
        : param exclude_list: list of features kept the same
        """
        feat_not_meanimp = mean_list + median_list + most_freq_list
        df = df.apply(pd.to_numeric, errors='ignore').reset_index(drop=True)
        col_idx = df.select_dtypes(include=[float, int, 'int64']).isnull().sum() \
            [df.select_dtypes(include=[float, int, 'int64']).isnull().sum() != 0].index

        col = [i for i in col_idx if i not in self.target + [label] + feat_not_meanimp]

        if action:
            if self.use_mean_method:
                for i in col:
                    try:
                        self.ud_fit(df[i], y=None, strategy='mean', prefix=i)
                    except:
                        print(i)
                        raise ValueError("Failed to impute using mean method")

            for i in mean_list:
                try:
                    self.ud_fit(df[i], y=None, strategy='mean', prefix=i)

                except:
                    print(i)
                    raise ValueError("Failed to impute using mean method")

            for i in median_list:
                try:
                    self.ud_fit(df[i], y=None, strategy='median', prefix=i)

                except:
                    print(i)
                    raise ValueError("Failed to impute using median method")

            for i in most_freq_list:
                try:
                    self.ud_fit(df[i], y=None, strategy='most_frequent', prefix=i)

                except:
                    print(i)
                    raise ValueError("Failed to impute using most_frequent method")

            for i in exclude_list:
                self.ud_fit(df[i], y=None, strategy='nothing', prefix=i)

        else:
            for i in col:
                self.ud_fit(self, df[i], y=None, strategy='nothing', prefix=i)

    def ud_transform_all(self, df, label, action=False, exclude_list=[], write=False):
        """
        Impute features in the dataframe using saved methods

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise use "nothing" imputer 
        : param exclude_list: list of features kept the same
        : param write: whether to write recoding statement

        : return x: imputed dataframe
        """
        df = df.apply(pd.to_numeric, errors='ignore').reset_index(drop=True)
        col_idx = df.select_dtypes(include=[float, int, 'int64']).isnull().sum() \
            [df.select_dtypes(include=[float, int, 'int64']).isnull().sum() != 0].index
        col = [i for i in col_idx if i not in self.target + [label]]

        if action:
            for i in col:
                try:
                    if i not in exclude_list:
                        df[i] = self.ud_transform(self, df[i], prefix=i, write=write)

                    else:
                        pass

                except:
                    print("Failed to impute", i)

        else:
            pass

        return df

    def ud_fit_transform_all(self, df, label, action=False, mean_list=[], median_list=[],
                             most_freq_list=[], exclude_list=[], write=True):
        """
        Impute features in the dataframe using designated methods

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param mean_list: list of features to be imputed using mean value
        : param median_list: list of features to be imputed using median value
        : param most_freq_list: list of features to be imputed using most frequent value
        : param exclude_list: list of features kept the same
        : param write: whether to write recoding statement

        : return x: imputed dataframe
        """
        self.ud_fit_all(df, label, mean_list=mean_list, median_list=median_list,
                        most_freq_list=most_freq_list, exclude_list=exclude_list)

        return self.ud_transform_all(df, label, action=action, exclude_list=exclude_list, write=write)
