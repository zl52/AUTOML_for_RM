import pandas as pd;
import numpy as np
from sklearn.preprocessing import Imputer

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 8. FEATURE IMPUTATION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class FeatureImputer(Imputer):
    """
    User-defined imputer can impute features seperately using assigned imputation strategy.
    Imputation strategies for each feature are stored in imputr.
    """

    def __init__(self,
                 use_mean_method=True,
                 target=[],
                 impute_object='<NA>*',
                 silent=False,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        use_mean_method: boolean
                If True, use mean value to impute features
        target: list
                List of default boolean targets
        impute_object: str
                Used to impute features of type obejct
        silent: boolean
                If True, restrict the print of imputation process
        recoding_dict: dict
                Dictionary recording recoding statements for each feature
        feat_dict: dict
                Dictionary recording changes of feature names
        """
        self.use_mean_method = use_mean_method
        self.target = target
        self.impute_object = impute_object
        self.silent = silent
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict

    def single_fit(self, x, y=None, method='mean'):
        """
        Fit imputer by checking feature x based on strategy assigned.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Feature to be imputed
        y: array-like of shape (n_samples,)
                Dependent variable
        method: str
                Imputation strategy ('mean', 'median', 'most_frequent' and 'nothing')
        """
        ori_name = x.name
        if method == 'nothing':
            self.__dict__.update({'%s_imputer'%ori_name: 'nothing'})
        else:
            imputr = Imputer(strategy=method)
            imputr.fit(np.array(x).reshape(-1, 1))
            imputr.feat_name = ori_name
            self.__dict__.update({'%s_imputer'%ori_name: imputr})

    def single_transform(self, x, y=None, write_recoding_statement=True):
        """
        Impute feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Feature to be imputed, 
        y: array-like of shape (n_samples,)
                Dependent variable
        write_recoding_statement: boolean
                If True, write recoding statement

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Imputed feature
        """
        ori_name = x.name
        imputr = self.__dict__.get('%s_imputer'%ori_name)
        if imputr == 'nothing':
            x = np.array(x).reshape(-1, 1)
        else:
            x = imputr.transform(np.array(x).reshape(-1, 1))
            if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                recoding_statement = ""
                recoding_statement += "\ndf.loc[:,'%s'] = df['%s'].fillna(%s)"%(ori_name, ori_name, str(imputr.statistics_[0]))
                key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                self.feat_dict.update({key: ori_name})
                self.recoding_dict[key] += recoding_statement
        return x

    def fit(self, df, label, mean_list=[], median_list=[], most_freq_list=[], exclude_list=[], write_recoding_statement=True):
        """
        Fit imputer for all features in the dataframe using assigned method.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        mean_list: list
                List of features to be imputed using mean value
        median_list: list
                List of features to be imputed using median value
        most_freq_list: list
                List of features to be imputed using most frequent value
        exclude_list: list
                List of features excluded from being imputed
        """
        df = df.apply(pd.to_numeric, errors='ignore').reset_index(drop=True)
        non_mean_list = median_list + most_freq_list
        null_set = set(df.isnull().sum()[df.isnull().sum() != 0].index)
        auto_mean_list = list(set(null_set) & set(df.select_dtypes(exclude=[object])) - \
                         set(self.target + [label] + non_mean_list + exclude_list))
        nothing_list = list(set(null_set) - set(self.target + [label] + non_mean_list + exclude_list + auto_mean_list))
        if not self.use_mean_method:
            auto_mean_list = mean_list
        for i in auto_mean_list:
            self.single_fit(df[i], y=None, method='mean')
            print("Apply \'mean\' method to impute %s"%i)
        for i in median_list:
            self.single_fit(df[i], y=None, method='median')
            print("Apply \'median\' method to impute %s"%i)
        for i in most_freq_list:
            self.single_fit(df[i], y=None, method='most_frequent')
            print("Apply \'most_frequent\' method to impute %s"%i)
        for i in nothing_list:
            self.single_fit(df[i], y=None, method='nothing')
            print("Apply \'nothing\' method to impute %s"%i)

    def transform(self, df, label, action=False, exclude_list=[], write_recoding_statement=True):
        """
        Impute features in the dataframe with fitted imputer.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        action: 
                Take action when the param is True.
        exclude_list: list
                List of features excluded from being imputed
        write_recoding_statement: boolean
                If True, write recoding statement

        Returns
        ----------
        : return x: imputed dataframe
        """
        df = df.apply(pd.to_numeric, errors='ignore').reset_index(drop=True)
        null_set = set(df.isnull().sum()[df.isnull().sum() != 0].index)
        cov_null_set = set(null_set) - set(df.select_dtypes(include=[object])) - set(self.target + [label] + exclude_list)
        cav_null_set = set(null_set) - set(df.select_dtypes(exclude=[object])) - set(self.target + [label] + exclude_list)
        if action:
            if self.impute_object is not None:
                df[list(cav_null_set)] = df[list(cav_null_set)].fillna(self.impute_object)
            for i in cov_null_set:
                try:
                    df[i] = self.single_transform(df[i], write_recoding_statement)
                except Exception as e:
                    print("Failed to impute %s"%i)
                    print(e)
        return df

    def fit_transform(self, df, label, action=False, mean_list=[], median_list=[],
                      most_freq_list=[], exclude_list=[], write_recoding_statement=True):
        """
        Fit imputers for each feature in the dataframe using assigned methods and impute features.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                label will be used in the modeling process
        action: boolean
                Fit imputer when the param is True. Otherwise use "nothing" imputer 
        mean_list: list
                List of features to be imputed using mean value
        median_list: list
                List of features to be imputed using median value
        most_freq_list: list
                List of features to be imputed using most frequent value
        exclude_list: list
                List of features excluded from being imputed
        write_recoding_statement: boolean
                If True, write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Dataframe with missing value imputed
        """
        self.fit(df, label, mean_list=mean_list, median_list=median_list, most_freq_list=most_freq_list, exclude_list=exclude_list)
        return self.transform(df, label, action=action, exclude_list=exclude_list, write_recoding_statement=write_recoding_statement)