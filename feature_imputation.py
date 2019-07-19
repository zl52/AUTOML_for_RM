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
    User-defined imputer can impute features seperately using assigned imputation method.
    Imputation strategies for each feature are stored in imputr.
    """

    def __init__(self,
                 use_mean_method=True,
                 target=TARGET,
                 impute_object='<NA>*',
                 silent=False,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        use_mean_method: boolean
                Use mean value as imputing value
        target: list
                List of default boolean targets
        impute_object: str
                Use to impute features of type obejct
        silent: boolean
                Restrict the print of imputation process
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

    def single_fit(self, x, y=None, method='mean', prefix='', write_recoding_statement=True):
        """
        Fit imputer by checking feature x based on method assigned.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Feature to be imputed
        y: array-like of shape (n_samples,)
                Dependent variable
        method: str
                Imputation method ('mean', 'median', 'most_frequent' and 'nothing')
        prefix: str
                Prefix (eg. original column name)
        write_recoding_statement: boolean
                Write recoding statement
        """
        if method == 'nothing':
            self.__dict__.update({'%s_imputer'%prefix: 'nothing'})
        else:
            imputr = Imputer(strategy=method)
            imputr.fit(np.array(x).reshape(-1, 1))
            imputr.feat_name = prefix
            self.__dict__.update({'%s_imputer'%prefix: imputr})
            if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                ori_name = prefix
                recoding_statement = ""
                recoding_statement += "\ndf.loc[:,'%s'] = df['%s'].fillna(%s)"%(ori_name, ori_name, str(imputr.statistics_[0]))
                key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                self.feat_dict.update({key: ori_name})
                self.recoding_dict[key] += recoding_statement

    def single_transform(self, x, prefix=''):
        """
        Impute feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Feature to be imputed, 

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Imputed feature
        """
        imputr = self.__dict__.get('%s_imputer'%prefix)
        if imputr == 'nothing':
            x = np.array(x).reshape(-1, 1)
        else:
            x = imputr.transform(np.array(x).reshape(-1, 1))
        return x

    def single_fit_transform(self, x, y=None, method='mean', prefix=''):
        """
        Fit imputer by checking feature x based on method assigned and impute the feature.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Feature to be imputed
        y: array-like of shape (n_samples,)
                Dependent variable
        method: str
                Imputation method ('mean', 'median', 'most_frequent' and 'nothing')
        prefix: str
                Prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Imputed feature
        """
        self.single_fit(x, y=y, method=method, prefix=prefix)
        return self.single_transform(x, prefix=prefix)

    def fit(self, df, label, action=True, mean_list=[], median_list=[], most_freq_list=[],
                   exclude_list=[], write_recoding_statement=True):
        """
        Fit imputer for all features in the dataframe using assigned methods.

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
                List of features kept the same
        write_recoding_statement: boolean
                Write recoding statement
        """
        df = df.apply(pd.to_numeric, errors='ignore').reset_index(drop=True)
        not_auto_mean_list = median_list + most_freq_list + mean_list
        feats_has_null = [i for i in df.isnull().sum()[df.isnull().sum() != 0].index.tolist() if i not in exclude_list]
        auto_mean_list = [i for i in feats_has_null if i in df.select_dtypes(include=[float, int, 'int64']).columns]
        auto_mean_list = [i for i in auto_mean_list if i not in self.target + [label] + not_auto_mean_list]

        if action:
            if self.use_mean_method:
                for i in auto_mean_list:
                    try:
                        self.single_fit(df[i], y=None, method='mean', prefix=i,
                                        write_recoding_statement=write_recoding_statement)
                    except:
                        print("Failed to impute %s using mean method"%i)
            for i in mean_list:
                try:
                    self.single_fit(df[i], y=None, method='mean', prefix=i,
                                    write_recoding_statement=write_recoding_statement)
                except:
                   print("Failed to impute %s using mean method"%i)
            for i in median_list:
                try:
                    self.single_fit(df[i], y=None, method='median', prefix=i,
                                    write_recoding_statement=write_recoding_statement)
                except:
                    print("Failed to impute %s using median method"%i)
            for i in most_freq_list:
                try:
                    self.single_fit(df[i], y=None, method='most_frequent', prefix=i,
                                    write_recoding_statement=write_recoding_statement)
                except:
                    print("Failed to impute %s using most_frequent method"%i)

            for i in set(feats_has_null) - set(not_auto_mean_list) - set(auto_mean_list):
                self.single_fit(df[i], y=None, method='nothing', prefix=i,
                                write_recoding_statement=write_recoding_statement)
        else:
            for i in feats_has_null:
                self.single_fit(df[i], y=None, method='nothing', prefix=i,
                                write_recoding_statement=write_recoding_statement)

    def transform(self, df, label, action=False, exclude_list=[]):
        """
        Impute features in the dataframe using saved methods.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        label: str
                Label will be used in the modeling process
        action: 
                Take action when the param is True. Otherwise use "nothing" imputer 
        exclude_list: list
                List of features kept the same

        Returns
        ----------
        : return x: imputed dataframe
        """
        df = df.apply(pd.to_numeric, errors='ignore').reset_index(drop=True)
        feats_has_null = df.isnull().sum()[df.isnull().sum() != 0].index.tolist()
        cov_feats_has_null = [i for i in feats_has_null if i not in exclude_list + df.select_dtypes(include=[object]).columns.tolist()]
        cav_feats_has_null = list(set(feats_has_null) - set(cov_feats_has_null))
        if action:
            if self.impute_object is not None:
                df[cav_feats_has_null] = df[cav_feats_has_null].fillna(self.impute_object)
            for i in cov_feats_has_null:
                try:
                    df[i] = self.single_transform(df[i], prefix=i)
                except:
                    print("Failed to impute %s"%i)
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
                List of features kept the same
        write_recoding_statement: boolean
                Write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Imputed dataframe
        """
        self.fit(df, label, action=action, mean_list=mean_list, median_list=median_list, most_freq_list=most_freq_list,
                 exclude_list=exclude_list, write_recoding_statement=write_recoding_statement)
        return self.transform(df, label, action=action, exclude_list=exclude_list)
