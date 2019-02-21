import pandas as pd
import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from tools import *
from recoding_statement import GENERATE_OUTPUT_FILES


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################5. FEATURE TRANSFORMATION#####################################
######################################                        ######################################
####################################################################################################
####################################################################################################


class UD_TRANSFORMER():
    """
    TODO: MINMAX SCALER
    User-defined basic transformer which can log-transform, sqrt-transform or square-transform features
    """

    def __init__(self, recoding_dict, feat_dict, target=TARGET, drop_ori_feat=True, silent=False):
        """
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : params target: list of default boolean targets
        : param drop_ori_feat: whether to drop original feature
        : params silent: whether to print details of transformation process
        """
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict
        self.target = target
        self.drop_ori_feat = drop_ori_feat
        self.silent = silent

    def ud_square_fit_transform(self, x, y=None):
        """
        Transform x

        : param x: feature to be transformed, pd.series of shape (n_samples,)
        : param y: label

        : return x: transformed feature, pd.series of shape (n_samples,)
        """
        x = x.apply(np.square)
        self.__dict__.update({x.name + '_transformer': 'square'})
        x.name = x.name + '_square'

        return x

    def ud_sqrt_fit_transform(self, x, y=None):
        """
        Transform x

        : param x: feature to be transformed, pd.series of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: transformed feature, pd.series of shape (n_samples,)
        """
        x = x.apply(lambda x: np.sqrt(max(x, 0)))
        self.__dict__.update({x.name + '_transformer': 'sqrt'})
        x.name = x.name + '_sqrt'

        return x

    def ud_log_fit_transform(self, x, y=None):
        """
        Transform x

        : param x: feature to be transformed, pd.series of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: transformed feature, pd.series of shape (n_samples,)
        """
        x = x.apply(lambda x: np.log(max(x, 0.00001)))
        self.__dict__.update({x.name + '_transformer': 'log'})
        x.name = x.name + '_log'

        return x

    def ud_best_transformation_fit_transform(self, x, y, action=True):
        """
        Find the best transformation for x

        : param x: feature to be transformed, pd.series of shape (n_samples,)
        : param y: label
        : param action: only take action when action is True. Otherwise return origin values

        : return x: transformed feature, pd.series of shape (n_samples,)
        """
        if y is None:
            raise Exception('Transformation selection needs valid y label.')

        ori_name = x.name
        recoding_statement = ""

        if action:
            x_square = self.ud_square_fit_transform(x, y=None)
            x_sqrt = self.ud_sqrt_fit_transform(x, y=None)
            x_log = self.ud_log_fit_transform(x, y=None)
            df_trans = pd.DataFrame({'x_ori': x, 'x_square': x_square,
                                     'x_sqrt': x_sqrt, 'x_log': x_log, 'label': y}).dropna()
            f_stat, _ = f_regression(df_trans.drop('label', axis=1), df_trans['label'])

            if np.argmin(f_stat) == 0:
                self.__dict__.update({ori_name + '_transformer': 'without'})

            elif np.argmin(f_stat) == 1:
                x = x_square
                self.__dict__.update({ori_name + '_transformer': 'square'})
                recoding_statement += "\ndf['" + x.name + "'] = df['" \
                                      + ori_name + "'].apply(np.square)"

            elif np.argmin(f_stat) == 2:
                x = x_sqrt
                self.__dict__.update({ori_name + '_transformer': 'sqrt'})
                recoding_statement += "\ndf['" + x.name + "'] = df['" \
                                      + ori_name + "'].apply(lambda x: np.sqrt(max(x, 0)))"

            elif np.argmin(f_stat) == 3:
                x = x_log
                self.__dict__.update({ori_name + '_transformer': 'log'})
                recoding_statement += "\ndf['" + x.name + "'] = df['" \
                                      + ori_name + "'].apply(lambda x: np.log(max(x, 0.00001)))"

        else:
            self.__dict__.update({ori_name + '_transformer': 'without'})

        self.recoding_dict[ori_name] += recoding_statement
        self.feat_dict.update({ori_name: x.name})

        if not self.silent:
            print('Best transformation for {c} is {trans} transformation' \
                  .format(c=ori_name, trans=self.__dict__.get(ori_name + '_transformer')))

        return x

    def ud_fit_all(self, df, label, action=True, exclude_list=[]):
        """
        Find the best transformation for each feature in the the input dataframe

        : param df: the input dataframe
        : param label: label will be used in the modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being transformed when action is True
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in self.target + [label]]

        if action:
            for i in col:
                if i not in exclude_list:
                    self.ud_best_transformation_fit_transform(x=df[i], y=df[label], action=True)

                else:
                    self.ud_best_transformation_fit_transform(x=df[i], y=df[label], action=False)

        else:
            for i in col:
                self.ud_best_transformation_fit_transform(x=df[i], y=df[label], action=False)

    def ud_transform_all(self, df, label, action=False, exclude_list=[]):
        """
        Find the best transformation for each feature in the the input dataframe

        : param df: the input dataframe
        : param label: target value will be used in modeling process
        : param action: only take action when the param is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being transformed

        : return x: transformed dataframe
        """
        df = df.apply(pd.to_numeric, errors='ignore')
        col = [i for i in df.select_dtypes(include=[float, int, 'int64']).columns \
               if i not in self.target + [label]]

        if action:
            for i in col:
                if i not in exclude_list:
                    try:
                        transformr = self.__dict__.get(i + '_transformer')

                        if transformr == 'square':
                            df[i + '_square'] = self.ud_square_fit_transform(df[i], y=None)

                            if self.drop_ori_feat:
                                del df[i]

                        elif transformr == 'sqrt':
                            df[i + '_sqrt'] = self.ud_sqrt_fit_transform(df[i], y=None)

                            if self.drop_ori_feat:
                                del df[i]

                        elif transformr == 'log':
                            df[i + '_log'] = self.ud_log_fit_transform(df[i], y=None)

                            if self.drop_ori_feat:
                                del df[i]

                        elif transformr == 'without':
                            pass

                    except:
                        print('{i}\'s transformer is not defined'.format(i=i))

                else:
                    pass

        else:
            pass

        return df

    def ud_fit_transform_all(self, df, label, action=False, exclude_list=[]):
        """
        Find the best transformation for each feature in the the input dataframe

        : param df: the input dataframe
        : param label: target value will be used in the modeling process
        : param action: only take action when action is True. Otherwise return origin values
        : param exclude_list: list of features excluded from being transformed

        : return x: transformed dataframe
        """
        if action:
            self.ud_fit_all(df, label=label, action=action, exclude_list=exclude_list)

            return self.ud_transform_all(df, label=label, action=action,
                                         exclude_list=exclude_list)

        else:
            return df