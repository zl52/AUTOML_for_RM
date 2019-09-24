import numpy as np
import pandas as pd

from tools import *
from feature_encoding import WoeEncoder
from data_cleaning import desc_stat
from feature_transformation import FeatureTransformer
from feature_trimming import FeatureTrimmer
from feature_scaling import FeatureScaler
from feature_encoding import FeatureEncoder


class GenerateOutputFiles:

    def __init__(self,
                 df_ori,
                 label,
                 target=TARGET,
                 exclude_list=[]):
        """
        Parameters
        ----------
        df_ori: pd.DataFrame
                Original dataframe before feature engineering,
                including transformation, scaling, encoding and imputation.
        label: str
                Label will be used in the modeling process
        target: list
                List of default boolean targets
        exclude_list: list
                List of features excluded from being recoded
        feats: list
                Features used to train the model
        recoding_dict: dict
                Dictionary recording recoding statements for each feature
        feat_dict: dict
                Dictionary recording changes of feature names
        recoding_statement: str
                Recoding statement to output
        """
        self.df_ori = df_ori
        self.label = label
        self.target = target
        self.exclude_list = exclude_list
        self.feats = set(df_ori.columns) - set(self.exclude_list + [self.label] + self.target)
        self.recoding_dict = dict(zip(self.feats, [''] * len(self.feats)))
        self.feat_dict = dict(zip(self.feats, self.feats))
        self.recoding_statement = ''

    def generate_lr_recoding_dict(self, file_name, useTransformr=True, useTrimmr=True, 
                                  useScalr=True, useWoeEncoder=True, useImputr=True,
                                  cut_method='qcut', scaler_type='StandardScaler',
                                  encoding="utf-8", silent=True):
        df = self.df_ori.copy()
        with HiddenPrints():
            basic_transformr = FeatureTransformer(target=self.target, drop_features=True, silent=silent,
                                                  recoding_dict=self.recoding_dict, feat_dict=self.feat_dict)
            df = basic_transformr.fit_transform(df, label=self.label, fit_action=useTransformr,
                                                transform_action=useTransformr,exclude_list=self.exclude_list)

            trimmr = FeatureTrimmer(target=self.target, silent=silent, recoding_dict=self.recoding_dict,
                                    feat_dict=self.feat_dict)
            df = trimmr.fit_transform(df, action=useTrimmr, exclude_list=self.exclude_list)

            encodr = FeatureEncoder(target=self.target, use_woe_encoder=useWoeEncoder, drop_features=True,
                                    we_cut_method=cut_method, be_cut_method=cut_method, recoding_dict=self.recoding_dict,
                                    feat_dict=self.feat_dict)
            df = encodr.fit_transform(df, self.label, exclude_list=self.exclude_list)

            scalr = FeatureScaler(target=self.target, silent=silent, scaler_type=scaler_type,
                                  recoding_dict=self.recoding_dict, feat_dict=self.feat_dict)
            df = scalr.fit_transform(df, label=self.label, action=useScalr,
                                     exclude_list=self.exclude_list+encodr.final_OneHotEncoder_new_feat)
        self.write_recoding_txt(file_name, encoding=encoding)

    def write_recoding_txt(self, file_name, encoding="utf-8"):
        """
        Write recoding statements.

        Parameters
        ----------
        file_name: str
                Output file's name
        encoding: str
                Encoding standard
        """
        for k, v in self.recoding_dict.items():
            self.recoding_statement += '\n%s Recoding Statement for %s %s\n%s\n' %('#'*40, k, '#'*40, v)

        self.recoding_file = file_name
        fo = open(self.recoding_file, "w", encoding=encoding)
        fo.write(self.recoding_statement)
        fo.close()
        self.final_feats = list(self.feat_dict.values())
        self.recoding_dict = dict(zip(self.feats, [''] * len(self.feats)))
        self.feat_dict = dict(zip(self.feats, self.feats))

    def write_statistical_summary(self, file_name, encoding="utf-8"):
        """
        Write statistical summary.

        Parameters
        ----------
        file_name: str
                Output file's name
        encoding: str
                Encoding standard

        Returns
        ----------
        dcp_stat: pd.DataFrame
                Descriptive stats
        """
        self.dcp_stat = desc_stat(self.df_ori[list(self.feats)], target=self.target, is_ratio_pct=True,
                                  use_formater=True, silent=True)
        woe_stat = self.woe_recoding_stat('', encoding=encoding, write_recoding_statement=False)
        iv_stat = woe_stat.groupby('feature').iv.max()
        self.dcp_stat = pd.concat([self.dcp_stat, iv_stat], axis=1)
        self.dcp_stat.to_excel(file_name, encoding=encoding)

        return self.dcp_stat

    def exec_recoding(self, df_e, encoding="utf-8"):
        """
        Execute recoding statement to the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        encoding: str
                Encoding standard

        Returns
        ----------
        df_copy: pd.DataFrame
                Recoded dataframe
        """
        df = df_e.copy()
        fo = open(self.recoding_file, 'r', encoding=encoding)
        recoding_text = fo.read()
        fo.close()
        exec(recoding_text)
        return df

    def woe_recoding_stat(self, file_name, encoding="utf-8", unseen='<UNSEEN>*', unknown='<NA>*', 
                          write_recoding_statement=True):
        """
        Output recoding woe recoding statistics.

        Parameters
        ----------
        file_name: str
                Output file's name
        encoding: str
                Encoding standard
        unseen: str, int or float
                String or value used to denote unknown case (no information)
        unknown: str, int or float
                String or value used to denote unseen case (case not yet seen)
        write_recoding_statement: boolean
                Write recoding statement
        """
        df_copy = self.df_ori.copy()
        cav_list = self.df_ori[list(self.feats)].select_dtypes(include=[object]).columns.tolist()
        cov_list = self.df_ori[list(self.feats)].select_dtypes(exclude=[object]).columns.tolist()
        col = ['feature', 'iv', 'woe', 'group','depvar_count', 'depvar_rate', 'count', 'count_proportion',
               'lift', 'max', 'min']
        cav_recoding_stat = pd.DataFrame(columns=col)

        for i in cav_list:
            we_tmp = WoeEncoder(x_dtypes='cav', cut_method='dt')
            we_tmp.fit(df_copy[i], df_copy[self.label], prefix=i)
            df_map = pd.DataFrame([we_tmp.dmap], index=['value']).T.drop([unseen], axis=0)
            df_map = df_map.reset_index(drop=False).set_index('value')
            df_copy[i].fillna(unknown, inplace=True)
            df_map2 = df_map[df_map['index'] != unknown]

            for value in df_map2.index.unique():
                if type(df_map2.loc[value, 'index']) != pd.Series:
                    group_list = [df_map2.loc[value, 'index']]

                else:
                    group_list = df_map2.loc[value, 'index'].values.tolist()

                tmp = df_copy.loc[[x in group_list for x in df_copy[i]],]
                cav_recoding_stat = \
                    pd.concat([cav_recoding_stat,
                               pd.DataFrame([i, we_tmp.iv[0], value, str(group_list),
                                             tmp[self.label].sum(), tmp[self.label].mean(),
                                             tmp[self.label].count(), tmp[self.label].count() / len(df_copy),
                                             tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                             np.nan, np.nan], index=col).T],
                              axis=0)

            tmp = df_copy[df_copy[i] == unknown]
            cav_recoding_stat = \
                pd.concat([cav_recoding_stat,
                           pd.DataFrame([i, we_tmp.iv[0], df_map[df_map['index'] == unknown].index[0],
                                         unknown, tmp[self.label].sum(), tmp[self.label].mean(),
                                         tmp[self.label].count(), tmp[self.label].count() / len(df_copy),
                                         tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                         np.nan, np.nan], index=col).T],
                          axis=0)

        cov_recoding_stat = pd.DataFrame(columns=col)

        for i in cov_list:
            we_tmp = WoeEncoder(x_dtypes='cov', cut_method='dt')
            we_tmp.fit(df_copy[i], df_copy[self.label], prefix=i)

            for interval, value in we_tmp.dmap.items():
                if interval not in [unseen, unknown]:
                    tmp = df_copy.loc[[x in interval for x in df_copy[i]],]
                    cov_recoding_stat = \
                        pd.concat([cov_recoding_stat,
                                   pd.DataFrame([i, we_tmp.iv[0], value, interval, tmp[self.label].sum(),
                                                 tmp[self.label].mean(), tmp[self.label].count(),
                                                 tmp[self.label].count() / len(df_copy),
                                                 tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                                 tmp[i].max(), tmp[i].min()], index=col).T],
                                  axis=0)

                elif interval == unknown:
                    tmp = df_copy[df_copy[i].isnull()]
                    cov_recoding_stat = \
                        pd.concat([cov_recoding_stat,
                                   pd.DataFrame([i, we_tmp.iv[0], value, interval, tmp[self.label].sum(),
                                                 tmp[self.label].mean(), tmp[self.label].count(),
                                                 tmp[self.label].count() / len(df_copy),
                                                 tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                                 np.nan, np.nan], index=col).T],
                                  axis=0)

        cov_recoding_stat.sort_values(by=['iv', 'feature', 'group'], ascending=[False, False, True],
                                      inplace=True)
        cav_recoding_stat.sort_values(by=['iv', 'feature', 'group'], ascending=[False, False, False],
                                      inplace=True)

        stat = pd.concat([cov_recoding_stat, cav_recoding_stat], axis=0)
        stat[['depvar_rate', 'count_proportion']] = \
                                    stat[['depvar_rate', 'count_proportion']].applymap(
                                    lambda x: '{:.2f}%'.format(x * 100) if not np.isnan(x) else x)
        self.recoding_stat = stat
        if write_recoding_statement:
            self.recoding_stat.to_excel(file_name, encoding=encoding, index=None)
        return self.recoding_stat