import numpy as np
import pandas as pd

from feature_encoding import UD_WoeEncoder
from data_cleaning import desc_stat
from tools import *


class GENERATE_OUTPUT_FILES:

    def __init__(self, df_ori, label, target=TARGET, exclude_list=[]):
        """
        : params df_ori: original dataframe
        : params label: label will be used in the modeling process
        : params target: list of default boolean targets
        : params exclude_list: list of features excluded from being recoded
        : params feats: features used to train the model
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : params recoding_statement: recoding statement to output
        """
        self.df_ori = df_ori
        self.label = label
        self.target = target
        self.exclude_list = exclude_list
        self.feats = set(df_ori.columns) - set(self.exclude_list + [self.label] + self.target)
        self.recoding_dict = dict(zip(self.feats, [''] * len(self.feats)))
        self.feat_dict = dict(zip(self.feats, self.feats))
        self.recoding_statement = ''

    def write_recoding_txt(self, file, encoding="utf-8"):
        """
        Output recoding statements

        : params file: output file's name
        : params encoding: encoding standard
        """
        for k, v in self.recoding_dict.items():
            self.recoding_statement += '\n' + '#' * 40 + ' Recoding Statement for ' \
                                    + str(k) + ' ' + '#' * 40 + '\n' + v + '\n'

        self.recoding_file = file
        fo = open(self.recoding_file, "w", encoding=encoding)
        fo.write(self.recoding_statement)
        fo.close()

    def write_statistical_summary(self, file, encoding="utf-8"):
        """
        Output recoding statements

        : params file: output file's name
        : params encoding: encoding standard
        """
        self.dcp_stat = desc_stat(self.df_ori[list(self.feats)], target=self.target, ratio_pct=True,
                                  use_formater=True, silent=True)
        woe_stat = self.woe_recoding_stat('', encoding=encoding, write=False)
        iv_stat = woe_stat.groupby('feature').iv.max()
        self.dcp_stat = pd.concat([self.dcp_stat, iv_stat], axis=1)
        self.dcp_stat.to_excel(file, encoding=encoding)

        return self.dcp_stat

    def exec_recoding(self, df, encoding="utf-8"):
        """
        Execute recoding statement to the input dataframe

        : params df: the input dataframe
        : params encoding: encoding standard

        : return df_copy: recoded dataframe
        """
        df_copy = df.copy()
        fo = open(self.recoding_file, 'r', encoding=encoding)
        recoding_text = fo.read()
        fo.close()
        exec(recoding_text)

        return df_copy

    def woe_recoding_stat(self, file, encoding="utf-8", unseen='<UNSEEN>*', unknown='<NA>*', write=True):
        """
        Output recoding woe recoding statistics

        : params file: output file's name
        : params encoding: encoding standard
        : params unseen: string or value used to denote unknown case (no information)
        : params unknown: string or value used to denote unseen case (case not yet seen)
        """
        df_copy = self.df_ori.copy()
        cav_list = self.df_ori[list(self.feats)].select_dtypes(include=[object]).columns.tolist()
        cov_list = self.df_ori[list(self.feats)].select_dtypes(exclude=[object]).columns.tolist()

        col = ['feature', 'iv', 'woe', 'group','depvar_count', 'depvar_rate', 'count', 'count_proportion',
               'lift', 'max', 'min']
        cav_recoding_stat = pd.DataFrame(columns=col)

        for i in cav_list:
            we_tmp = UD_WoeEncoder(x_dtypes='cav', cut_method='dt')
            we_tmp.ud_fit(df_copy[i], df_copy[self.label], prefix=i)
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
            we_tmp = UD_WoeEncoder(x_dtypes='cov', cut_method='dt')
            we_tmp.ud_fit(df_copy[i], df_copy[self.label], prefix=i)

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

        if write:
            self.recoding_stat.to_excel(file, encoding=encoding, index=None)

        return self.recoding_stat