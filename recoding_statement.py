import pandas as pd
import numpy as np

from tools import *
from feature_encoding import UD_WoeEncoder

class generate_recoding_statement:

    def __init__(self, dataset, label, target=TARGET, exclude_list=[]):
        """
        : params dataset: the input dataset
        : param label: label will be used in the modeling process
        : params target: list of default boolean targets
        : param exclude_list: list of features excluded from being recoded
        : params col: features used to train the model
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : param recoding_statement: recoding statement to output
        """
        self.dataset = dataset
        self.label = label
        self.target = target
        self.exclude_list = exclude_list
        self.col = set(dataset.columns) - set(self.exclude_list + [self.label] + self.target)
        self.recoding_dict = dict(zip(self.col, [''] * len(self.col)))
        self.feat_dict = dict(zip(self.col, self.col))
        self.recoding_statement = ''

    def write_recoding_txt(self, file, encoding="utf-8"):
        """
        Output recoding statements

        : params file: out file's name
        : param encoding: encoded standard
        """
        for k, v in self.recoding_dict.items():
            self.recoding_statement += '\n' + '#' * 40 + ' Recoding Statement for ' \
                                       + str(k) + ' ' + '#' * 40 + '\n' + v + '\n'

        self.file = file
        fo = open(self.file, "w", encoding=encoding)
        fo.write(self.recoding_statement)
        fo.close()

    def exec_recoding(self, df, encoding="utf-8"):
        """
        Execute recoding statement to the input dataframe

        : params df: the input dataframe
        : param encoding: encoded standard

        : return df_copy: recoded dataframe
        """
        df_copy = df.copy()
        fo = open(self.file, 'r', encoding=encoding)
        recoding_text = fo.read()
        fo.close()
        exec(recoding_text)

        return df_copy

    def woe_recoding_stat(self, file, encoding="utf-8"):
        """
        Output recoding woe recoding statistics

        : params file: out file's name
        : param encoding: encoded standard
        """
        df_copy = self.dataset.copy()
        cav_list = self.dataset[list(self.col)].select_dtypes(exclude=[float, int, 'int64']).columns.tolist()
        cov_list = self.dataset[list(self.col)].select_dtypes(include=[float, int, 'int64']).columns.tolist() 
        
        cav_recoding_stat = pd.DataFrame()
        for i in cav_list:
            we_tmp = UD_WoeEncoder(x_dtypes='cav', cut_method='dt')
            we_tmp.ud_fit_transform(df_copy[i], df_copy[self.label], prefix=i)
            df_map = pd.DataFrame([we_tmp.dmap], index=['value']).T.drop(['<UNSEEN>*'], axis=0)
            df_map = df_map.reset_index(drop=False).set_index('value')

            df_copy[i].fillna('<NA>*',inplace=True)
            df_map2 = df_map[df_map['index']!='<NA>*']
            for value in df_map2.index.unique():
                if type(df_map2.loc[value, 'index']) != pd.Series:
                    group_list = [df_map2.loc[value, 'index']]

                else:
                    group_list = df_map2.loc[value, 'index'].values.tolist()
                
                tmp = df_copy.loc[[x in group_list for x in df_copy[i]],]
                cav_recoding_stat = \
                         pd.concat([cav_recoding_stat, 
                                    pd.DataFrame([i, we_tmp.iv[0], value, str(group_list),
                                                  tmp[self.label].sum(),tmp[self.label].mean(),
                                                  tmp[self.label].count(), tmp[self.label].count() / len(df_copy),
                                                  tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                                  np.nan, np.nan]).T],
                                    axis=0)

            tmp = df_copy[df_copy[i] == '<NA>*']
            cav_recoding_stat = \
                     pd.concat([cav_recoding_stat, 
                                pd.DataFrame([i, we_tmp.iv[0], df_map[df_map['index']=='<NA>*'].index[0],
                                             '<NA>*', tmp[self.label].sum(),tmp[self.label].mean(),
                                             tmp[self.label].count(), tmp[self.label].count() / len(df_copy),
                                             tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                             np.nan, np.nan]).T],
                                axis=0)
                    
        cov_recoding_stat = pd.DataFrame()
        for i in cov_list:
            we_tmp = UD_WoeEncoder(x_dtypes='cov', cut_method='dt')
            we_tmp.ud_fit_transform(df_copy[i], df_copy[self.label], prefix=i)

            for interval, value in we_tmp.dmap.items():
                if interval not in ['<UNSEEN>*','<NA>*']:
                    tmp = df_copy.loc[[x in interval for x in df_copy[i]],]
                    cov_recoding_stat = \
                         pd.concat([cov_recoding_stat, 
                                    pd.DataFrame([i, we_tmp.iv[0], value, interval, tmp[self.label].sum(),
                                                  tmp[self.label].mean(), tmp[self.label].count(),
                                                  tmp[self.label].count() / len(df_copy),
                                                  tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                                  tmp[i].max(), tmp[i].min()]).T],
                                    axis=0)
                    
                elif interval == '<NA>*':
                    tmp = df_copy[df_copy[i].isnull()]
                    cov_recoding_stat = \
                         pd.concat([cov_recoding_stat, 
                                    pd.DataFrame([i, we_tmp.iv[0], value, interval, tmp[self.label].sum(),
                                                  tmp[self.label].mean(), tmp[self.label].count(),
                                                  tmp[self.label].count() / len(df_copy),
                                                  tmp[self.label].mean() * 100 / df_copy[self.label].mean(),
                                                  np.nan, np.nan]).T],
                                    axis=0)
                    
        cov_recoding_stat.columns = cav_recoding_stat.columns = ['feature','iv','woe','group',
                                                                 'depvar_count','depvar_rate','count',
                                                                 'count_proportion', 'lift','max','min']
        cov_recoding_stat.sort_values(by=['iv','feature','group'],ascending=[False,False,True],inplace=True)
        cav_recoding_stat.sort_values(by=['iv','feature','group'],ascending=[False,False,False],inplace=True)

        stat = pd.concat([cov_recoding_stat, cav_recoding_stat], axis=0)
        stat[['depvar_rate','count_proportion']] = stat[['depvar_rate','count_proportion']].applymap(
                                    lambda x: '{:.2f}%'.format(x * 100) if not np.isnan(x) else x)
        stat.to_excel(file, encoding = encoding, index = None)

        return stat