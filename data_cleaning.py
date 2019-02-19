import pandas as pd
import numpy as np

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################    1. DATA CLEANING    ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def na_detection(df, na_list_appd=[]):
    """
    Detect "NA"s in the dataframe

    : params df: the input dataframe
    : params na_list_appd: list of extra strings to be replaced by np.nan

    : return: the output dataframe of "NA" replaced
    """
    na_list = ['', 'null', 'NULL', 'Null', 'NA', 'na', 'Na', 'nan', 'NAN', 'Nan', 'NaN', '未知', '无'] \
              + na_list_appd
    df = df.replace(na_list, np.nan)

    return df


def desc_stat(df, target=TARGET, ratio_pct=True, use_formater=True, silent=True):
    """
    Generate each feature's descriptive statistical summary, including missing count, missing rate, 
    coverage count, coverage rate, unique-values-count, high-frequency-value, high-frequency-value's count,
    high-frequency-value's probability of occurrence, 1% percentile and 99% percentile

    : params df: the input dataframe
    : params target: list of default boolean targets
    : params ratio_pct: whether to output ratio in percentage
    : params use_formater: whether to output float numbers in format
    : params silent: whether to print statistical summary

    : return df: the dataframe of sample's statistical summary  
    """
    df = na_detection(df)
    col = [i for i in (set(df.columns) - set(target))]
    cav_list = [i for i in col if i in df.select_dtypes(include=[object])]
    cov_list = [i for i in col if i in df.select_dtypes(exclude=[object])]

    try:
        stat_cov = df[cov_list].describe().T
        col_name = stat_cov.columns.tolist()
        col_name.insert(4, '1%')
        col_name.insert(8, '99%')
        stat_cov['1%'] = df[stat_cov.index.tolist()].apply(lambda x: x.quantile(0.01), axis=0)
        stat_cov['99%'] = df[stat_cov.index.tolist()].apply(lambda x: x.quantile(0.99), axis=0)
        stat_cov = stat_cov.reindex(columns=col_name)

    except:
        stat_cov = pd.DataFrame()
        print('There are no continous variables or quantiles failed to calculate')


    try:
        stat_cav = df[cav_list].describe().T.drop(['freq', 'top', 'unique'],axis=1)

    except:
        stat_cav = pd.DataFrame()
        print('There are no categorical variables')

    stat = pd.concat([stat_cov, stat_cav], axis=0)
    stat = stat.assign( \
        missing_cnt=df[col].apply(lambda x: len(x) - x.count(), axis=0),
        missing_rate=df[col].apply(lambda x: (len(x) - x.count()) / len(x), axis=0),
        coverage_count=df[col].apply(lambda x: x.count(), axis=0),
        coverage_rate=df[col].apply(lambda x: x.count() / len(x), axis=0),
        unique_value_cnt=df[col].apply(lambda x: x.nunique(), axis=0),
        HF_value=df[col].apply(lambda x: x.value_counts().index[0] \
            if x.count() != 0 else np.nan, axis=0),
        HF_value_cnt=df[col].apply(lambda x: x.value_counts().iloc[0] \
            if x.count() != 0 else np.nan, axis=0),
        HF_value_pct=df[col].apply(lambda x: x.value_counts().iloc[0] / len(x) \
            if x.count() != 0 else np.nan, axis=0))

    stat['count'] = stat['count'].astype(int)

    if ratio_pct:
        stat[['missing_rate', 'coverage_rate', 'HF_value_pct']] = \
            stat[['missing_rate', 'coverage_rate', 'HF_value_pct']].applymap(lambda x: '{:.2f}%'.format(x * 100))

    if use_formater:
        formater = "{0:.02f}".format
        stat[stat.select_dtypes(include=[float]).columns] = \
            stat[stat.select_dtypes(include=[float]).columns].applymap(formater)

    if not silent:
        print("Generate each feature\'s descriptive statistical summary, including missing count,",
              "missing rate, \n coverage count, coverage rate, unique values count, high-frequency",
              "value, high-frequency value's count, high-frequency value's\n probability of occurrence")
        print('\n' + '_' * 120 + ' \n')

    return stat



class DATA_FILTER(object):
    """
    Filter columns with number of unique values below defined threshold, columns with missing rates
    above defined threshold and columns with a value of high frequency above defined threshold
    """

    def __init__(self, nunique_thr=2, missing_rate_thr=0.99, HF_value_thr=0.98,
                 drop_cols=[], silent=False):
        """
        : params nunique_thr: threshold of unique-value-count
        : params missing_rate_thr: threshold of missing rate
        : params HF_value_thr: threshold of high-frequency value's occurrence propability
        : params drop_cols: list of columns should be removed manually
        : params silent: whether to print details of filtering process
        """
        self.nunique_thr = nunique_thr
        self.missing_rate_thr = missing_rate_thr
        self.HF_value_thr = HF_value_thr
        self.drop_cols = drop_cols
        self.silent = silent

    def ud_fit(self, df):
        """
        Fit transformer by checking the dataframe

        : params df: the input dataframe
        """
        stat = desc_stat(df, target=[], ratio_pct=False, use_formater=False)
        drop1 = stat[stat['unique_value_cnt'] < self.nunique_thr].index.tolist()
        drop2 = stat[stat['missing_rate'] > self.missing_rate_thr].index.tolist()
        drop3 = stat[stat['HF_value_pct'] > self.HF_value_thr].index.tolist()
        self.drop_cols = drop1 + drop2 + drop3 + self.drop_cols

        stat['unique_value_cnt'] = stat['unique_value_cnt'].astype(int)
        stat[['missing_rate', 'coverage_rate']] = \
            stat[['missing_rate', 'coverage_rate']].applymap(lambda x: '{:.2f}%'.format(x * 100))

        if not self.silent:
            print('Statistics of columns to be dropped \n')

            print(stat[['unique_value_cnt', 'missing_rate', 'HF_value_pct']].loc[drop1 + drop2 + drop3])
            print('\n' + '_ ' * 60 + ' \n')

            print('Missing rates of these columns exceed defined threshold: {thr}\n\n' \
                  .format(thr=self.missing_rate_thr), '\n '.join(drop1))
            print('\n' + '_ ' * 60 + ' \n')

            print('Unique values of these columns do not exceed defined threshold: {thr}\n\n' \
                  .format(thr=self.nunique_thr), '\n '.join(drop2))
            print('\n' + '_ ' * 60 + ' \n')

            print('Percentages of high-frequency-values of these columns exceed defined threshold: {thr}\n\n' \
                  .format(thr=self.HF_value_thr), '\n '.join(drop3))
            print('\n' + '_ ' * 60 + ' \n')

    def ud_transform(self, df):
        """
         Transform the dataframe

        : params df: the input dataframe

        : return df: the dataframe of columns having been filtered
        """
        df = na_detection(df)

        try:
            df.drop(self.drop_cols, axis=1, inplace=True)

            if not self.silent:
                print('Columns shown as below are dropped\n\n', '\n '.join(self.drop_cols))

            return df.apply(pd.to_numeric, errors='ignore')

        except:
            raise ValueError("Some columns to drop are not in the dataframe")

    def ud_fit_transform(self, df):
        """
        Fit transformer by checking the dataframe and transform the dataframe

        : params df: the input dataframe

        : return df: the dataframe of filtered columns
        """
        self.ud_fit(df)

        return self.ud_transform(df)
