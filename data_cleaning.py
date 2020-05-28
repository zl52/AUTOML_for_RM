import pandas as pd
import numpy as np
import missingno as msno

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################    1. DATA CLEANING    ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################



def plot_missingno(df):
    """
    Visualize the nullity of the given dataframe.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    """
    col_num = df.shape[1]
    msno.matrix(df, labels=True, figsize=(col_num/1.5, 6), fontsize=10)
    print('Figure 1: A matrix visualization of the nullity of the given DataFrame')
    msno.bar(df, figsize=(col_num/1.5, 6), fontsize=10)
    print('Figure 2: A bar chart visualization of the nullity of the given DataFrame')
    msno.heatmap(df, figsize=(col_num/2, col_num/3.2), fontsize=10)
    print('Figure 3: Presents a \'seaborn\' heatmap visualization of nullity correlation in the given DataFrame')


def na_detection(df, na_list_appd=[]):
    """
    Detect "NA"s in the dataframe and replace them with np.nan.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    na_list_appd: list
            List of extra na strings to be replaced

    Returns
    ----------
    df: pd.DataFrame
            The output dataframe where na strings are replaced with np.nan
    """
    na_list = ['', 'null', 'NULL', 'Null', 'NA', 'na', 'Na', 'nan', 'NAN', 'Nan', 'NaN', '未知', '无'] \
              + na_list_appd
    df = df.replace(na_list, np.nan)
    return df


def desc_stat(df, target=[], is_ratio_pct=True, use_formater=True, silent=True):
    """
    Generate descriptive statistical summary, including missing value count, missing rate, coverage count,
    coverage rate, unique-value-count, high-frequency-value, high-frequency-value's count,
    high-frequency-value's probability of occurrence, 1% percentile and 99% percentile.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    target: list
            list of target columns
    is_ratio_pct: boolean
            If True, display ratio in percentage
    use_formater: boolean
            If True, display float numbers in format
    silent: boolean
            If True, restrict the print of detailed statements

    Returns
    ----------
    stat: pd.DataFrame
            Statistical summary  
    """
    df = na_detection(df)
    cav_list = list(set(df.select_dtypes(include=[object])) - set(target))
    cov_list = list(set(df.select_dtypes(exclude=[object])) - set(target))
    col = cav_list + cov_list

    # statistical summary for continuous columns
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

    # statistical summary for categorical columns
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
        # HF: high-frequency
        HF_value=df[col].apply(lambda x: x.value_counts().index[0] if x.count() != 0 else np.nan, axis=0),
        HF_value_cnt=df[col].apply(lambda x: x.value_counts().iloc[0] if x.count() != 0 else np.nan, axis=0),
        HF_value_pct=df[col].apply(lambda x: x.value_counts().iloc[0] / len(x) if x.count() != 0 else np.nan, axis=0))
    stat['count'] = stat['count'].astype(int)
    if is_ratio_pct:
        trans_col = ['missing_rate', 'coverage_rate', 'HF_value_pct']
        stat[trans_col] = stat[trans_col].applymap(lambda x: '{:.2f}%'.format(x * 100))
    if use_formater:
        trans_col = stat.select_dtypes(include=[float]).columns
        stat[trans_col] = stat[trans_col].applymap(lambda x: '{0:.02f}'.format(x))
    if not silent:
        print("Generate descriptive statistical summary, including missing count, missing rate,",
              "\n coverage count, coverage rate, unique values count, high-frequency value,",
              "high-frequency value's count, high-frequency value's\n probability of occurrence")
        print('\n%s \n'%('_'*120))
    return stat


class DataFilter(object):
    """
    Drop columns having unique-value-counts below threshold, columns having missing rates
    above threshold and columns having high-frequency-value above threshold.
    """

    def __init__(self,
                 nunique_thr=2,
                 missing_rate_thr=0.99,
                 HF_value_thr=0.98,
                 exclude_list=[],
                 silent=False):
        """
        Parameters
        ----------
        nunique_thr: float
                Threshold of unique-value-count
        missing_rate_thr: float
                Threshold of missing rate
        HF_value_thr: float
                Threshold of high-frequency value's occurrence propability
        exclude_list: list
                List of columns removed manually
        silent: boolean
                If True, restrict the print of filtering process
        """
        self.nunique_thr = nunique_thr
        self.missing_rate_thr = missing_rate_thr
        self.HF_value_thr = HF_value_thr
        self.exclude_list = exclude_list
        self.silent = silent

    def fit(self, df):
        """
        Fit transformer by checking the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe
        """
        df = na_detection(df)
        stat = desc_stat(df, is_ratio_pct=False, use_formater=False)
        drop1 = stat[stat['unique_value_cnt'] < self.nunique_thr].index.tolist()
        drop2 = stat[stat['missing_rate'] > self.missing_rate_thr].index.tolist()
        drop3 = stat[stat['HF_value_pct'] > self.HF_value_thr].index.tolist()
        self.exclude_list = drop1 + drop2 + drop3 + self.exclude_list
        trans_col = ['missing_rate', 'coverage_rate']
        stat[trans_col] = stat[trans_col].applymap(lambda x: '{:.2f}%'.format(x * 100))
        if not self.silent:
            print('Statistics of columns to be dropped \n')
            print(stat[['unique_value_cnt', 'missing_rate', 'HF_value_pct']].loc[drop1 + drop2 + drop3])
            print('\n%s \n'%('_ '*60))
            print('Missing rates of these columns exceed defined threshold: %f\n\n'%self.missing_rate_thr)
            print('\n'.join(drop1))
            print('\n%s \n'%('_ '*60))
            print('Unique values of these columns do not exceed defined threshold: %f\n\n'%self.nunique_thr)
            print('\n'.join(drop2))
            print('\n%s \n'%('_ '*60))
            print('Percentages of high-frequency-values of these columns exceed defined threshold: %f\n\n'%self.HF_value_thr)
            print('\n'.join(drop3))
            print('\n%s \n'%('_ '*60))

    def transform(self, df):
        """
        Fit transformer by checking the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe

        Returns
        ----------
        df: pd.DataFrame
                The dataframe of remaining columns
        """
        df = na_detection(df)
        try:
            df = df.drop(self.exclude_list, axis=1)
            if not self.silent:
                print('Columns shown as below are dropped\n\n')
                print('\n'.join(self.exclude_list))
            return df.apply(pd.to_numeric, errors='ignore')
        except Exception as e:
            print(e)
            raise ValueError("Some columns to drop are not in the dataframe")

    def fit_transform(self, df):
        """
        Fit transformer by checking the dataframe and transform the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe

        Returns
        ----------
        df: pd.DataFrame
                The dataframe of remaining columns
        """
        self.fit(df)
        return self.transform(df)
