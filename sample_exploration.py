import pandas as pd;
import numpy as np

from scipy.stats import chi2_contingency, chisquare
from data_cleaning import na_detection

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 3. SAMPLE EXPLORATION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def get_pos_rate(df, target=TARGET, is_ratio_pct=True, silent=True, event=1):
    """
    Calculate positive rate (dependent variable's ratio of being positive).

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    target: list
            List of default boolean targets
    is_ratio_pct: boolean
            Output ratios in percentages
    silent: boolean
            Restrict the print of details
    event: int
            Value of target variable that stands for the event to predict

    Returns
    ----------
    dr: pd.DataFrame
            the dataframe of positive rate according to all targets
    """
    dr = pd.DataFrame(columns=target, index=['positive_rate'])
    for i in target:
        dr.loc['positive_rate', i] = df[i].mean()
    if event == 0:
        dr = 1 - dr
    if is_ratio_pct:
        dr = dr.applymap(lambda x: '{:.1f}%'.format(x * 100) if x != np.nan else x)
    if not silent:
        print("Calculate each target\'s positive rate (dependent variable\'s ratio of being positive)\n")
        print(dr)
        print('\n%s \n'%('_'*120))
    return dr


def get_od_label(df, OD_TARGET=OD_TARGET, col='overdue_day', drop_od_col=False, silent=False, event=1):
    """
    Calculate samples' delay days of repayment and overdue rate in total according to each target.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    col: str
            Column representing delay days of repayment
    OD_TARGET: list
            List of delay days to calculate boolean labels
    drop_od_col: boolean
            drop the column representing delay days of repayment
    silent: boolean
            Restrict print of positve rates
    event: int
            Value of target variable that stands for the event to predict

    Returns
    ----------
    df: pd.DataFrame
            Dataframe of added boolean labels
    """
    try:
        for i in OD_TARGET:
            df['d{i}'.format(i=i)] = df[col].map(lambda x: 1 - event if x < i + 1 else event)
    except:
        raise ValueError("overdue targets must be a list of integers")
    target = default_target(OD_TARGET)
    _ = get_pos_rate(df, target, silent=silent, event=event)
    if drop_od_col == True:
        df = df.drop([col], axis=1)
    return df


def desc_stat(df, target=TARGET, is_ratio_pct=True, use_formater=True, silent=True):
    """
    Generate descriptive statistical summary, including missing count, missing rate, coverage count,
    coverage rate, unique-value-count, high-frequency-value, high-frequency-value's count,
    high-frequency-value's probability of occurrence, 1% percentile and 99% percentile.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    target: list
            list of default boolean targets
    is_ratio_pct: boolean
            Output ratio in percentage
    use_formater: boolean
            Output float numbers in format
    silent: boolean
            Restrict the print of detail statements

    Returns
    ----------
    stat: pd.DataFrame
            Statistical summary  
    """
    df = na_detection(df)
    col = [i for i in (set(df.columns) - set(target))]
    cav_list = [i for i in col if i in df.select_dtypes(include=[object]).columns]
    cov_list = [i for i in col if i in df.select_dtypes(exclude=[object]).columns]

    # statistical summary for categorical features
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

    # statistical summary for continuous features
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
        HF_value=df[col].apply(lambda x: x.value_counts().index[0] \
                               if x.count() != 0 else np.nan, axis=0),
        HF_value_cnt=df[col].apply(lambda x: x.value_counts().iloc[0] \
                                   if x.count() != 0 else np.nan, axis=0),
        HF_value_pct=df[col].apply(lambda x: x.value_counts().iloc[0] / len(x) \
                                   if x.count() != 0 else np.nan, axis=0))
    stat['count'] = stat['count'].astype(int)
    if is_ratio_pct:
        stat[['missing_rate', 'coverage_rate', 'HF_value_pct']] = \
            stat[['missing_rate', 'coverage_rate', 'HF_value_pct']].applymap(lambda x: '{:.2f}%'.format(x * 100))
    if use_formater:
        stat[stat.select_dtypes(include=[float]).columns] = \
            stat[stat.select_dtypes(include=[float]).columns].applymap(lambda x: '{0:.02f}'.format(x))
    if not silent:
        print("Generate descriptive statistical summary, including missing count, missing rate,",
              "\n coverage count, coverage rate, unique values count, high-frequency, value,",
              "high-frequency value's count, high-frequency value's\n probability of occurrence")
        print('\n%s \n'%('_'*120))
    return stat


def feats_coverage_stat(df, target=TARGET, is_ratio_pct=True, sort_value=True, silent=True, event=1):
    """
    Calculate positive rates of samples covered and uncovered by each feature.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    target: list
            List of default boolean targets
    is_ratio_pct: boolean
            Dispay ratios in percentage format
    sort_value: boolean
            Sort values
    event: int
            Value of target variable that stands for the event to predict
    silent: boolean
            Restrict the print of outputs

    Returns
    ----------
    stat: pd.Dataframe
            The dataframe of statistical summary   
    """
    df = na_detection(df)
    col = [i for i in (set(df.columns) - set(target))]
    stat = pd.DataFrame()
    for i in col:
        if df[i].count() != 0:
            cover_pos_rate = get_pos_rate(df[df[i].notnull()], target, is_ratio_pct=False, silent=True, event=event)
            uncover_pos_rate = get_pos_rate(df[df[i].isnull()], target, is_ratio_pct=False, silent=True, event=event)
            tmp = pd.concat([cover_pos_rate, uncover_pos_rate], axis=0)
            tmp.index = [[i, i], ['COVERED', 'UNCOVERED']]
            stat = pd.concat([stat, tmp], axis=0)
        else:
            print('%s does not cover the sample'%i)
    if sort_value:
        stat = stat.sort_values(by=target[0], ascending=False)
    if is_ratio_pct:
        stat = stat.applymap(lambda x: '{:.1f}%'.format(x * 100) if x != np.nan else x)
    if not silent:
        print("Calculate positive rate of samples covered and uncovered by each feature\n")
        print(stat.head(10))
        print('\n%s \n'%('_'*120))
    return stat


def sample_coverage_stat(df, target=TARGET, exclude_list=[], silent=True, event=1):
    """
    Calculate fully or partially covered sample's positive rate.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    target: list
            List of default boolean targets
    exclude_list: list
            Columns to be excluded when calculating coverage rate
    silent: boolean
            Restrict the print of outputs
    event: int
            Value of target variable that stands for the event to predict

    Returns
    ----------
    stat: dataframe of statistical summary   
    """
    col = list(set(df.columns.tolist()) - set(exclude_list))
    df = na_detection(df)
    df['covered'] = (df[col].notnull().sum(axis=1) > len(target)).astype(int)
    if event == 1:
        stat = pd.pivot_table(df, index='covered', values=target, margins=True)
    elif event == 0:
        stat = 1 - pd.pivot_table(df, index='covered', values=target, margins=True)
    if not silent:
        print("Calculate fully or partially covered samples and totally uncovered samples' positive",
              "rate of label\n")
        print(stat)
    return stat


def feats_coverage_chi_test(df, target=TARGET, thr=0.01, event=1, smoothing_param=0.0001):
    """
    Abandoned
    """
    res = pd.DataFrame()
    rowcnt = len(df)
    col = [i for i in (set(df.columns) - set(target))]
    for s in col:
        feature_add = pd.DataFrame()
        for t in target:
            feature_covered_rate = (df[df[s].isnull()][t] == event).sum() / rowcnt
            feature_uncovered_rate = (df[-df[s].isnull()][t] == event).sum() / rowcnt
            num1 = (df[df[s].isnull()][t] == 1 - event).sum() + smoothing_param
            num2 = (df[df[s].isnull()][t] == event).sum() + smoothing_param
            num3 = (df[-df[s].isnull()][t] == 1 - event).sum() + smoothing_param
            num4 = (df[-df[s].isnull()][t] == event).sum() + smoothing_param
            matrix = np.array([[num1, num2], [num3, num4]])
            feature_target_add = pd.DataFrame([feature_covered_rate,
                                              feature_uncovered_rate,
                                              np.NaN],
                                             index=[s + '_cover', s + '_uncover', 'sig'],
                                             columns=[str(t)]).T
            feature_target_add['sig'] = True if stats.chisquare(matrix)[1] < thr else False
            feature_add = pd.concat([feature_add, feature_target_add], axis=1)
        res = pd.concat([res, feature_add], axis=0)
    return np.round(res, 3)
