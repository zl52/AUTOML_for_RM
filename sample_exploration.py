import pandas as pd;
import numpy as np
from scipy.stats import chi2_contingency, chisquare
from sklearn.utils.multiclass import type_of_target

from data_cleaning import na_detection, desc_stat
from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 3. SAMPLE EXPLORATION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def get_pos_rate(df, target, is_ratio_pct=True, silent=True, event=1):
    """
    Calculate positive rate.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    target: list
            List of boolean targets
    is_ratio_pct: boolean
            If True, display ratios in percentages
    silent: boolean
            If True, restrict the print of details
    event: int
            Value of target variable that stands for the event to predict

    Returns
    ----------
    dr: pd.DataFrame
            the dataframe of positive rate according to all targets
    """
    if type(target) != list:
        target = list(target)
    for i in target:
        if type_of_target(df[i])!='binary':
            raise ValueError('Type of target must be binary')
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


def get_od_label(df, od_target, col='overdue_day', drop_od_col=False, silent=False, event=1):
    """
    Calculate samples' different delayed days of repayment and their overdue rates.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    col: str
            Column representing delay days of repayment
    od_target: list
            List of delay days to calculate boolean labels
    drop_od_col: boolean
            If True, drop the column representing delay days of repayment
    silent: boolean
            If True, restrict the printof positve rates
    event: int
            Value of target variable that stands for the event to predict

    Returns
    ----------
    df: pd.DataFrame
            Dataframe of added boolean labels
    """
    try:
        for i in od_target:
            df['d{i}'.format(i=i)] = df[col].map(lambda x: 1 - event if x < i + 1 else event)
    except:
        raise ValueError("overdue targets must be a list of integers")
    target = default_target(od_target)
    _ = get_pos_rate(df, target, silent=silent, event=event)
    if drop_od_col:
        df = df.drop([col], axis=1)
    return df


def feats_coverage_stat(df, target, is_ratio_pct=True, sort_value=True, silent=True, event=1):
    """
    Calculate positive rates of samples covered and uncovered by each feature.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    target: list
            List of default boolean targets
    is_ratio_pct: boolean
            If True, dispay ratios in percentage format
    sort_value: boolean
            If True, sort values
    event: int
            Value of target variable that stands for the event to predict
    silent: boolean
            If True, restrict the print of outputs

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


def sample_coverage_stat(df, target, exclude_list=[], silent=True, event=1):
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
            If True, restrict the print of outputs
    event: int
            Value of target variable that stands for the event to predict

    Returns
    ----------
    stat: dataframe of statistical summary   
    """
    col = list(set(df) - set(exclude_list))
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


# def feats_coverage_chi_test(df, target=TARGET, thr=0.01, event=1, smoothing_param=0.0001):
#     """
#     Abandoned
#     """
#     res = pd.DataFrame()
#     rowcnt = len(df)
#     col = [i for i in (set(df.columns) - set(target))]
#     for s in col:
#         feature_add = pd.DataFrame()
#         for t in target:
#             feature_covered_rate = (df[df[s].isnull()][t] == event).sum() / rowcnt
#             feature_uncovered_rate = (df[-df[s].isnull()][t] == event).sum() / rowcnt
#             num1 = (df[df[s].isnull()][t] == 1 - event).sum() + smoothing_param
#             num2 = (df[df[s].isnull()][t] == event).sum() + smoothing_param
#             num3 = (df[-df[s].isnull()][t] == 1 - event).sum() + smoothing_param
#             num4 = (df[-df[s].isnull()][t] == event).sum() + smoothing_param
#             matrix = np.array([[num1, num2], [num3, num4]])
#             feature_target_add = pd.DataFrame([feature_covered_rate,
#                                               feature_uncovered_rate,
#                                               np.NaN],
#                                              index=[s + '_cover', s + '_uncover', 'sig'],
#                                              columns=[str(t)]).T
#             feature_target_add['sig'] = True if stats.chisquare(matrix)[1] < thr else False
#             feature_add = pd.concat([feature_add, feature_target_add], axis=1)
#         res = pd.concat([res, feature_add], axis=0)
#     return np.round(res, 3)
