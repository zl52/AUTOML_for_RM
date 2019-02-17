import pandas as pd
import numpy as np
from collections import defaultdict, Counter


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################  2. DATA AGGREGATION   ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def get_name(col_list, prefix):
    return [prefix + str(i) for i in col_list]


def get_value(val_list, mult_val):
    return [i * mult_val for i in val_list]


def agg_statistics(df, uid, value, agg_func, suffix=''):
    """
    Aggregate data by defined aggregate key and time span

    : params df: the dataframe of current or latest samples we concern
    : params uid: column(s) by which history data are aggregated
    : params value: column(s) to be summaized
    : params agg_func: aggregation functions
    : params suffix: suffix

    : return tmp: a new feature describing an aggregate statistics

    Example:
        tmp = agg_statistics(df, ['uid', 'uid_trans_period_day'], ['history_trans_amount'], [np.sum, np.mean])
    """
    suffix = '_' + suffix if suffix else suffix
    tmp = df[uid + value].groupby(uid).agg(agg_func)
    tmp.columns = ['_'.join(col) for col in tmp.columns]
    tmp.columns = [col + suffix for col in tmp.columns]

    return tmp.reset_index(drop=False)


def span_statistics(df, df_history, uid_key, agg_key, value, span_key, span_dic, agg_func):
    """
    Apply RFM rule to do feature engineering where samples are aggregated by defined aggregate keys 
    and limitted by defined time spans in combinations

    : params df: the dataframe of current or latest samples we concern
    : params df_history: history data of concened samples
    : params uid_key: unique key for samples we concern
    : params agg_key: column(s) by which history data are aggregated
    : params value: column(s) to be summaized
    : params span_key: column by calculating current-history time difference
    : params span_dics: time-span(s) dictionary used to limit time span when aggregating history data
    : params agg_func: aggregation functions

    : return final_df: the dataframe of features derived from aggregate statistical summary

    Example:
        tmp = span_statistics(df, df_history, ['uid'], ['uid_trans_hour', 'uid_trans_period_day'], 
                              ['history_amount', 'history_current_amount_diff'], 
                              'history_current_time_diff', {'5m': 300, '30m' : 1800}, [np.sum, np.mean])
    """
    final_df = df[uid_key + agg_key].drop_duplicates()

    for k, v in span_dic.items():
        df_tmp = df_history[df_history[span_key] <= v]

        for uid in agg_key:
            print('>>> uid', uid, '\t\t span', k)
            print(df_tmp[uid].nunique())
            df_agg = agg_statistics(df_tmp, uid_key + [uid], value, agg_func, k + '_by_' + uid)
            final_df = pd.merge(final_df, df_agg, how='left', on=uid_key + [uid])

    return final_df


def nunique_stats(df, df_history, uid_key, value, feat):
    """
    Derive a new feature by calculating each uid_key's number of unique values in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    add = pd.DataFrame(df_history.groupby(uid_key)[value].nunique()).reset_index()
    add = add.rename(columns={value: feat})
    df = pd.merge(df, add, on=uid_key, how='left')

    return df


def cnt_stats(df, df_history, uid_key, value, feat):
    """
    Derive a new feature by calculating each uid_key's notnull count in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    add = pd.DataFrame(df_history.groupby(uid_key)[value].count()).reset_index()
    add = add.rename(columns={value: feat})
    df = pd.merge(df, add, on=uid_key, how='left')

    return df


def max_stats(df, df_history, uid_key, value, feat):
    """
    Derive a new feature by calculating each uid_key's maximum in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    add = pd.DataFrame(df_history.groupby(uid_key)[value].max()).reset_index()
    add = add.rename(columns={value: feat})
    df = pd.merge(df, add, on=uid_key, how='left')

    return df


def min_stats(df, df_history, uid_key, value, feat):
    """
    Derive a new feature by calculating each uid_key's minimum in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    add = pd.DataFrame(df_history.groupby(uid_key)[value].min()).reset_index()
    add = add.rename(columns={value: feat})
    df = pd.merge(df, add, on=uid_key, how='left')

    return df


def any_stats(df, df_history, uid_key, value, feat, certain_value):
    """
    Derive a new feature by calculating each uid_key whether to have a certain value in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    add = pd.DataFrame(df_history.groupby(uid_key)[value] \
                       .apply(lambda x: 1 if len(np.where(x == certain_value)[0]) != 0 else 0)) \
        .reset_index()
    add = add.rename(columns={value: feat})
    df = pd.merge(df, add, on=uid_key, how='left')

    return df


def any_cnt_stats(df, df_history, uid_key, value, feat, certain_value):
    """
    Derive a new feature by calculating each uid_key's maximum in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    add = pd.DataFrame(df_history.groupby(uid_key)[value] \
                       .apply(lambda x: len(np.where(x == certain_value)[0]))).reset_index()
    add = add.rename(columns={value: feat})
    df = pd.merge(df, add, on=uid_key, how='left')

    return df


def most_frequent(x):
    """
    Calculate most frequent number or string

    : params x: the input column
    : return: most frequent number or string
    """
    return Counter(x).most_common()[0][0]


def mode_stats(df, df_history, uid_key, value, feat):
    """
    Derive a new feature by calculating each uid_key's mode in the column we concern

    : params df: current or latest samples we concern
    : params df_history: history data of concened sample
    : params uid_key: unique key for the samples we concern
    : params value: column(s) to be summaized
    : params feat: column name of the new feature

    : return tmp: a new feat derived from an aggregate statistics
    """
    if type(df_history[value].iloc[0]) == str:
        add = pd.DataFrame(df_history.groupby(uid_key)[value].apply(most_frequent)).reset_index()
        add = add.rename(columns={value: feat})
        df = pd.merge(df, add, on=uid_key, how='left')

    elif type(data[value].iloc[0]) != object:
        add = pd.DataFrame(df_history.groupby(uid_key)[value] \
                           .apply(lambda x: stats.mode(x, nan_policy='omit')[0][0])) \
            .reset_index()
        add = add.rename(columns={value: feat})
        df = pd.merge(df, add, on=uid_key, how='left')

    else:
        print('Can not handle type Object')

    return df
