import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["axes.unicode_minus"] = False
mpl.style.use('ggplot')

import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from itertools import combinations
from scipy.stats import norm
import math

from tools import *
from feature_encoding import woe, get_cut_points, BinEncoder
from model_training import xgbt
from model_evaluation import model_summary


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 10. FEATURE EVALUATION ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def get_iv(df, target, trimmr=None, cav_appd_list=[], cov_appd_list=[], exclude_list=[],
           bins=10, woe_min=-20, woe_max=20, **kwargs):
    """
    Calculate information value (IV) for each feature according to each target.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    target: list or str
            List of default boolean targets or target used as label in the modeling process
    cav_appd_list: list
            Extra categorical variables to be added
    cov_appd_list: list
            Extra cotinuous variables to be added
    exclude_list: list
            Features to be excluded
    bins: int
            Number of bins
    woe_min: float or int
            Minimum of woe value
    woe_max: float or int
            Maximum of woe value

    Returns
    ----------
    df_iv: pd.DataFrame
            Dataframe of iv according to different target
    """
    df_copy = df.copy().apply(pd.to_numeric, errors='ignore')
    if trimmr is not None:
        try:
            df_copy = trimmr.transform(df_copy)
        except:
            raise ValueError("The trimmer is not valid")
    if type(target) != list:
        target = [target]
    for i in target:
        if type_of_target(df[i])!='binary':
            raise ValueError("Label must be binary for computation of information value")
    cav_list = list(set(df.select_dtypes(include=[object])) - set(target + exclude_list))
    cov_list = list(set(df.select_dtypes(exclude=[object])) - set(target + exclude_list))
    df_iv = pd.DataFrame(index=cav_list + cov_list, columns=['iv_for_%s'%t for t in target])
    df_copy[cav_list] = df_copy[cav_list].fillna('NA')
    df_copy[cov_list] = df_copy[cov_list].apply(lambda x: x.fillna(x.mean()), axis=1)
    for t in target:
        iv_list = []
        for col in cav_list:
            dmap, pp_map, np_map = woe(df_copy[col], df_copy[t], woe_min, woe_max)
            x_tmp = [dmap[i] for i in df_copy[col]]
            cut_points = get_cut_points(x_tmp, df_copy[t], cut_method='dt', **kwargs)
            x_tmp = pd.cut(x_tmp, cut_points)
            dmap2, pp_map2, np_map2 = woe(x_tmp, df_copy[t], woe_min, woe_max)
            iv_list += [sum([(pp_map2.get(i[0]) - np_map2.get(i[0])) * dmap2.get(i[0]) for i in dmap2.items()])]
        for col in cov_list:
            be = BinEncoder(bins=bins, cut_method='dt', labels=None)
            be.fit(df_copy[col], df_copy[t], prefix='')
            x_tmp = be.transform(df_copy[col])
            cut_points = be.cut_points
            dmap, pp_map, np_map = woe(x_tmp, df_copy[t], woe_min, woe_max)
            iv_list += [sum([(pp_map.get(i[0]) - np_map.get(i[0])) * dmap.get(i[0]) for i in dmap.items()])]
        df_iv['iv_for_' + str(t)] = iv_list
    return np.round(df_iv.sort_values(by=df_iv.columns[0], ascending=False), 3)


def plot_corr(df):
    """
    Plot correlation map.

    Parameters
    ----------
    df: pd.DataFrame
            The input dataframe
    """
    f, ax = plt.subplots(figsize=[0.5 * df.shape[1], 0.4 * df.shape[1]])
    corr = df.corr()
    sns.heatmap(corr,
#             mask=np.zeros_like(corr, dtype=np.bool),
#             cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            ax=ax)


def get_vif_cor(df, target=[], exclude_list=[], plot=False):
    """
    Calculate vifs and correlations of features in the dataframe.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    target: list
            List of default boolean targets
    exclude_list: list
            Columns to be excluded when calculating correlation
    plot: boolean
            If True, plot correlation map

    Returns
    ----------
    vif: Dataframe
            Vifs of features in the dataframe
    cor: Dataframe
            Correlations of features in the dataframe
    """
    col = list(set(df) - set(target) - set(exclude_list))
    df_tmp = df[col].select_dtypes(exclude=[object])
    df_tmp['intercept'] = 1
    vif = pd.DataFrame(index=df_tmp.columns)
    vif["vif"] = [variance_inflation_factor(df_tmp.values, i) for i in range(df_tmp.shape[1])]
    vif = vif.sort_values('vif', ascending=False)
    vif = vif.drop(['intercept'])
    cor = df[col].select_dtypes(exclude=[object]).corr()
    if plot:
        print("Correlation map of features\n")
        plot_corr(df[col].select_dtypes(exclude=[object]))
    return np.round(vif, 3), np.round(cor, 3)


def remove_extr_outlier(df, feat, quantile=0.05, etra_ratio=1.2):
    """
    Remove outliers in the designated feature.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    feat: list
            List of features to be concerned
    quantile: float
            Designated quantiles
    etra_ratio: float
            Ratio times designated quantiles to define ouliers

    Returns
    ----------
    df_copy: Dataframe
            Dataframe with outliers removed
    """
    df_copy = df.copy()
    hb = np.quantile(df_copy[feat].dropna(), 1-quantile) * etra_ratio
    lb = np.quantile(df_copy[feat].dropna(), quantile) / etra_ratio
    df_copy.loc[df_copy[feat]>hb, feat] = hb
    df_copy.loc[df_copy[feat]<lb, feat] = lb
    return df_copy


def plot_hist(df, feat, ax=None, extr_outlier_removed=False):
    """
    Plot histogram.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    feat: str
            Feature to be plotted
    ax: matplotlib.axe
            Axe at which it plots
    extr_outlier_removed: boolean
            If True, remove outliers
    """
    if not ax:
        f, ax = plt.subplots()
    df_copy = df.copy()
    if extr_outlier_removed:
        df_copy = remove_extr_outlier(df_copy, feat)
    try:
        sns.distplot(df_copy[feat].dropna(), ax=ax)
    except:
        sns.distplot(df_copy[feat].dropna(), ax=ax, kde=False)
    ax.set(xlabel=feat, ylabel="Distribution", title= "Distribution for %s"%feat)


def plot_hist_all(df, target=[], exclude_list=[], extr_outlier_removed=False):
    """
    Plot histograms for all features in the dataframe.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    target: list
            List of default boolean targets
    exclude_list: list
            List excluded from being analyzed
    extr_outlier_removed: boolean
            If True, remove outliers
    """
    col = list(set(df.select_dtypes(exclude=[object])) - set(target) - set(exclude_list))
    height = math.ceil(len(col)/6)
    fig_inx = 0
    fig, axes = plt.subplots(height,
                             6 if len(col)>=6 else len(col),
                             figsize=(30 if len(col)>=6 else len(col) * 5, 6 * height))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.3)
    for i in col:
        if len(col)>=6:
            ax = axes[fig_inx//6][fig_inx%6]
        else:
            ax = axes[fig_inx]
        plot_hist(df, i, ax=ax, extr_outlier_removed=extr_outlier_removed)
        fig_inx += 1
        

def plot_joint_dist(df, feat, label, nunique_thr=10, ax=None, extr_outlier_removed=False):
    """
    Plot joint distribution for all features in the dataframe.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    feat: list
            List of default boolean targets
    label: str
            Label will be used in the modeling process
    nunique_thr: int
            Unique value < 10, draw barplot; Unique value >= 10, draw violinplot
    ax: matplotlib.axe
            Axe at which it plots
    extr_outlier_removed: boolean
            If True, remove outliers
    """
    if not ax:
        f, ax = plt.subplots()
    if extr_outlier_removed:
        df_copy = remove_extr_outlier(df, feat)
    if df[feat].nunique() < nunique_thr:
        sns.barplot(x=feat, y=label, data=df_copy, ax=ax)
        ax.set(title= 'barplot for %s'%feat)
    else:
        sns.violinplot(x=label, y=feat, data=df_copy, ax=ax)
        ax.set(title= 'violinplot for %s'%feat)


def plot_joint_dist_all(df, label, target=[], exclude_list=[], nunique_thr=10, extr_outlier_removed=False):
    """
    Plot joint distribution for all features in the dataframe.

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    label: str
            Label will be used in the modeling process
    target: list
            List of default boolean targets
    exclude_list: list
            List excluded from being analyzed
    nunique_thr: int
            If unique values < 10, draw barplot; Else if unique values >= 10, draw violinplot
    extr_outlier_removed: boolean
            If True, remove outliers
    """
    col = list(set(df.select_dtypes(exclude=[object])) - set(target) - set(exclude_list)) +  [label]
    bar_col = [i for i in col if df[i].nunique() < nunique_thr]
    height = math.ceil(len(bar_col)/6)
    fig_inx = 0
    fig, axes = plt.subplots(height,
                     6 if len(bar_col)>=6 else len(bar_col),
                     figsize=(30 if len(bar_col)>=6 else len(bar_col) * 5, 6 * height))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=None, hspace=0.3)
    for bc in bar_col:
        if len(bar_col)>=6:
            ax = axes[fig_inx//6][fig_inx%6]
        else:
            ax = axes[fig_inx]            
        plot_joint_dist(df, bc, label=label, nunique_thr=nunique_thr, ax=ax, extr_outlier_removed=extr_outlier_removed)
        fig_inx += 1
        vio_col = [i for i in col if df[i].nunique() >= nunique_thr]
    height = math.ceil(len(vio_col)/6)
    fig_inx = 0
    fig, axes = plt.subplots(height,
                     6 if len(vio_col)>=6 else len(bar_col),
                     figsize=(30 if len(vio_col)>=6 else len(vio_col) * 5, 6 * height))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=None, hspace=0.3)
    for vc in vio_col:
        if len(vio_col)>=6:
            ax = axes[fig_inx//6][fig_inx%6]
        else:
            ax = axes[fig_inx]
        plot_joint_dist(df, vc, label=label, nunique_thr=nunique_thr, ax=ax, extr_outlier_removed=extr_outlier_removed)
        fig_inx += 1


def feature_combination(df, label, num_boost_round=1000, params=XGB_PARAMS, pos_label=1,
                        exclude_list=[]):
    """
    Try out all combination of features in order to get best model.
    (Warning: only used for comparisons of integrated scores)

    Parameters
    ----------
    df: pd.Dataframe
            The input dataframe
    label: str
            Label will be used in the modeling process
    num_boost_round: int
            num_boost_round
    params: dict
            xgb parameters
    pos_label: int
            Event denotes positive label

    Returns
    ----------
    res: pd.Dataframe
            Dataframe recording result (auc and ks)
    """
    if df.shape[1] >=10:
        raise Exception("Too many combinations to iterate")
    col = list(set(df.columns) - set([label] + exclude_list))
    x = df[col]
    y = df[label]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    xgb_res = pd.DataFrame()
    for i in range(2, x.shape[1] + 1):
        print(i)
        cc = list(combinations(x.columns, i))
        for j in range(len(cc)):
            print(list(cc[j]))
            model, pred_train_value, pred_val_value = xgbt(x_train[list(cc[j])]
                                                           , y_train
                                                           , x_val[list(cc[j])]
                                                           , y_val
                                                           , None
                                                           , params=params
                                                           , num_boost_round=num_boost_round
                                                           , early_stopping_rounds=50
                                                           , make_prediction=True)
            add = model_summary(pred_train_value, y_train, pred_val_value, y_val,
                                pos_label=pos_label, use_formater=False, plot=False)
            add['combination'] = '+'.join(list(cc[j]))
            add = add.reset_index().set_index('combination')
            xgb_res = pd.concat([xgb_res, add], axis=0)
    if len(col) == 2:
        return xgb_res
    else:
        train_res = xgb_res.groupby(['index', 'combination']).sum().loc['train']
        val_res = xgb_res.groupby(['index', 'combination']).sum().loc['val']
        train_res = train_res.rename(columns={'auc': 'train_auc', 'ks': 'train_ks'})
        val_res = val_res.rename(columns={'auc': 'val_auc', 'ks': 'val_ks'})
        res = pd.concat([val_res, train_res], axis=1)
        res = res.sort_values(by='val_auc', ascending=False)
        return res[['val_auc', 'val_ks', 'train_auc', 'train_ks']]


# def kendall(df, target, bins=10):
#     """
#     abandoned
#     """
#     res = pd.DataFrame()
#     col = list(set(df.columns) - set(target))
#     stats = sorting_stat(df, target, bins=bins)

#     for i in target:
#         index = pd.MultiIndex.from_product([[i], ['min', 'max', 'kendall']], names=['target', 'keys'])
#         res_tmp = pd.DataFrame(index=index)

#         for c in col:
#             try:
#                 df = pd.DataFrame()
#                 df[i] = stats[c][c + '_' + i]
#                 df['group'] = list(range(10, 0, -1))
#                 min_value = df.min()[i]
#                 max_value = df.max()[i]
#                 kd = df.corr(method='kendall')[i][1]
#                 res_tmp[s] = [min_value, max_value, kd]

#             except:
#                 pass

#         res = pd.concat([res, res_tmp])

#     res = np.abs(np.round(res, 2))
#     kd = res.xs('kendall', level='keys').T

#     return kd


# def sorting_stat(df, target, bins=10):
#     """
#     abandoned
#     """
#     col = [i for i in (set(df.columns) - set(target) - set(df.select_dtypes(include=[object]).columns))]
#     stats = {}

#     for c in col:
#         res_col = pd.DataFrame()
#         tmp = df.dropna(subset=[c])
#         tmp['range'] = pd.qcut(tmp[c], q=np.linspace(0, 1, bins + 1),
#                                duplicates='drop', precision=0, retbins=True)[0]
#         print('>>> number of bins:', tmp['range'].nunique())

#         for n in target:
#             res = tmp.groupby('range')[n].value_counts(normalize=True, sort=False).xs(1, level=n)
#             res = res.to_frame(name=c + '_' + n)
#             res_col = pd.concat([res_col, res], axis=1)

#         stats.update({c: np.round(res_col, 2)})

#         return stats


# def sorting_plot_for_scores(df, target, reverse_col=[], bins=10, plot_all=True, plot_num=False,
#                             figsize=(7, 5)):
#     """
#     Plot figures showing each scores's sorting ability

#     df: dataframe
#     target: list of default boolean targets
#     reverse_col: columns to be reverse (we assume that all scores are the higher the better)
#     bins: number of bins
#     plot_all: plot all features' sorting ability in one figure
#     plot_num: wheter plot numbers
#     figsize: figure size

#     """
#     col = [i for i in (set(df.select_dtypes(exclude=[object]).columns) - set(target))]
#     tmp = df.copy()
#     stats_list = []

#     if reverse_col is not []:
#         for s in reverse_col:
#             tmp['neg_' + s] = tmp[s].max() - tmp[s]
#     for i in target:
#         if plot_all == True:
#             f = plt.figure(figsize=figsize, dpi=100)
#             ax1 = f.add_subplot(111)
#         for s in col:
#             if plot_all == False:
#                 f = plt.figure(figsize=figsize, dpi=100)
#                 ax1 = f.add_subplot(111)
#             if s in reverse_col:
#                 try:
#                     tmp[s + '_bin_range'] = pd.qcut(tmp[s].dropna(), q=np.linspace(0, 1, bins + 1),
#                                                     precision=0, retbins=True)[0]
#                     tmp[s + 'neg_bin_range'] = pd.qcut(tmp['neg_' + s].dropna(),
#                                                        q=np.linspace(0, 1, bins + 1),
#                                                        precision=0, retbins=True)[0]
#                     stats = bin_stat(tmp, s + 'neg_bin_range', i, bins, False, 'qcut')
#                     stats2 = bin_stat(tmp, s + '_bin_range', i, bins, False, 'qcut')
#                     stats_list += [stats2]
#                 except:
#                     print("{s} can't be cut by 10".format(s=s))
#             else:
#                 try:
#                     tmp[s + '_bin_range'] = pd.qcut(tmp[s].dropna(), q=np.linspace(0, 1, bins + 1),
#                                                     precision=0, retbins=True)[0]
#                     print(haha)

#                     stats = stats2 = bin_stat(tmp, s + '_bin_range', i, bins, False, 'qcut')
#                     stats_list += [stats2]
#                 except:
#                     print("{s} can't be cut by {b}".format(s=s, b=bins))

#             xx = range(1, bins + 1)
#             if plot_all == False:
#                 try:
#                     plt.bar(xx, stats['counts'], width=0.5, color='orange', yerr=0.000001)
#                     plt.xticks(xx, list(range(1, 1 + bins)))
#                     plt.ylabel('number')
#                     ax1.set_ylim([0, max(stats['counts']) * 1.5])
#                     plt.title('{}\'s sorting ability for {}'.format(s, i))
#                     ax2 = ax1.twinx()
#                     plt.plot(xx, stats['accum' + i], linestyle='--', marker='o', markersize=5,
#                              target='accum {}'.format(s))
#                     if plot_num == True:
#                         for a, b in zip(xx, stats['accum' + i]):
#                             plt.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=7)
#                 except:
#                     pass
#             try:
#                 plt.plot(xx, stats[i], linestyle='--', marker='o', markersize=5, target=s)
#                 plt.ylabel('overdue rate')
#                 plt.xlabel('groups(bad -> good)')
#                 if plot_num == True:
#                     for a, b in zip(xx, stats[i]):
#                         plt.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=7)

#                 if plot_all == False:
#                     plt.legend(loc='upper right')
#                     plt.show()
#             except:
#                 pass

#         if plot_all == True:
#             plt.title('comparison of sorting ability for {}'.format(i))
#             plt.legend(loc='upper right')
#             plt.show()


# #    return stats_list