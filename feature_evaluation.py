import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd;

mpl.rcParams["axes.unicode_minus"] = False
mpl.style.use('ggplot')

import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from itertools import combinations
from scipy.stats import norm

from tools import *
from feature_encoding import woe, get_cut_points, UD_BinEncoder
from model_training import xgbt
from model_evaluation import model_summary


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 10. FEATURE EVALUATION ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def get_iv(df, target=TARGET, trimmr=None, cav_list_appd=[], cov_list_appd=[], exclude_list=[],
           bins=10, woe_min=-20, woe_max=20, **kwargs):
    """
    Calculate information value (IV) for each feature according to each target

    : params df: the input dataframe
    : params target: list of default boolean targets
    : params cav_list_appd: extra categorical variables to be added
    : params cov_list_appd: extra cotinuous variables to be added
    : params exclude_list: columns to be excluded
    : params bins: number of bins
    : param woe_min: minimum of woe value
    : param woe_max: maximum of woe value
    """
    df_copy = df.copy().apply(pd.to_numeric, errors='ignore')
    if trimmr is not None:
        try:
            df_copy = trimmr.ud_transform(df_copy)

        except:
            raise ValueError("The trimmer is not valid")

    if type(target) == str:
        target = [target]

    cav_list = [i for i in df.select_dtypes(include=[object]).columns if i not in exclude_list + target]
    cov_list = [i for i in df.select_dtypes(exclude=[object]).columns if i not in exclude_list + target]
    df_iv = pd.DataFrame(index=cav_list + cov_list, columns=['iv_for_' + str(t) for t in target])
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
            be = UD_BinEncoder(bins=bins, cut_method='dt', labels=None)
            be.ud_fit(df_copy[col], df_copy[t], prefix='')
            x_tmp = be.ud_transform(df_copy[col])
            cut_points = be.cut_points
            dmap, pp_map, np_map = woe(x_tmp, df_copy[t], woe_min, woe_max)
            iv_list += [sum([(pp_map.get(i[0]) - np_map.get(i[0])) * dmap.get(i[0]) for i in dmap.items()])]

        df_iv['iv_for_' + str(t)] = iv_list

    return np.round(df_iv.sort_values(by=df_iv.columns[0], ascending=False), 3)


def plot_corr(df):
    """
    Plot correlation map

    : params df: the input dataframe
    """
    f, ax = plt.subplots(figsize=[0.5 * df.shape[1], 0.4 * df.shape[1]])
    corr = df.corr()
    sns.heatmap(corr,
                mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True,
                ax=ax)


def get_vif_cor(df, target=TARGET, plot=False):
    """
    Calculate vifs and correlations of columns in the dataframe

    : params df: the input dataframe
    : params target: list of default boolean targets
    : params plot: whether to plot correlation map

    :return: vif and correlation of features in the dataframe
    """
    col = [i for i in (set(df.columns) - set(target))]
    df_tmp = df[col].select_dtypes(exclude=[object])
    df_tmp['intercept'] = 1
    vif = pd.DataFrame(index=df_tmp.columns)
    vif["vif"] = [variance_inflation_factor(df_tmp.values, i) for i in range(df_tmp.shape[1])]
    vif = vif.sort_values('vif', ascending=False)
    vif = vif.drop(['intercept'])
    cor = df.corr()

    if plot == True:
        print("Correlation map of features\n")
        plot_corr(df)

    return np.round(vif, 3), np.round(cor, 3)


def plot_hist(df, feat, bins=20, ax=None):
    """
    Plot histogram

    : params df: the input dataframe
    : params feat: feature to be plot
    : params bins: number of bins
    """
    df = df.dropna(subset=[feat])
    (mu, sigma) = norm.fit(df[feat])
    # fit a normally distributed curve
    bins = min(min(bins, df[feat].nunique()) * 2, 100)

    if not ax:
        f, ax = plt.subplots(dpi=100)

    n, bins, patches = ax.hist(df[feat], bins, density=True, facecolor='orange', alpha=0.75)
    y = norm.pdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=2)
    plt.ylabel('Probability')
    plt.xlabel(feat)


def plot_hist_all(df, target=TARGET, bins=20):
    """
    Plot histograms for all features in the dataframe

    : params df: the input dataframe
    : params bins: number of bins
    """
    width = int(df.shape[1]) + 1
    fig_inx = 1
    fig = plt.figure(figsize=(30, 6 * width))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.3)
    col = [i for i in df.select_dtypes(exclude=[object]).columns if i not in set(target)]

    for i in col:
        ax = fig.add_subplot(width, 5, fig_inx)
        plot_hist(df, i, bins=bins, ax=ax)
        fig_inx += 1


def feature_combination(df, target=TARGET, num_boost_round=1000, params=XGB_PARAMS, pos_label=1,
                        exclude_list=[]):
    """
    Run through all combination of features to get best model 
    (Warning: only used for comparisons of integrated scores)

    : params df: the input dataframe
    : params target: list of default boolean targets
    : params num_boost_round: num_boost_round
    : params params: xgb parameters
    : params pos_label: positive label
    """
    if df.shape[1] >=10:
        raise Exception("To many features")

    col = [i for i in (set(df.columns) - set(target + exclude_list))]
    x = df[col]
    y = df[target]
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

#     : params df: dataframe
#     : params target: list of default boolean targets
#     : params reverse_col: columns to be reverse (we assume that all scores are the higher the better)
#     : params bins: number of bins
#     : params plot_all: plot all features' sorting ability in one figure
#     : params plot_num: wheter plot numbers
#     : params figsize: figure size

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