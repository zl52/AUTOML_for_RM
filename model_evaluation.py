import pandas as pd;
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["axes.unicode_minus"] = False
mpl.style.use('ggplot')
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, make_scorer, f1_score, fbeta_score, precision_score, \
    roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.utils.multiclass import type_of_target

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################  13. MODEL EVALUATION  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################

def get_mae(depvar_value, pred_label):
    """
    Calculate mae for predicted dataset.

    : params depvar_value: real label
    : params pred_label: predicted label
    """
    mae = metrics.mean_absolute_error(depvar_value,pred_label)
    return mae


def get_mse(depvar_value, pred_label):
    """
    Calculate mse for predicted dataset.

    : params depvar_value: real label
    : params pred_label: predicted label
    """
    mse = metrics.mean_squared_error(depvar_value,pred_label)
    return mse


def get_rmse(depvar_value, pred_label):
    """
    Calculate rmse for predicted dataset.

    : params depvar_value: real label
    : params pred_label: predicted label
    """
    rmse = np.sqrt(metrics.mean_squared_error(depvar_value,pred_label))
    return rmse


def get_msle(depvar_value, pred_label):
    """
    Calculate msle for predicted dataset.

    : params depvar_value: real label
    : params pred_label: predicted label
    """
    rmse = metrics.mean_squared_log_error(depvar_value,pred_label)
    return rmse


def get_precision(depvar_value, pred_label, pos_label=1):
    """
    Calculate precision rate for predicted dataset.

    : params pred_label: real label
    : params pred_label: predicted label
    : params pos_label: positive label(0 or 1)
    """
    precision = metrics.precision_score(depvar_value, pred_label, pos_label)
    return precision


def get_false_neg_rate(depvar_value, pred_label, pos_label=1):
    """
    Calculate false negative rate (eg missed fraudulent cases) for predicted dataset.

    : params depvar_value: real label
    : params pred_label: predicted label
    : params pos_label: positive label(0 or 1)
    """
    tn, fp, fn, tp = metrics.confusion_matrix(depvar_value, pred_label).ravel()
    miss_neg_rate = fn / (fn + tn)
    # 误报为负占所有预测为负的比例
    return miss_neg_rate


def get_auc_ks(depvar_value, pred_value, pos_label=1):
    """
    Calculate AUC and KS for predicted dataset.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params pos_label: positive label(0 or 1)
    """
    fpr, tpr, thr = metrics.roc_curve(depvar_value, pred_value, pos_label=pos_label)
    ks_list = tpr - fpr
    indx = np.argmax(ks_list)
    thr_point = thr[indx]

    if pos_label == 0:
        pred_value = 1 - pred_value

    pred_label = pd.Series(pred_value).map(lambda x: 1 if x > thr_point else 0)
    ac = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    return pred_label, ac, ks, thr_point, fpr, tpr


def get_pos_neg_rate(pred_label):
    """
    Calculate positive rate and negative rate for predicted dataset.

    : params pred_label: predicted label
    """
    neg_rate = sum(pred_label == 0) / len(pred_label)
    pos_rate = sum(pred_label == 1) / len(pred_label)
    return neg_rate, pos_rate


def get_f_beta(depvar_value, pred_label, pos_label, beta):
    """
    Calculate positive rate and negative rate for predicted dataset.

    : params depvar_value: real label
    : params pred_label: predicted label
    : params pos_label: positive label(0 or 1)
    : params beta: beta for calculate f_beta
    """
    f_beta = metrics.fbeta_score(depvar_value, pred_label, beta=beta)
    return f_beta


def get_clf_statistics(pred_value, depvar_value, pos_label=1, beta=1, extra_stat=False, silent=True):
    """
    Calculate all statistics for predicted value if label is binary.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params pos_label: positive label(0 or 1)
    : params beta: beta for calculate f_beta
    : params extra_stat: determine if extra statistics is needed
    : params silent: If True, restrict prints
    """
    try:
        pred_label, ac, ks, thr_point, fpr, tpr = get_auc_ks(depvar_value, pred_value, pos_label)
    except Exception as e:
        pred_label, ac, ks, thr_point, fpr, tpr = None, None, None, None, None, None
        print(e)

    try:
        miss_neg_rate = get_false_neg_rate(depvar_value, pred_label, pos_label)
    except Exception as e:
        miss_neg_rate = None
        print(e)

    try:
        neg_rate, pos_rate = get_pos_neg_rate(pred_label)
    except Exception as e:
        neg_rate, pos_rate = None, None
        print(e)

    if extra_stat:
        try:
            recall = get_recall(depvar_value, pred_label, pos_label)
        except Exception as e:
            recall = None
            print(e)

        try:
            precision = get_precision(depvar_value, pred_label, pos_label)
        except Exception as e:
            precision = None
            print(e)

        try:
            f_beta = get_f_beta(depvar_value, pred_label, pos_label, beta)
        except Exception as e:
            f_beta = None
            print(e)

        if not silent:
            print('outputs are auc, ks, threshold, negativerate, missing_negative_rate, recall rate, \
                   precision rate, f_beta, fpr, tpr')
        return ac, ks, thr_point, pos_rate, neg_rate, miss_neg_rate, recall, precision, f_beta, fpr, tpr

    else:
        if not silent:
            print('outputs are auc, ks, threshold, negativerate, missing_negative_rate, fpr, tpr')
        return ac, ks, thr_point, neg_rate, miss_neg_rate, fpr, tpr


def get_reg_statistics(pred_value, depvar_value, silent=True):
    """
    Calculate all statistics for predicted value if label is continuous.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params silent: If True, restrict prints
    """
    try:
        mae = get_mae(depvar_value, pred_value)
    except Exception as e:
        mae = None
        print(e)

    try:
        mse = get_mse(depvar_value, pred_value)
    except Exception as e:
        mse = None
        print(e)

    try:
        msle = get_msle(depvar_value, pred_value)
    except Exception as e:
        msle = None
        print(e)

    try:
        rmse = get_rmse(depvar_value, pred_value)
    except Exception as e:
        rmse = None
        print(e)

    if not silent:
        print('outputs are mae, mse, msle, rmse')
    return mae, mse, msle, rmse


def get_xgb_fi(model, method='interaction', alpha=0.7, top=20):
    """
    Get feature importance from xgb model.

    : params model: input model
    : params method: method for choosing features
    : params alpha: weight parameter for importance_type 
                    (larger the alpha, larger the weight for importance_type = gain)
    : params top: number of top features from each importance_type
    """
    importance_by_weight = model.get_score(importance_type='weight')
    importance_by_gain = model.get_score(importance_type='gain')
    importance_by_weight = pd.DataFrame(importance_by_weight, index=range(1)).T
    importance_by_weight.columns = {'weight'}
    importance_by_weight = importance_by_weight.sort_values(by='weight', ascending=False)
    importance_by_weight['weight_rank'] = range(1, len(importance_by_weight) + 1)
    importance_by_gain = pd.DataFrame(importance_by_gain, index=range(1)).T
    importance_by_gain.columns = {'gain'}
    importance_by_gain = importance_by_gain.sort_values(by='gain', ascending=False)
    importance_by_gain['gain_rank'] = range(1, len(importance_by_gain) + 1)
    fi = pd.concat([importance_by_weight, importance_by_gain], axis=1) \
        .sort_values(by='gain_rank', ascending=True)
    if method == 'interaction':
        col1 = fi.sort_values(by='gain_rank', ascending=True).head(top).index.tolist()
        col2 = fi.sort_values(by='weight_rank', ascending=True).head(top).index.tolist()
        col = col1 + list(set(col2) - set(col1))
        return fi, col

    elif method == 'rank':
        fi['rank'] = fi['gain_rank'] * alpha + fi['weight_rank'] * (1 - alpha)
        fi = fi.sort_values(by='rank', ascending=True)
        col = fi.head(top).index.tolist()
        return fi, col

    else:
        print('wrong method choice. (rank or interaction)')


def bin_stat(pred_, bin_range, target, bins=None, reverse=False, how='qcut', with_label=True, extra_stat=True):
    """
    Get bins' statistics.

    : params pred_: predicted value
    : params bin_range: col of bin ranges
    : params target: label of interest
    : params reverse: If True, reverse the ordering of bin ranges
    : params how: how to cut the range
    : params with_label: with label or not
    : params extra_stat: If True, calculate extra statistics
    """
    if bins is None:
        bins=pred_[bin_range].nunique()
        
    if with_label == True:
        res = (pred_.groupby(bin_range)[target].sum() / pred_.groupby(bin_range)[target].count()).to_frame()
        res = res.rename(columns={target: 'pos_rate'})
        res['count'] = pred_[bin_range].value_counts(sort=False)
        res['pos_count'] = res['count'] * res['pos_rate']
        res['expected_count'] = res['count'] * pred_[target].mean()
        res['count_proportion'] = pred_[bin_range].value_counts(normalize=True, sort=False)
        res['recall_num'] = pred_.groupby(bin_range)[target].apply(lambda x: sum(x == 0))
        res['recall_rate'] = pred_.groupby(bin_range)[target].apply(
                                          lambda x: sum(x == 0) / sum(pred_[target] == 0))
        res = res.fillna(0)
        if extra_stat:
            if reverse == False:
                if how == 'qcut':
                    res['accum_pos_rate'] = [np.mean(res['pos_rate'].iloc[:i]) for i in range(1, bins + 1)]
                    res['accum_count'] = [np.sum(res['count_proportion'].iloc[:i]) for i in range(1, bins + 1)]
                    res['accum_recall_rate'] = [np.sum(res['recall_rate'].iloc[:i]) for i in range(1, bins + 1)]
                elif how == 'cut':
                    res['accum_pos_rate'] = [(res['pos_rate'][:i] * res['count'].iloc[:i]).sum() \
                                             / res['count'][:i].sum() for i in range(1, bins + 1)]
                    res['accum_count'] = [np.sum(res['count_proportion'].iloc[:i]) for i in range(1, bins + 1)]
                    res['accum_recall_rate'] = [np.sum(res['recall_num'][:i]) / sum(pred_[target] == 0)
                                                for i in range(1, bins + 1)]
                else:
                    raise ValueError("\"how\" must chosen between qcut and cut")

            else:
                if how == 'qcut':
                    res['accum_pos_rate'] = [np.mean(res['pos_rate'].iloc[i:]) for i in range(bins)]
                    res['accum_count'] = [np.sum(res['count_proportion'].iloc[i:]) for i in range(bins)]
                    res['accum_recall_rate'] = [np.sum(res['recall_rate'].iloc[i:]) for i in range(bins)]
                elif how == 'cut':
                    res['accum_pos_rate'] = [(res['pos_rate'].iloc[i:] * res['count'].iloc[i:]).sum() \
                                             / res['count'].iloc[i:].sum() \
                                             for i in range(bins)]
                    res['accum_count'] = [np.sum(res['count_proportion'].iloc[i:]) for i in range(bins)]
                    res['accum_recall_rate'] = [np.sum(res['recall_rate'].iloc[i:]) for i in range(bins)]
                else:
                    raise ValueError("\"how\" must chosen between qcut and cut")
    else:
        res = pred_.groupby(bin_range)[bin_range].count().to_frame()
        res['count_proportion'] = pred_[bin_range].value_counts(normalize=True, sort=False)
        res = res.rename(columns={'bin_range': 'count'})

        if not extra_stat:
            if reverse == False:
                res['accum_count'] = [np.sum(res['count_proportion'].iloc[:i]) for i in range(1, bins + 1)]

            else:
                res['accum_count'] = [np.sum(res['count_proportion'].iloc[i:]) for i in range(bins)]

    res = res.fillna(0)
    return np.round(res, 4)


def plot_ks_curve(depvar_value, pred_value, ax=None, pos_label=1, reverse=False):
    """
    plot ks curve.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params ax: Axe of figure
    : params pos_label: positive label(0 or 1)
    : params reverse: If True, reverse the ordering
    """
    if reverse == True:
        pred_value = pred_value.max() - pred_value
    fpr, tpr, thr = roc_curve(depvar_value, pred_value, pos_label=1)
    ks = (tpr - fpr).max()
    t, s = sorted(pred_value), len(pred_value)
    depth = [0] + [1 - (t.index(i) + 1) / s for i in thr[1:]]
    thr_point = depth[np.argmax(tpr - fpr)]
    if not ax:
        f, ax = plt.subplots(dpi=100)
    ax.plot(depth, fpr, lw=2, label='FPR')
    ax.plot(depth, tpr, lw=2, label='TPR')
    ax.plot(depth, tpr - fpr, lw=2,
            label='KS curve (%0.2f)' % ks)
    ax.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6))
    ax.plot([0, 0, 1], [0, 1, 1], lw=1, linestyle=':')
    ax.plot([thr_point, thr_point], [0, 1], lw=1, linestyle='-', label='threshold (%0.2f)' % thr_point)
    ax.set_xlabel('Depth', fontsize='x-large')
    ax.set_ylabel('True Positive Rate / False Positive Rate', fontsize='x-large')
    plt.title('Model\'s KS Curve', fontsize='xx-large')
    plt.legend(loc='upper left', fontsize='x-large')


def plot_lift_curve(depvar_value, pred_value, ax=None, pos_label=1, skip_depth_percent=0.05,
                    reverse=False, silent=True):
    """
    plot lift curve.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params ax: Axe of figure
    : params pos_label: positive label(0 or 1)
    : params skip_depth_percent: If True, skip percents of sorted values
    : params reverse: If True, reverse the ordering
    : params silent: If True, restrict prints
    """
    if reverse == True:
        pred_value = pred_value.max() - pred_value
    t = pd.DataFrame([depvar_value.tolist(), pred_value.tolist()], index=['true', 'pred']).T
    fpr, tpr, thr = roc_curve(t['true'], t['pred'], pos_label=pos_label)
    pos_rate = sum(depvar_value == 1) / depvar_value.shape[0]
    t, s = t.sort_values(by='pred').reset_index(drop=True), len(pred_value)
    lift = [0] + [(precision_score(t['true'], [int(i > j) for i in t['pred']])) / pos_rate for j in thr[1:]]
    depth = [0] + [1 - (np.where(t['pred'] == i)[0][0] + 1) / s for i in thr[1:]]
    idx = np.where(np.array(lift) > 1)[0][0]
    if skip_depth_percent:
        idx = max(idx, int(len(lift) * skip_depth_percent))
    if not silent:
        print('lift curve skip first {idx} indexes'.format(idx=idx))
    if not ax:
        f, ax = plt.subplots(dpi=100)
    ax.plot(depth[idx:], lift[idx:], lw=2, label='lift curve')
    ax.hlines(y=1, lw=1, label='random curve', xmin=0, xmax=1, color='c', linestyle='-.')
    ax.set_xlabel('Depth', fontsize='x-large')
    ax.set_ylabel('Lift', fontsize='x-large')
    plt.title('Model\'s Lift Curve for train set', fontsize='xx-large')
    plt.legend(loc='upper right', fontsize='x-large')


def plot_pass_pos_rate_curve(depvar_value, pred_value, ax=None, reverse=False):
    """
    plot curve describing overdue rate versus pass rate.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params ax: Axe of figure
    : params reverse: If True, reverse the ordering
    """
    if reverse == True:
        pred_value = pred_value.max() - pred_value
    t = pd.DataFrame([depvar_value.tolist(), pred_value.tolist()], index=['true', 'pred']).T
    t, s = t.sort_values(by='pred').reset_index(drop=True), len(pred_value)
    pss = np.linspace(1, s, s) / s
    od_rate = [t.iloc[:np.where(t['pred'] == i)[0][0]].true.sum() / s for i in t['pred']]
    if not ax:
        f, ax = plt.subplots(dpi=100)
    ax.plot(pss, od_rate, color='darkblue', lw=2, label='pass label curve')
    ax.plot([0, 1], [0, depvar_value.mean()], linestyle='--', color=(0.6, 0.6, 0.6),
            label='random curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Pass Rate', fontsize='x-large')
    ax.set_ylabel('Positive Rate', fontsize='x-large')
    plt.title('Model\'s Label_rate-overdue_rate Curve', fontsize='xx-large')
    plt.legend(loc="upper left", fontsize='x-large')


def plot_pr_curve(depvar_value, pred_value, ax=None, silent=True):
    """
    plot curve describing precision rate versus recall rate.

    : params depvar_value: real label
    : params pred_value: predicted value
    : params ax: Axe of figure
    : params silent: If True, restrict prints
    """
    precision, recall, _ = precision_recall_curve(depvar_value, pred_value)
    f1 = 2 * precision * recall / (precision + recall)
    if not ax:
        f, ax = plt.subplots(dpi=100)
    ax.plot(recall, precision, color='darkblue', lw=2, label='P-R curve')
    ax.plot(recall, f1, color='darkred', lw=2, label='F1 curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall Rate', fontsize='x-large')
    ax.set_ylabel('Precision Rate', fontsize='x-large')
    plt.title('Model\'s Precision-Recall Curve', fontsize='xx-large')
    plt.legend(loc="upper left", fontsize='x-large')
    if not silent:
        print('maximum of f1 scoare is ', f1.max())


def model_summary(pred_train_value, real_train_label, pred_val_value, real_val_label, pred_test_value=None,
                  real_test_label=None, ax=None, pos_label=1, use_formater=True, plot=True):
    """
    Get mddel summry by evaluating its performance on train, validation and test set.

    : params pred_train_value: predicted value for train set
    : params real_train_label: real label for train set
    : params pred_val_value: predicted value for validation set
    : params real_val_label: real label for validation set
    : params pred_test_value: predicted value for test set
    : params real_test_label: real label for test set
    : params ax: Axe of figure
    : params pos_label: positive label(0 or 1), active when lable is binary
    : params use_formater: If True, output float numbers in format
    : params plot: If True, plot ROC if label is binary
    """
    cal_val = True if pred_val_value is not None and real_val_label is not None else False
    cal_test = True if pred_test_value is not None and real_test_label is not None else False
    if ud_type_of_target(real_train_label)=='binary':
        col = ['auc', 'ks', 'threshold', 'pass_rate', 'missing_negative_rate']
        res_train = get_clf_statistics(pred_train_value, real_train_label, pos_label=pos_label)
        ac, _, _, _, _, fpr, tpr = res_train
        res = pd.DataFrame([res_train[:-2]], index=['train'], columns=col)
        if plot:
            if not ax:
                f, ax = plt.subplots(dpi=100)
            ax.plot(fpr, tpr, label='train AUC = %0.2f' % ac, lw=2)
            ax.plot([0, 1], [0, 1])
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_ylabel('True Positive Rate', fontsize='x-large')
            ax.set_xlabel('False Positive Rate', fontsize='x-large')
            plt.title('ROC curve', fontsize='xx-large')
            plt.legend(loc='lower right', fontsize='x-large')
        if cal_val:
            res_val = get_clf_statistics(pred_val_value, real_val_label, pos_label=pos_label)
            ac2, _, _, _, _, fpr2, tpr2 = res_val
            res_val = pd.DataFrame([res_val[:-2]], index=['val'], columns=col)
            res = pd.concat([res, res_val], axis=0)
            if plot:
                ax.plot(fpr2, tpr2, label='validation AUC = %0.2f' % ac2, lw=2)
                plt.legend(loc='lower right', fontsize='x-large')
        else:
            print("No validation set or no labels for validation set")
        if cal_test:
            res_test = get_clf_statistics(pred_test_value, real_test_label, pos_label=pos_label)
            ac3, _, _, _, _, fpr3, tpr3 = res_test
            res_test = pd.DataFrame([res_test[:-2]], index=['test'], columns=col)
            res = pd.concat([res, res_test], axis=0)
            if plot:
                ax.plot(fpr3, tpr3, label='test AUC = %0.2f' % ac3, lw=2)
                plt.legend(loc='lower right', fontsize='x-large')
        else:
            print("No test set or no labels for test set")
        if use_formater:
            res['pass_rate'] = res['pass_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
            res['missing_negative_rate'] = res['missing_negative_rate'].map(lambda x: '{:.1f}%'.format(x * 100))

    elif ud_type_of_target(real_train_label)=='continuous':
        col = ['mae', 'mse', 'msle', 'rmse']
        res_train = get_reg_statistics(pred_train_value, real_train_label)
        res = pd.DataFrame([res_train], index=['train'], columns=col)
        if cal_val:
            res_val = get_reg_statistics(pred_train_value, real_train_label)
            res_val = pd.DataFrame([res_val], index=['val'], columns=col)
            res = pd.concat([res, res_val], axis=0)
        else:
            print("No validation set or no labels for validation set")
        if cal_test:
            res_test = get_reg_statistics(pred_train_value, real_train_label)
            res_test = pd.DataFrame([res_test], index=['test'], columns=col)
            res = pd.concat([res, res_test], axis=0)
        else:
            print("No test set or no labels for test set")

    res = np.round(res, 4)
    return res


def plot_sorting_ability(pred_train_value, real_train_label, pred_val_value, real_val_label,
                         pred_test_value=None, real_test_label=None, Good_to_bad=False,
                         bins=10, how='qcut', model_name='xgb'):
    """
    Plot a figure showing model's sorting ability.

    : params pred_train_value: predicted value for train set
    : params pred_val_value: predicted value for validation set
    : params real_train_label: real label for train set
    : params real_val_label: real label for validation set
    : params pred_test_value: predicted value for test set
    : params real_test_label: real label for test set
    : params Good_to_bad: bins' order
    : params bins: number of bins
    : params how: how to cut the range
    : params model_name: model name
    """
    plot_val = True if pred_val_value is not None and real_val_label is not None else False
    plot_test = True if pred_test_value is not None else False
    fig_cnt = 1 + plot_val + plot_test

    # train dataset
    pred_train_value_ = pd.DataFrame({'prob': list(pred_train_value), 'pos_rate': list(real_train_label)})

    if how == 'qcut':
        pred_train_value_['range'], cutpoint = \
            pd.qcut(pred_train_value_['prob'], q=np.linspace(0, 1, bins + 1), retbins=True, precision=5,
                    duplicates='drop')

        if pred_train_value_['range'].nunique() != bins:
            bins = pred_train_value_['range'].nunique()

    cutpoint = [0] + sorted(cutpoint)[1:-1] + [1]
    pred_train_value_['range'] = pd.cut(pred_train_value_['prob'], cutpoint, precision=5)
    train_stats = bin_stat(pred_train_value_, 'range', 'pos_rate')
    print('cutpoints derived from train set are ', ', '.join([str(i) for i in cutpoint]))

    if Good_to_bad == False:
        xx = range(bins, 0, -1)
        xlabel = 'groups(bad -> good)'
    else:
        xx = range(1, bins + 1)
        xlabel = 'groups(good -> bad)'

    fig = plt.figure(figsize=(fig_cnt*10, 8))
    ax1 = fig.add_subplot(1, fig_cnt, 1)
    ax1.bar(xx, train_stats['count'], width=0.5, color='orange', yerr=0.000001)
    ax1.set_xticks(xx, list(range(1, bins + 1, 1)))
    ax1.set_ylabel('number', fontsize='x-large')
    ax1.set_xlabel(xlabel, fontsize='x-large')
    ax1.set_ylim([0, max(train_stats['count']) * 1.5])
    ax1_twin = ax1.twinx()
    ax1_twin.plot(xx, train_stats['pos_rate'], linestyle='--', marker='o', markersize=5,
                  label='train set', lw=2)
    ax1_twin.plot(xx, train_stats['accum_pos_rate'], linestyle='--', marker='o', markersize=5,
                  label='train set/accumulated', lw=2)

    for a, b in zip(xx, train_stats['pos_rate']):
        ax1_twin.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=10)

    for a, b in zip(xx, train_stats['accum_pos_rate']):
        ax1_twin.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=10)

    ax1_twin.set_ylabel('overdue rate', fontsize='x-large')
    plt.title('{}\'s sorting ability'.format(model_name), fontsize='xx-large')
    plt.legend(loc='upper right', fontsize='x-large')

    # validation dataset
    fig_cnt_tmp = 1
    if plot_val:
        fig_cnt_tmp += 1
        pred_val_value_ = pd.DataFrame({'prob': list(pred_val_value), 'pos_rate': list(real_val_label)})
        pred_val_value_['range'] = pd.cut(pred_val_value_['prob'], cutpoint, precision=5)
        val_stats = bin_stat(pred_val_value_, 'range', 'pos_rate', how='cut', bins=bins)
        ax2 = fig.add_subplot(1, fig_cnt, fig_cnt_tmp)
        ax2.bar(xx, val_stats['count'], width=0.5, color='orange', yerr=0.000001)
        ax2.set_xticks(xx, list(range(1, bins + 1, 1)))
        ax2.set_ylabel('number', fontsize='x-large')
        ax2.set_xlabel(xlabel, fontsize='x-large')
        ax2.set_ylim([0, max(val_stats['count'])])
        ax2_twin = ax2.twinx()
        ax2_twin.plot(xx, val_stats['pos_rate'], linestyle='--', marker='o', markersize=5,
                      label='validation set', lw=2)
        ax2_twin.plot(xx, val_stats['accum_pos_rate'], linestyle='--', marker='o', markersize=5,
                      label='validation set/accumulated', lw=2)

        for a, b in zip(xx, val_stats['pos_rate']):
            plt.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=10)

        for a, b in zip(xx, val_stats['accum_pos_rate']):
            plt.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=10)

        ax2_twin.set_ylabel('overdue rate', fontsize='x-large')
        plt.title('{}\'s sorting ability'.format(model_name), fontsize='xx-large')
        plt.legend(loc='upper right', fontsize='x-large')

        train_stats['pos_rate'] = train_stats['pos_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
        train_stats['accum_pos_rate'] = train_stats['accum_pos_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
        train_stats['count_proportion'] = train_stats['count_proportion'].map(lambda x: '{:.1f}%'.format(x * 100))
        train_stats['accum_count'] = train_stats['accum_count'].map(lambda x: '{:.1f}%'.format(x * 100))

        val_stats['pos_rate'] = val_stats['pos_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
        val_stats['accum_pos_rate'] = val_stats['accum_pos_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
        val_stats['count_proportion'] = val_stats['count_proportion'].map(lambda x: '{:.1f}%'.format(x * 100))
        val_stats['accum_count'] = val_stats['accum_count'].map(lambda x: '{:.1f}%'.format(x * 100))

    # test dataset
    if plot_test and real_test_label is not None:
        fig_cnt_tmp += 1
        pred_test_value_ = pd.DataFrame({'prob': pred_test_value, 'pos_rate': real_test_label})
        pred_test_value_['range'] = pd.cut(pred_test_value_['prob'], cutpoint, precision=5)
        test_stats = bin_stat(pred_test_value_, 'range', 'pos_rate', how='cut', bins=bins)

        ax3 = fig.add_subplot(1, fig_cnt, fig_cnt_tmp)
        ax3.bar(xx, test_stats['count'], width=0.5, color='orange', yerr=0.000001)
        ax3.set_xticks(xx, list(range(1, bins + 1, 1)))
        ax3.set_ylabel('number', fontsize='x-large')
        ax3.set_ylim([0, max(test_stats['count'])])
        ax3.set_xlabel(xlabel, fontsize='x-large')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(xx, test_stats['pos_rate'], linestyle='--', marker='o', markersize=5,
                      label='test set', lw=2)
        ax3_twin.plot(xx, test_stats['accum_pos_rate'], linestyle='--', marker='o', markersize=5,
                      label='test set/accumulated', lw=2)

        for a, b in zip(xx, test_stats['pos_rate']):
            plt.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=10)

        for a, b in zip(xx, test_stats['accum_pos_rate']):
            plt.text(a, b + 0.005, '%.2f' % b, ha='center', va='bottom', fontsize=10)

        ax3_twin.set_ylabel('overdue rate', fontsize='x-large')
        plt.title('{}\'s sorting ability'.format(model_name), fontsize='xx-large')
        plt.legend(loc='upper right', fontsize='x-large')

        test_stats['pos_rate'] = test_stats['pos_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
        test_stats['accum_pos_rate'] = test_stats['accum_pos_rate'].map(lambda x: '{:.1f}%'.format(x * 100))
        test_stats['count_proportion'] = test_stats['count_proportion'].map(lambda x: '{:.1f}%'.format(x * 100))
        test_stats['accum_count'] = test_stats['accum_count'].map(lambda x: '{:.1f}%'.format(x * 100))

    elif plot_test and real_test_label is None:
        fig_cnt_tmp += 1
        pred_test_value_ = pd.DataFrame({'prob': pred_test_value})
        pred_test_value_['range'] = pd.cut(pred_test_value_['prob'], cutpoint, precision=5)
        test_stats = bin_stat(pred_test_value_, 'range', None, how='cut', bins=bins)

        ax3 = fig.add_subplot(1, fig_cnt, fig_cnt_tmp)
        ax3.bar(xx, test_stats['count'], width=0.5, color='orange', yerr=0.000001)
        ax3.set_xticks(xx, list(range(1, bins + 1, 1)))
        ax3.set_ylabel('number', fontsize='x-large')
        ax3.set_ylim([0, max(test_stats['count']) * 1.2])

        for a, b in zip(xx, test_stats['count']):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

        for a, b in zip(xx, test_stats['count']):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

        ax3.set_xlabel(xlabel, fontsize='x-large')
        plt.title('{}\'s sorting ability'.format(model_name), fontsize='xx-large')
        plt.legend(loc='upper right', fontsize='x-large')

        test_stats['count_proportion'] = test_stats['count_proportion'].map(lambda x: '{:.1f}%'.format(x * 100))
        test_stats['accum_count'] = test_stats['accum_count'].map(lambda x: '{:.1f}%'.format(x * 100))

    if plot_val and plot_test:
        return np.round(train_stats, 3), np.round(val_stats, 3), np.round(test_stats, 3)
    elif not plot_val and plot_test:
        return np.round(train_stats, 3), np.round(test_stats, 3)        
    elif plot_val and not plot_test:
        return np.round(train_stats, 3), np.round(val_stats, 3)
    else:
        return np.round(train_stats, 3)


def plot_all_figures(pred_train_value, real_train_label, pred_val_value, real_val_label,
                     pred_test_value=None, real_test_label=None,
                     fig_list=['roc', 'ks', 'lift', 'pr', 'pl', 'sort'],
                     model_name='xgb', pos_label=1, Good_to_bad=False, bins=10, reverse=False):
    """
    Plot all figures indicating performance of the model.

    : params pred_train_value: predicted value for train set
    : params pred_val_value: predicted value for validation set
    : params real_train_label: real label for train set
    : params real_val_label: real label for validation set
    : params pred_test_value: predicted value for test set
    : params real_test_label: real label for test set
    : params fig_list: list of figure types to plot
    : params model_name: model name
    : params pos_label: positive label(0 or 1)
    : params Good_to_bad: order of bins
    : params bins: number of bins
    : params use_formater: If True, output float numbers in format
    : params reverse: If True, reverse the ordering
    """
    avai_fig_list = ['roc', 'ks', 'lift', 'pr', 'pl', 'sort']

    if len(set(fig_list) - set(avai_fig_list)) != 0:
        raise ValueError("Figure Types must chosen among 'roc','ks','lift','pr','pl' and 'sort")

    width = len(set(fig_list))
    plot_val = True if pred_val_value is not None and real_val_label is not None else False
    plot_test = True if pred_test_value is not None and real_test_label is not None else False
    fig_inx = 1
    fig = plt.figure(figsize=(10 * 3, 8 * width))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.3)

    if 'roc' in fig_list:
        ax1 = fig.add_subplot(width, 3, fig_inx)
        res = model_summary(pred_train_value, real_train_label, pred_val_value, real_val_label,
                            pred_test_value, real_test_label, ax=ax1, use_formater=True, plot=True)

        fig_inx += 3
        print(res)

    if 'ks' in fig_list:
        ax4 = fig.add_subplot(width, 3, fig_inx)
        plot_ks_curve(real_train_label, pred_train_value, ax=ax4, pos_label=pos_label, reverse=reverse)
        plt.title('{model_name} model\'s KS Curve for train set'.format(model_name=model_name))
        if plot_val:
            ax5 = fig.add_subplot(width, 3, fig_inx + 1)
            plot_ks_curve(real_val_label, pred_val_value, ax=ax5, pos_label=pos_label, reverse=reverse)
            plt.title('{model_name} model\'s KS Curve for validation set'.format(model_name=model_name))
        if plot_test:
            ax6 = fig.add_subplot(width, 3, fig_inx + 2)
            plot_ks_curve(real_test_label, pred_test_value, ax=ax6, pos_label=pos_label, reverse=reverse)
            plt.title('{model_name} model\'s KS Curve for test set'.format(model_name=model_name))

        fig_inx += 3

    if 'lift' in fig_list:
        ax7 = fig.add_subplot(width, 3, fig_inx)
        plot_lift_curve(real_train_label, pred_train_value, ax=ax7, pos_label=pos_label,
                        skip_depth_percent=0.05, reverse=reverse)
        plt.title('{model_name} model\'s Lift Curve for train set'.format(model_name=model_name))
        if plot_val:
            ax8 = fig.add_subplot(width, 3, fig_inx + 1)
            plot_lift_curve(real_val_label, pred_val_value, ax=ax8, pos_label=pos_label,
                            skip_depth_percent=0.05, reverse=reverse)
            plt.title('{model_name} model\'s Lift Curve for validation set'.format(model_name=model_name))
        if plot_test:
            ax9 = fig.add_subplot(width, 3, fig_inx + 2)
            plot_lift_curve(real_test_label, pred_test_value, ax=ax9, pos_label=pos_label,
                            skip_depth_percent=0.05, reverse=reverse)
            plt.title('{model_name} model\'s Lift Curve for test set'.format(model_name=model_name))

        fig_inx += 3

    if 'pr' in fig_list:
        ax10 = fig.add_subplot(width, 3, fig_inx)
        plot_pr_curve(real_train_label, pred_train_value, ax=ax10)
        plt.title('{model_name} model\'s Precision-Recall Curve for train set' \
                  .format(model_name=model_name))
        if plot_val:
            ax11 = fig.add_subplot(width, 3, fig_inx + 1)
            plot_pr_curve(real_val_label, pred_val_value, ax=ax11)
            plt.title('{model_name} model\'s Precision-Recall Curve for validation set' \
                      .format(model_name=model_name))
        if plot_test:
            ax12 = fig.add_subplot(width, 3, fig_inx + 2)
            plot_pr_curve(real_test_label, pred_test_value, ax=ax12)
            plt.title('{model_name} model\'s Precision-Recall Curve for test set' \
                      .format(model_name=model_name))
        fig_inx += 3

    if 'pl' in fig_list:
        ax13 = fig.add_subplot(width, 3, fig_inx)
        plot_pass_pos_rate_curve(real_train_label, pred_train_value, ax=ax13, reverse=reverse)
        plt.title('{model_name} model\'s Pass rate-Positive_rate Curve for train set' \
                  .format(model_name=model_name))
        if plot_val:
            ax14 = fig.add_subplot(width, 3, fig_inx + 1)
            plot_pass_pos_rate_curve(real_val_label, pred_val_value, ax=ax14, reverse=reverse)
            plt.title('{model_name} model\'s Pass rate-ositive_rate Curve for validation set' \
                      .format(model_name=model_name))
        if plot_test:
            ax15 = fig.add_subplot(width, 3, fig_inx + 2)
            plot_pass_pos_rate_curve(real_test_label, pred_test_value, ax=ax15, reverse=reverse)
            plt.title('{model_name} model\'s Pass rate-Positive rate Curve for test set' \
                      .format(model_name=model_name))

    """
    new figure types to add

    """

    if 'sort' in fig_list:
        sort_res = plot_sorting_ability(pred_train_value, real_train_label, pred_val_value,
                                        real_val_label, pred_test_value=pred_test_value,
                                        real_test_label=real_test_label, Good_to_bad=False,
                                        bins=10, how='qcut', model_name=model_name)