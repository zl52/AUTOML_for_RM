import numpy as np
import pandas as pd
import bisect
import math
from collections import defaultdict, Counter
from scipy import stats
from scipy.stats import chisquare
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target

from model_evaluation import bin_stat
from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   7. FEATURE ENCODING  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def decision_tree_cutpoints(x, y, max_depth=4, min_samples_leaf=0.05, max_leaf_nodes=10,
                            monotonic_bin=False, spearman_cor_thr=0.1, random_state=7):
    """
    A decision tree method which gives cut points to bin continuous features.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
            Independent variable to be encoded
    y: array-like of shape (n_samples,)
            Dependent variable
    max_depth: int
            Maximum depth of the tree
    min_samples_leaf: int or float
            Minimum number of samples required to be at a leaf node
    max_leaf_nodes: int
            Grow a tree with max_leaf_nodes in best-first fashion. 
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
    monotonic_bin: boolean
            Bins are ordered monotonically based on odds of dependent variables
    spearman_cor_thr: float
            Threshold of spearman correlation
    random_state: int
            Random seed

    Returns
    ----------
    cut_points: list 
            List of cut points
    """
    if y is None:
        raise ValueError("y must not be none if \'dt\' is chosen")
    try:
        df = pd.DataFrame({'feature': np.array(x), 'label': np.array(y)})
    except:
        df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.columns = ['feature', 'label']
    df2 = df[df['feature'].notnull()]
    df_na = df[df['feature'].isnull()]
    x = df2['feature']
    y = df2['label']
    if not monotonic_bin:
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                    max_leaf_nodes=max_leaf_nodes, random_state=random_state)
        dt.fit(np.array(x).reshape(-1, 1), np.array(y))
        thr = dt.tree_.threshold
        cut_points = sorted(thr[np.where(thr != -2)])
    else:
        r = 0
        while np.abs(r) < spearman_cor_thr and max_leaf_nodes > 1:
            dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        max_leaf_nodes=max_leaf_nodes, random_state=random_state)
            dt.fit(np.array(x).reshape(-1, 1), np.array(y))
            thr = dt.tree_.threshold
            cut_points = sorted(thr[np.where(thr != -2)])
            df['bin_range'] = pd.cut(df['feature'], cut_points, include_lowest=True)
            df_na['bin_range'] = '_MISSING_'
            df = pd.concat([df, df_na]).reset_index(drop=True)
            stat = df.groupby('bin_range', as_index=True)
            r, _ = stats.spearmanr(stat.mean().feature, stat.mean().label)
            max_leaf_nodes -= 1
    return cut_points


def chisquare_cutpoints(x, y, bins=10, min_pct_thr=0):
    """
    Method gives cut points of a continuous feature based on chisquare test.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
            Independent variable to be encoded
    y: array-like of shape (n_samples,)
            Dependent variable
    bins: int
            Maximal number of bins
    min_pct_thr: int or float
            Minimum count-percentage of interval

    Returns
    ----------
    cut_points: list 
            List of cut points
    """
    if y is None:
        raise ValueError("y must not be none if \'chi\' is chosen")
    if type_of_target(y)!='binary':
        raise ValueError("y must be binary if \'chi\' is chosen")
    try:
        df = pd.DataFrame({'feature': np.array(x), 'label': np.array(y)})
    except:
        df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.columns = ['feature', 'label']
    df = df[df['feature'].notnull()]
    value_list, value_cnt = list(sorted(df.feature.unique())), df.feature.nunique()
    if value_cnt <= 10:
        print("The number of original levels is less than or equal to max interval counts")
        return value_list
    else:
        if value_cnt > 100:
            df['bin_range'], cutpoints = pd.qcut(df['feature'], q=np.linspace(0, 1, 101),
                                                 retbins=True, precision=6, duplicates='drop')
        else:
            df['bin_range'] = df['feature']
        stat1 = bin_stat(df, 'bin_range', 'label', reverse=False, how='cut', extra_stat=False)
        interval_list, interval_cnt = [[i] for i in sorted(df.bin_range.unique())], df.bin_range.nunique()
        while (len(interval_list) > bins):
            chisqList = []
            for i in range(len(interval_list) - 1):
                interval_sublist1 = interval_list[i] + interval_list[i + 1]
                test_stat1 = stat1[stat1.index.isin(interval_sublist1)]
                chisq1 = chisquare(test_stat1['pos_count'], test_stat1['expected_count'])
                chisqList.append(chisq1)
            interval_merged_idx1 = chisqList.index(min(chisqList))
            interval_merged_idx2 = chisqList.index(min(chisqList)) + 1
            interval_list[interval_merged_idx1] = interval_list[interval_merged_idx1]\
                                                + interval_list[interval_merged_idx2]
            interval_list.remove(interval_list[interval_merged_idx2])
        try:
            cutpoints = sorted([interval_list[0][0].left] + [i[-1].right for i in interval_list])
        except:
            cutpoints = sorted(list(set([interval_list[0][0] - 0.001] + [i[-1] for i in interval_list])))
        df['bin_range'] = pd.cut(df['feature'], bins=cutpoints, precision=6, retbins=True)[0]
        stat2 = bin_stat(df, 'bin_range', 'label', reverse=False, how='cut', extra_stat=False)
        interval_list = [[i] for i in stat2.index.tolist()]
        min_pos_rate, max_pos_rate = stat2['pos_rate'].min(), stat2['pos_rate'].max()

        while min_pos_rate == 0 or max_pos_rate == 1:
            bad_interval = stat2.iloc[np.where(stat2['pos_rate'].isin([0, 1]))[0]].index.tolist()[0]
            bad_interval_idx = interval_list.index([bad_interval])
            if bad_interval_idx == 0:
                cutpoints.pop(1)
            elif bad_interval_idx == stat2.shape[0] - 1:
                cutpoints.pop(-2)
            else:
                interval_sublist2 = [bad_interval] + interval_list[bad_interval_idx - 1]
                test_stat2 = stat2[stat2.index.isin(interval_sublist2)]
                chisq2 = chisquare(test_stat2['pos_count'], test_stat2['expected_count'])
                interval_sublist3 = [bad_interval] + interval_list[bad_interval_idx + 1]
                test_stat3 = stat2[stat2.index.isin(interval_sublist3)]
                chisq3 = chisquare(test_stat3['pos_count'], test_stat3['expected_count'])
                if chisq2 < chisq3:
                    cutpoints.remove(cutpoints[bad_interval_idx])
                else:
                    cutpoints.remove(cutpoints[bad_interval_idx + 1])
            df['bin_range'] = pd.cut(df['feature'], bins=cutpoints, precision=6, retbins=True)[0]
            stat3 = stat2 = bin_stat(df, 'bin_range', 'label', reverse=False, how='cut', extra_stat=False)
            interval_list = [[i] for i in stat2.index.tolist()]
            min_pos_rate, max_pos_rate = stat2['pos_rate'].min(), stat2['pos_rate'].max()

        if min_pct_thr > 0:
            min_pct_interval, min_pct = np.argmin(stat3['count_proportion']), np.min(stat3['count_proportion'])
            while min_pct < min_pct_thr and len(cutpoints) > 2:
                bad_interval = np.argmin(stat3['count_proportion'])
                bad_interval_idx = interval_list.index([bad_interval])
                if bad_interval_idx == 0:
                    cutpoints.pop(1)
                elif bad_interval_idx == stat3.shape[0] - 1:
                    cutpoints.pop(-2)
                else:
                    interval_sublist4 = [bad_interval] + interval_list[bad_interval_idx - 1]
                    test_stat4 = stat3[stat3.index.isin(interval_sublist4)]
                    chisq4 = chisquare(test_stat4['pos_count'], test_stat4['expected_count'])

                    interval_sublist5 = [bad_interval] + interval_list[bad_interval_idx + 1]
                    test_stat5 = stat3[stat3.index.isin(interval_sublist5)]
                    chisq5 = chisquare(test_stat5['pos_count'], test_stat5['expected_count'])
                    if chisq4 < chisq5:
                        cutpoints.remove(cutpoints[bad_interval_idx])
                    else:
                        cutpoints.remove(cutpoints[bad_interval_idx + 1])
                df['bin_range'] = pd.cut(df['feature'], bins=cutpoints, precision=6, retbins=True)[0]
                stat4 = stat3 = bin_stat(df, 'bin_range', 'label', reverse=False, how='cut', extra_stat=False)
                interval_list = [[i] for i in stat3.index.tolist()]
                min_pct_interval, min_pct = np.argmin(stat3['count_proportion']), np.min(stat3['count_proportion'])
        return cutpoints


def get_cut_points(x, y=None, bins=10, cut_method='dt', precision=6, **kwargs):
    """
    Get cut points using the method assigned. Defined methods include 'cut', 'qcut', 'dt' and 'chi, 
    standing for 'cut by defined cut points', 'equally cut', 'cut by decision tree cutpoints'
    and 'cut using chisquare statistics respectively'.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
            Independent variable to be encoded
    y: array-like of shape (n_samples,)
            Dependent variable
    bins: int
            Maximal number of bins
    cut_method: str
            Cut method ('cut'(defualt), 'qcut', 'dt' or 'chi')
    precision: float
            Precision
    kwargs: dict
            Dictionary of params for decision tree

    Returns
    ----------
    cut_points: list 
            List of cut points
    """
    if cut_method == 'cut':
        _, cut_points = pd.cut(x, bins=bins, retbins=True, precision=precision)
    elif cut_method == 'qcut':
        _, cut_points = pd.qcut(x, q=bins, retbins=True, duplicates='drop', precision=precision)
    elif cut_method == 'dt':
        cut_points = decision_tree_cutpoints(x, y, **kwargs)
    elif cut_method == 'chi':
        cut_points = chisquare_cutpoints(x, y, bins=bins)
    else:
        raise ValueError("Method must chosen among \'cut\',\'qcut\', \'dt\' and \'chi\'")
    if cut_method != 'dt':
        cut_points = cut_points[1:-1]
    cut_points = list(cut_points)
    cut_points.append(np.inf)
    cut_points.insert(0, -np.inf)
    return cut_points


def woe(x, y, woe_min=-20, woe_max=20):
    """
    Get feature's woe encoding dictionary.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
            Feature to be encoded
    y: array-like of shape (n_samples,)
            Dependent variable
    woe_min: int or float
            Minimum of woe value
    woe_max: int or float
            Maximum of woe value

    Returns
    ----------
    dmap: dict
            Woe dictionary
    pp_map: dict
            Dictionary of each bin's positive-count's proportion 
    np_map: dict
            Dictionary of each bin's negative-count's proportion 
    """
    x = np.array(x)
    y = np.array(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    dmap, pp_map, np_map = {}, {}, {}
    for k in pd.Series(x).unique():
        indice = np.where(x == k)[0]
        pos_prop = (y[indice] == 1).sum() / pos
        neg_prop = (y[indice] == 0).sum() / neg
        if pos_prop == 0:
            woe = woe_min
        elif neg_prop == 0:
            woe = woe_max
        else:
            woe = math.log(pos_prop / neg_prop)
        dmap[k] = woe
        pp_map[k] = pos_prop
        np_map[k] = neg_prop
    return dmap, pp_map, np_map


class LabelEncoder_ud(LabelEncoder):
    """
    User-defined label encoder which can encode both categorical or continuous features (regarded as categorical feature)
    and deal with unseen value (overcome drawbacks of sklearn's label encoder).
    Mapping relations are stored in dmap.
    """

    def __init__(self,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*'):
        """
        Parameters
        ----------
        unknown: str
                String used to denote NA
        unseen: str
                String used to denote unseen cases
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        """
        self.unknown = unknown
        self.unseen = unseen
        self.x_dtypes = 'cav'

    def fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)
        """
        x = pd.Series(x).fillna(self.unknown).astype(str)
        le = LabelEncoder()
        le.fit(x)
        le_classes = le.classes_.tolist()
        bisect.insort_left(le_classes, self.unseen)
        le.classes_ = le_classes
        self.encoder = le
        l = list(x.unique()) + [self.unseen]
        self.dmap = dict(zip(self.encoder.transform(l), l))
        self.feat_name = '%s_le'%prefix

    def transform(self, x, y=None):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x).fillna(self.unknown).astype(str)
        x = [s if s in self.encoder.classes_ else self.unseen for s in x]
        return self.encoder.transform(x)

    def fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x and transform it.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y=y, prefix=prefix)
        return self.transform(x, y=y)


class OneHotEncoder_ud(OneHotEncoder):
    """
    User-defined onehot encoder which can encode both categorical or continuous feature (regarded as categorical feature)
    and deal with unseen value (overcome drawbacks of sklearn's OneHot encoder).
    Mapping relations are stored in dmap.
    """

    def __init__(self,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*'):
        """
        Parameters
        ----------
        unknown: str
                Used to denote NA
        unseen: str
                Used to denote unseen cases
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        """
        self.unknown = unknown
        self.unseen = unseen
        self.x_dtypes = 'cav'

    def fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)
        """
        x = pd.Series(x).fillna(self.unknown).astype(str)
        le = LabelEncoder_ud()
        le.fit(x)
        self.le = le
        self.dmap = le.dmap
        x = self.le.transform(x)
        onehot = OneHotEncoder()
        onehot.fit(x.reshape(-1, 1))
        self.encoder = onehot
        active_features_ = self.encoder.active_features_
        self.feat_name = ['%s_%s_oe'%(prefix, str(self.dmap[i])) for i in active_features_ if str(self.dmap[i])!=self.unknown]

    def transform(self, x, y=None):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x).fillna(self.unknown).astype(str)
        x = self.le.transform(x)
        return self.encoder.transform(x.reshape(-1, 1)).toarray()

    def fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x and transform it.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y=y, prefix=prefix)
        return self.transform(x, y=y)


class CountEncoder(object):
    """
    User-defined count encoder which replaces categorical features or continuous feature (regarded as categorical feature)
    with its counts and replace unseen values with 1. Can use log-transform to avoid the impact of outliers.
    Mapping relations are stored in dmap.
    """

    def __init__(self,
                 base_value=1,
                 use_log_transform=False,
                 smoothing_param=1,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*'):
        """
        Parameters
        ----------
        base_value: int
                Default count of unseen values
        use_log_transform: boolean
                Use log transformation
        smoothing_param: float
                Smoothing parameter
        unknown: str or int or float
                String or value used to denote NA
        unseen: str or int or float
                String or value used to denote unseen cases
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        """
        self.base_value = base_value
        self.use_log_transform = use_log_transform
        self.smoothing_param = 1
        self.unknown = unknown
        self.unseen = unseen
        self.x_dtypes = 'cav'

    def fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x).fillna(self.unknown)
        # l = list(x.unique())
        # while (self.unknown in l):
        #     self.unknown = self.unknown + '*'
        # while (self.unseen in l):
        #     self.unseen = self.unseen + '*'

        self.dmap = Counter(x)
        self.feat_name = prefix + '_ce'
        self.dmap.update({self.unseen: self.base_value})
        self.dmap.update({self.unknown: self.base_value})

    def transform(self, x, y=None):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x).fillna(self.unknown)
        x = np.array([self.dmap[i] + self.smoothing_param if i in self.dmap.keys() \
                      else self.base_value for i in x])
        if self.use_log_transform:
            x = np.log(x)
        return x

    def fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x and transform it.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y=y, prefix=prefix)
        return self.transform(x, y=y)


class BinEncoder(object):
    """
    User-defined bin encoder can convert a continuous feature to a discrete feature.
    Mapping relations are stored in dmap.
    """

    def __init__(self,
                 bins=10,
                 cut_method='dt',
                 labels=None,
                 interval=True,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*',
                 **kwargs):
        """
        bins: int
                Number of bins
        cut_method: str
                Cut method ('cut'(defualt), 'qcut', 'dt' or 'chi')
        labels: list
                Category names for bins
        interval: boolean
                If interval is True, param labels is deactivated
        unknown: str or int or float
                String or value used to denote NA
        unseen: str or int or float
                String or value used to denote unseen cases
        base_value: int or float
                Default bin value for unseen values
        kwargs: dict
                Dictionary of params for decision tree
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        """
        self.bins = bins
        self.cut_method = cut_method
        self.labels = labels
        self.interval = interval
        self.unknown = unknown
        self.unseen = unseen
        self.base_value = 'np.nan'
        self.kwargs = kwargs
        self.x_dtypes = 'cov'

    def fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        if self.labels and len(self.labels) != self.bins:
            raise ValueError('The length of bin labels must be equal to the number of bins')
        if y is None:
            self.cut_method = 'qcut'
        self.cut_points = get_cut_points(x, y, self.bins, self.cut_method, **self.kwargs)
        if self.interval and self.labels is None:
            self.labels = np.arange(len(self.cut_points) - 1)
            self.dmap = dict(zip(pd.cut(x, self.cut_points, precision=6).unique().sort_values(), self.labels))
            self.dmap.update({self.unseen: self.base_value})
            self.dmap.update({self.unknown: self.base_value})
        self.feat_name = '%s_be'%prefix

    def transform(self, x, y=None):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.cut(x, bins=self.cut_points, labels=self.labels, precision=6)
        return np.array(x)

    def fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x and Encode it.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y=y, prefix=prefix)
        return self.transform(x, y=y)


class WoeEncoder(object):
    """
    User-defined woe encoder which converts both categorical and continuous features
    to discrete features. Mapping relations are stored in dmap.
    """

    def __init__(self,
                 cut_method='dt',
                 bins=10,
                 woe_min=-20,
                 woe_max=20,
                 labels=None,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*',
                 x_dtypes='cov',
                 **kwargs):
        """
        cut_method: str
                Cut method ('cut'(defualt), 'qcut', 'dt' or 'chi')
        bins: int
                Number of bins
        woe_min: int or float
                Minimum of woe value
        woe_max: int or float
                Maximum of woe value
        labels: list
                Category names for bins
        unknown: str or int or float
                String or value used to denote NA
        unseen: str or int or float
                String or value used to denote unseen cases
        base_value: int or float
                Default bin value for unseen values
        kwargs: dict
                Dictionary of params for decision tree
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        """
        self.cut_method = cut_method
        self.bins = bins
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.labels = labels
        self.unknown = unknown
        self.unseen = unseen
        self.base_value = 0
        self.kwargs = kwargs
        self.x_dtypes = x_dtypes

    def fit(self, x, y, prefix=''):
        """
        Fit transformer by checking feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x)
        if y is None:
            raise ValueError("y must not be none if WoeEncoder is chosen")
        if type_of_target(y)!='binary':
            raise ValueError("y must be binary if WoeEncoder is chosen")

        if not self.cut_method and x.count()/x.nunique()< 10:
            self.cut_method = 'dt'
            # print('%d samples are too few for %d categories and need to be binned'%(len(x[~np.isnan(x)]), len(np.unique(x[~np.isnan(x)]))))

        if not self.cut_method or x.nunique()<=5: # too few unique values and be regarded as categorical variable
            self.x_dtypes = 'cav'
            x = x.fillna(self.unknown)
            self.dmap, pp_map, np_map = woe(x, y, self.woe_min, self.woe_max)
            self.iv = sum([(pp_map.get(i[0]) - np_map.get(i[0])) * self.dmap.get(i[0]) for i in self.dmap.items()])

        elif self.x_dtypes == 'cav': # for categorical variable
            x = x.fillna(self.unknown)
            self.cut_method = 'dt'
            self.dmap, _, _ = woe(x, y, self.woe_min, self.woe_max)
            x_tmp = [self.dmap[i] for i in x]
            cut_points = get_cut_points(x_tmp, y, self.bins, self.cut_method, **self.kwargs)
            x_tmp = pd.cut(x_tmp, cut_points, precision=6)
            dmap2, pp_map2, np_map2 = woe(x_tmp, y, self.woe_min, self.woe_max)
            for i in self.dmap.items():
                self.dmap[i[0]] = dmap2[pd.cut([i[1]], cut_points, precision=6)[0]]
            self.iv = sum([(pp_map2.get(i[0]) - np_map2.get(i[0])) * dmap2.get(i[0]) for i in dmap2.items()])

        elif self.x_dtypes == 'cov': # for continuous variable
            be = BinEncoder(self.bins, self.cut_method, self.labels, interval=False)
            x = be.fit_transform(x, y, prefix)
            x = pd.Series([i if not pd.isna(i) else self.unknown for i in x])
            self.cut_points = be.cut_points
            self.dmap, pp_map, np_map = woe(x, y, self.woe_min, self.woe_max)
            self.iv = sum([(pp_map.get(i[0]) - np_map.get(i[0])) * self.dmap.get(i[0]) for i in self.dmap.items()])
        else:
            raise ValueError("x_dtypes must be chosen between \'cov\' and \'cav\'")

        if not self.unknown in self.dmap.keys():
            self.dmap.update({self.unknown: self.base_value})
        self.dmap.update({self.unseen: self.base_value})
        self.feat_name = '%s_we'%prefix

    def transform(self, x, y=None):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        if self.cut_method is not None and self.x_dtypes == 'cov' and pd.Series(x).nunique()>5:
            x = pd.cut(x, bins=self.cut_points, precision=6)
        x = pd.Categorical(x).add_categories(self.unknown).fillna(self.unknown)
        # x = x.fillna(self.unknown)
        x = np.array([self.dmap[i] if i in self.dmap.keys() else self.base_value for i in x])
        return x

    def fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking feature x and transform it.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y=y, prefix=prefix)
        return self.transform(x, y=y)


class TargetEncoder(object):
    """
    User-defined target encoder which encodes features both categorical or continuous features (regarded as categorical feature)
    based on target of interest. 
    Mapping relations are stored in dmap.
    """

    def __init__(self,
                 random_noise=0.005,
                 smoothing_param=0.01,
                 random_seed=10,
                 unknown = '<NA>*',
                 unseen = '<UNSEEN>*'):
        """
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        random_noise: float
                Random noise
        smoothing_param: float
                Smotthing parameter
        random_seed: int
                Random seed
        unknown: str or int or float
                Used to denote NA
        unseen: str or int or float
                Used to denote unseen cases
        """
        self.x_dtypes = 'cav'
        self.random_noise = random_noise
        self.smoothing_param = smoothing_param
        self.random_seed = random_seed
        self.unknown = unknown
        self.unseen = unseen

    def fit(self, x, y, prefix=''):
        """
        Fit transformer by checking feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        if y is None:
            raise ValueError("y must not be none if TargetEncoder is chosen")
        x = np.array(pd.Series(x).fillna(self.unknown))
        y = np.array(y)
        self.dmap = {}
        self.classes_ = np.unique(x)
        np.random.seed(self.random_seed)
        self.bias = np.random.normal(0, self.random_noise, len(self.classes_))
        self.base_value = y.mean()
        for i, key in enumerate(self.classes_):
            l = y[x == key]
            num = sum(l) + self.smoothing_param * len(l) * self.base_value
            deno = len(l) + self.smoothing_param * len(l)
            value = num/deno + self.bias[i]
            self.dmap[key] = value
        self.dmap.update({self.unseen: self.base_value})
        self.dmap.update({self.unknown: self.base_value})
        self.feat_name = '%s_te'%prefix

    def transform(self, x, y=None):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x).fillna(self.unknown)
        x = np.array([self.dmap[i] if i in self.dmap.keys() else self.base_value for i in x])
        return x

    def fit_transform(self, x, y, prefix=''):
        """
        Fit transformer by checking feature x and transform it.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y, prefix=prefix)
        return self.transform(x, y=y)


class NaEncoder(object):
    """
    User-defined na encoder which encodes features according its missing value status.
    Mapping relations are stored in dmap.
    """

    def __init__(self,
                 base_value=1,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*'):
        """
        x_dtypes: str
                Types of input feature x, can be 'cat' or 'cov', representing
                'categorical feature' and 'continuous feature respectively'
        base_value: int or float
                Default value for unseen values
        unknown: str
                String or value used to denote NA
        unseen: str or int or float
                String or value used to denote unseen cases
        """
        self.x_dtypes = 'cav'
        self.base_value = base_value
        self.unknown = unknown
        self.unseen = unseen

    def fit(self, x, y=None, prefix=''):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable
        prefix: str
                prefix (eg. original column name)

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.feat_name = '%s_ne'%prefix

    def transform(self, x, y=None, prefix=''):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        x = pd.Series(x)
        x = np.array(x.isnull().astype(int))
        if self.base_value!=1:
            x = 1-x
        return x

    def fit_transform(self, x, y=None, prefix=''):
        """
        Encode feature x.

        Parameters
        ----------
        x: array-like of shape (n_samples,)
                Independent variable to be encoded
        y: array-like of shape (n_samples,)
                Dependent variable

        Returns
        ----------
        x: array-like of shape (n_samples,)
                Encoded feature
        """
        self.fit(x, y=y, prefix=prefix)
        return self.transform(x, y=y, prefix=prefix)


class FeatureEncoder:
    """
    User-defined encoder can encode features seperately based on the assigned strategies.
    Encoding strategies for each feature are stored in encodr.
    """

    def __init__(self,
                 target=[],
                 use_woe_encoder=True,
                 we_cut_method='dt',
                 be_cut_method='dt',
                 drop_features=True,
                 unknown='<NA>*',
                 unseen='<UNSEEN>*',
                 silent=False,
                 recoding_dict=None,
                 feat_dict=None):
        """
        Parameters
        ----------
        target: list
                List of default boolean targets
        use_woe_encoder: boolean
                If True, use woe encoder for each feature
        we_cut_method: str
                Cut method for woe encoder
        be_cut_method: str
                Cut method for bin encoder
        drop_features: boolean
                If True, drop original feature after encoded
        unknown: str or int or float
                Used to denote NA
        unseen: str or int or float
                Used to denote unseen cases
        silent: boolean
                If True, restrict the print of encoding process
        recoding_dict: dict
                Dictionary recording recoding statements for each feature
        feat_dict: dict
                Dictionary recording changes of feature names
        final_WoeEncoder_feat: list
                List of feats encoded using WoeEncoder
        final_BinEncoder_feat: list
                List of feats encoded using BinEncoder
        final_CountEncoder_feat: list
                List of feats encoded using CountEncoder
        final_OneHotEncoder_feat: list
                List of feats encoded using OneHotEncoder
        final_TargetEncoder_feat: list
                List of feats encoded using TargetEncoder
        final_NaEncoder_feat: list
                List of feats encoded using WoeEncoder

        """
        self.target = target
        self.use_woe_encoder = use_woe_encoder
        self.we_cut_method = we_cut_method
        self.be_cut_method = be_cut_method
        self.drop_features = drop_features
        self.unknown = unknown
        self.unseen = unseen
        self.silent = silent
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict
        self.final_WoeEncoder_feat = []
        self.final_BinEncoder_feat = []
        self.final_CountEncoder_feat = []
        self.final_OneHotEncoder_feat = []
        self.final_TargetEncoder_feat = []
        self.final_NaEncoder_feat = []

    def write_recoding_statement(self, encodr, ori_name):
        """
        Parameters
        ----------
        encodr: encoder
                Input encoder 
        ori_name: str
                original name of the encoded feature
        """
        # statements for OneHotEncoder
        if str(type(encodr)) == '<class \'feature_encoding.OneHotEncoder_ud\'>':
            try:
                recoding_statement = "\ndf['%s'] = df['%s'].fillna('%s').astype(str)"%(ori_name, ori_name, self.unknown)
                for value in encodr.dmap.values():
                    recoding_statement_group1 = "df.loc[:, '%s_%s'] = 0"%(ori_name, str(value))
                    recoding_statement_group2 = "df.loc[df['%s'] == '%s', '%s_%s'] = 1"%(ori_name, str(value), ori_name, str(value))
                    recoding_statement += "\n" + recoding_statement_group1 + "\n" + recoding_statement_group2
                return recoding_statement
            except Exception as e:
                print(e)
                raise ValueError("Recoding statement is incorrect for OneHotEncoder_ud")

        # statements for NaEncoder
        elif str(type(encodr)) == '<class \'feature_encoding.NaEncoder\'>':
            try:
                recoding_statement = ""
                recoding_statement += "\ndf['%s'] = df['%s'].isnull().astype(int)"%(encodr.feat_name, ori_name)
                return recoding_statement
            except Exception as e:
                print(e)
                raise ValueError("Recoding statement is incorrect for NaEncoder")

        # statements for continuous variables
        elif encodr.x_dtypes == 'cov':
            try:
                recoding_statement = ""
                recoding_statement += "\ndf.loc[:, '%s'] = %s"%(encodr.feat_name, str(encodr.dmap.get(self.unseen)))
                recoding_statement += "\ndf.loc[df['%s'].isnull(), '%s'] = %s"%(ori_name, encodr.feat_name, str(encodr.dmap.get(self.unknown)))
                recoding_statement += "\ndf['%s'] = np.round(df['%s'], 6)"%(ori_name, ori_name)
                dmap_tmp = encodr.dmap.copy()
                dmap_tmp.pop(self.unknown)
                dmap_tmp.pop(self.unseen)
                for interval in sorted(dmap_tmp):
                    value = dmap_tmp[interval]
                    recoding_statement_group = ''
                    if interval.closed == 'left':
                        recoding_statement_group += "\ndf.loc[(%s <= df['%s']) & (df['%s'] < %s), '%s'] = %s"%(
                                                    str(interval.left).replace('inf', 'np.inf'), ori_name, ori_name, 
                                                    str(interval.right).replace('inf', 'np.inf'), encodr.feat_name, str(value))
                    elif interval.closed == 'right':
                        recoding_statement_group += "\ndf.loc[(%s < df['%s']) & (df['%s'] <= %s), '%s'] = %s"%(
                                                    str(interval.left).replace('inf', 'np.inf'), ori_name, ori_name, 
                                                    str(interval.right).replace('inf', 'np.inf'), encodr.feat_name, str(value))
                    recoding_statement += "\n" + recoding_statement_group
                return recoding_statement

            except Exception as e:
                print(e)
                raise ValueError("Recoding statement is incorrect for encoding continuous variable")

        # statements for categorical variables
        elif encodr.x_dtypes == 'cav':
            try:
                recoding_statement = ""
                recoding_statement += "\ndf['%s'] = df['%s'].fillna('%s')"%(ori_name, ori_name, str(encodr.unknown))
                recoding_statement += "\ndf.loc[:, '%s'] = %s"%(encodr.feat_name, str(encodr.dmap.get(self.unseen)))
                df_map = pd.DataFrame([encodr.dmap], index=['value']).T.drop([self.unseen], axis=0)
                df_map = df_map.reset_index(drop=False).set_index('value')
                for value in df_map.index.unique():
                    if type(df_map.loc[value, 'index']) != pd.Series:
                        group_list = str([df_map.loc[value, 'index']])
                    else:
                        group_list = str(df_map.loc[value, 'index'].values.tolist())
                    recoding_statement_group = "df.loc[[x in %s for x in df['%s']],'%s'] = %s"%(group_list, 
                                                        ori_name, encodr.feat_name, str(value))
                    recoding_statement += "\n" + recoding_statement_group
                return recoding_statement
            except Exception as e:
                print(e)
                raise ValueError("Recoding statement is incorrect for encoding other categorical variable")

    def fit(self, df, label, WoeEncoder_feat=[], BinEncoder_feat=[], CountEncoder_feat=[],
            OneHotEncoder_feat=[], TargetEncoder_feat=[], NaEncoder_feat=[], exclude_list=[]):
        """
        Fit encoders for each feature in the dataframe.

        Parameters
        ----------
        df: array-like of shape (n_samples,)
                Feature to be imputed
        label: str
                Label will be used to train models
        WoeEncoder_feat: list
                List of features manually chosen to apply woe encoder
        BinEncoder_feat: list
                List of features manually chosen to apply bin encoder
        CountEncoder_feat: list
                List of features manually chosen to apply count encoder
        OneHotEncoder_feat: list
                List of features manually chosen to apply onehot encoder
        TargetEncoder_feat: list
                List of features manually chosen to apply onehot encoder
        NaEncoder_feat: list
                List of features manually chosen to apply na encoder
        exclude_list: list
                List of features excluded from being encoded
        """
        if self.use_woe_encoder:
            manual_feat = WoeEncoder_feat + BinEncoder_feat + CountEncoder_feat + \
                          TargetEncoder_feat + OneHotEncoder_feat + NaEncoder_feat
            auto_WoeEncoder_feat = list(set(df.columns.tolist()) - set(
                self.target + [label] + exclude_list + manual_feat))

            for i in auto_WoeEncoder_feat:
                # try:
                    if df[i].dtypes==object:
                        we_tmp = WoeEncoder(x_dtypes='cav', cut_method=self.we_cut_method, unknown=self.unknown, unseen=self.unseen)
                    else:
                        we_tmp = WoeEncoder(x_dtypes='cov', cut_method=self.we_cut_method, unknown=self.unknown, unseen=self.unseen)
                    we_tmp.fit_transform(df[i], df[label], prefix=i)
                    self.__dict__.update({'%s_encoder'%i: we_tmp})
                    self.final_WoeEncoder_feat += [i]
                    if not self.silent:
                        print('Apply WoeEncoder to encode {i}'.format(i=i))
                # except Exception as e:
                #     print("Failed to fit WoeEncoder for %s"%i)
                #     print(e)

        for i in WoeEncoder_feat:
            try:
                if df[i].dtypes==object:
                    we_tmp = WoeEncoder(x_dtypes='cav', cut_method=self.we_cut_method, unknown=self.unknown, unseen=self.unseen)
                else:
                    we_tmp = WoeEncoder(x_dtypes='cov', cut_method=self.we_cut_method, unknown=self.unknown, unseen=self.unseen)
                we_tmp.fit_transform(df[i], df[label], prefix=i)
                self.__dict__.update({'%s_encoder'%i: we_tmp})
                self.final_WoeEncoder_feat += [i]
                if not self.silent:
                    print('Apply WoeEncoder to encode {i}'.format(i=i))
            except Exception as e:
                    print("Failed to fit WoeEncoder for %s"%i)
                    print(e)

        for i in BinEncoder_feat:
            try:
                be_tmp = BinEncoder(cut_method=self.be_cut_method, unknown=self.unknown, unseen=self.unseen)
                be_tmp.fit_transform(df[i], y=df[label], prefix=i)
                self.__dict__.update({'%s_encoder'%i: be_tmp})
                self.final_BinEncoder_feat += [i]
                if not self.silent:
                    print('Apply BinEncoder to encode {i}'.format(i=i))
            except Exception as e:
                print("Failed to fit BinEncoder for %s"%i)
                print(e)

        for i in CountEncoder_feat:
            try:
                ce_tmp = CountEncoder(unknown=self.unknown, unseen=self.unseen)
                ce_tmp.fit_transform(df[i], prefix=i)
                self.__dict__.update({'%s_encoder'%i: ce_tmp})
                self.final_CountEncoder_feat += [i]
                print('Apply CountEncoder to encode {i}'.format(i=i))
            except Exception as e:
                if not self.silent:
                    print("Failed to fit CountEncoder for %s"%i)
                print(e)

        for i in OneHotEncoder_feat:
            try:
                oe_tmp = OneHotEncoder_ud(unknown=self.unknown, unseen=self.unseen)
                oe_tmp.fit_transform(df[i], prefix=i)
                self.__dict__.update({'%s_encoder'%i: oe_tmp})
                self.final_OneHotEncoder_feat += [i]
                print('Apply OneHotEncoder to encode {i}'.format(i=i))
            except Exception as e:
                if not self.silent:
                    print("Failed to fit OneHotEncoder for %s"%i)
                print(e)

        for i in TargetEncoder_feat:
            try:
                te_tmp = TargetEncoder(unknown=self.unknown, unseen=self.unseen)
                te_tmp.fit_transform(df[i], df[label], prefix=i)
                self.__dict__.update({'%s_encoder'%i: te_tmp})
                self.final_TargetEncoder_feat += [i]
                if not self.silent:
                    print('Apply TargetEncoder to encode {i}'.format(i=i))
            except Exception as e:
                print("Failed to fit TargetEncoder for %s"%i)
                print(e)

        for i in NaEncoder_feat:
            try:
                ne_tmp = NaEncoder()
                ne_tmp.fit_transform(df[i], prefix=i)
                self.__dict__.update({'%s_encoder'%i: ne_tmp})
                self.final_NaEncoder_feat += [i]
                if not self.silent:
                    print('Apply NaEncoder to encode {i}'.format(i=i))
            except Exception as e:
                print("Failed to fit NaEncoder for %s"%i)
                print(e)

    def transform(self, df, label=None, exclude_list=[], write_recoding_statement=False):
        """
        Try to encode each feature in the dataframe.

        Parameters
        ----------
        df: pd.DataFrame
                The input dataframe where features need to be encoded
        label: str
                Label will be used to train models
        exclude_list: list
                List of features excluded from being encoded
        write_recoding_statement: boolean
                If True, write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Dataframe where features are encoded
        """
        df = df.reset_index(drop=True)
        feat_to_transform = set(self.final_WoeEncoder_feat + self.final_BinEncoder_feat + \
                                self.final_CountEncoder_feat + self.final_TargetEncoder_feat + \
                                self.final_OneHotEncoder_feat + self.final_NaEncoder_feat) - set(exclude_list)
        for i in feat_to_transform:
            try:
                ori_name = i
                encodr = self.__dict__.get('%s_encoder'%i)
                if type(encodr.feat_name) == str:  # not OneHotEncoded
                    df_tmp = pd.DataFrame(encodr.transform(df[i]), columns=[encodr.feat_name])
                else:                              # OneHotEncoded
                    df_tmp = pd.DataFrame(encodr.transform(df[i]), columns=encodr.feat_name)
                if self.drop_features:
                    del df[i]
                df = pd.concat([df, df_tmp], axis=1)

                if write_recoding_statement and self.recoding_dict is not None and self.feat_dict is not None:
                    recoding_statement = self.write_recoding_statement(encodr, ori_name)
                    key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                    self.feat_dict.update({key: encodr.feat_name})
                    self.recoding_dict[key] += recoding_statement
            except Exception as e:
                print(e)
                print("Failed to encode %s"%i)

        self.final_OneHotEncoder_new_feat = []
        for i in range(len(self.final_OneHotEncoder_feat)):
            self.final_OneHotEncoder_new_feat += self.feat_dict[self.final_OneHotEncoder_feat[i]]
        return df

    def fit_transform(self, df, label, WoeEncoder_feat=[], BinEncoder_feat=[], CountEncoder_feat=[],
                      OneHotEncoder_feat=[], TargetEncoder_feat=[], NaEncoder_feat=[], exclude_list=[],
                      write_recoding_statement=True):
        """
        Try to fit encoders for each feature and encode each feature.

        Parameters
        ----------
        df: array-like of shape (n_samples,)
                Feature to be imputed
        label: str
                Label will be used to train models
        WoeEncoder_feat: list
                List of features manually chosen to apply woe encoder
        BinEncoder_feat: list
                List of features manually chosen to apply bin encoder
        CountEncoder_feat: list
                List of features manually chosen to apply count encoder
        OneHotEncoder_feat: list
                List of features manually chosen to apply onehot encoder
        TargetEncoder_feat: list
                List of features manually chosen to apply onehot encoder
        NaEncoder_feat: list
                List of features manually chosen to apply na encoder
        exclude_list: list
                List of features excluded from being encoded
        write_recoding_statement: dict
                If True, write recoding statement

        Returns
        ----------
        df: pd.DataFrame
                Dataframe where features are encoded
        """
        self.fit(df, label, WoeEncoder_feat=WoeEncoder_feat, BinEncoder_feat=BinEncoder_feat,
                 CountEncoder_feat=CountEncoder_feat, OneHotEncoder_feat=OneHotEncoder_feat,
                 TargetEncoder_feat=TargetEncoder_feat, NaEncoder_feat=NaEncoder_feat,
                 exclude_list=exclude_list)
        return self.transform(df, label, exclude_list=exclude_list, write_recoding_statement=write_recoding_statement)
