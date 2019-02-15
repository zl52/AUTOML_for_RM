import pandas as pd;
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import bisect
from collections import defaultdict, Counter
from scipy import stats
import math

from tools import *


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   8. FEATURE ENCODING  ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def decision_tree_cutpoints(x, y, max_depth=4, min_samples_leaf=0.05, max_leaf_nodes=10,
                            monotonic_bin=False, spearman_cor_thr=0.1, random_state=7):
    """
    A decision tree method which bins continuous features to categorical ones

    : param x: input samples
    : param y: label
    : param max_depth: maximum depth of the tree
    : param min_samples_leaf: minimum number of samples required to be at a leaf node
    : param max_leaf_nodes: grow a tree with max_leaf_nodes in best-first fashion. 
                            Best nodes are defined as relative reduction in impurity.
                            If None then unlimited number of leaf nodes.
    : param monotonic_bin: whether to order bins monotonically based on target values
    : param spearman_cor_thr: threshold of spearman correlation
    : param random_state: seed

    : return: list of cut points
    """
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
        try:
            dt.fit(np.array(x).reshape(-1, 1), np.array(y))

        except:
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float32')")

        thr = dt.tree_.threshold
        cut_points = sorted(thr[np.where(thr != -2)])

    else:
        while np.abs(r) < spearman_cor_thr and max_leaf_nodes > 1:
            dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        max_leaf_nodes=max_leaf_nodes, random_state=random_state)
            try:
                dt.fit(np.array(x).reshape(-1, 1), np.array(y))

            except:
                raise ValueError("Input contains NaN, infinity or a value too large for dtype('float32')")

            thr = dt.tree_.threshold
            cut_points = sorted(thr[np.where(thr != -2)])
            df['bin_range'] = pd.cut(df['feature'], cut_points, include_lowest=True)
            df_na['bin_range'] = '_MISSING_'
            df = pd.concat([df, df_na]).reset_index(drop=True)
            stat = df.groupby('bin_range', as_index=True)
            r, _ = stats.spearmanr(stat.mean().feature, stat.mean().label)
            max_leaf_nodes -= 1

    return cut_points


def get_cut_points(x, y=None, bins=10, cut_method='dt', precision=8, **kwargs):
    """
    Get cut points by different methods. Supported methods include 'cut', 'qcut' and 'dt', standing for
    'cut by defined cut points', 'equally cut' and 'cut by decision tree cutpoints' respectively

    : param x: input samples
    : param y: label
    : param bins: number of bins or defined cutpoints when 'cut' method is applied
    : param cut_method: cut method ('cut'(defualt), 'qcut' or 'dt')
    : param precision: precision

    : return: list of cut points
    """
    if cut_method == 'cut':
        _, cut_points = pd.cut(x, bins=bins, retbins=True, precision=precision)

    elif cut_method == 'qcut':
        _, cut_points = pd.qcut(x, q=bins, retbins=True, duplicates='drop', precision=precision)

    elif (cut_method == 'dt') & (y is not None):
        cut_points = decision_tree_cutpoints(x, y, **kwargs)

    elif y is None:
        raise ValueError("y must not be none if \'dt\' is chosen")

    else:
        raise ValueError("cut_method must chosen among \'cut\',\'qcut\' and \'dt\'")

    if cut_method != 'dt':
        cut_points = cut_points[1:-1]

    cut_points = list(cut_points)
    cut_points.append(np.inf)
    cut_points.insert(0, -np.inf)

    return cut_points


def woe(x, y, woe_min=-20, woe_max=20):
    """
    Get feature's woe encoding dictionary

    : param x: feature to be encoded, array-like of shape (n_samples,)
    : param y: label
    : param woe_min: minimum of woe value
    : param woe_max: maximum of woe value

    : return dmap: woe dictionary
    : return pp_map: dictionary of each bin's positive-count's proportion 
    : return np_map: dictionary of each bin's negative-count's proportion 
    """
    x = np.array(x)
    y = np.array(y)

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    dmap, pp_map, np_map = {}, {}, {}

    for k in np.unique(x):
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


class UD_LabelEncoder(LabelEncoder):
    """
    User-defined label encoder which can encode both categorical or continuous features
    and deal with unseen value (overcome drawbacks of sklearn's label encoder).
    Mapping relations are stored in dmap
    """

    def __init__(self, unknown='<NA>*'):
        self.unknown = unknown
        """
        : param unknown: string or value used to denote NA
        """

    def ud_fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        self.fillna = False
        x = pd.Series(x)

        if x.dtypes == object:
            x.fillna(self.unknown, inplace=True)
            self.fillna = True

        l = list(x.unique())
        unseen_str = '<UNSEEN>'
        unseen_int = -9999

        while (unseen_str in l):
            unseen_str = unseen_str + '*'

        while (unseen_int in l):
            unseen_int = unseen_int + 0.0001

        le = LabelEncoder()
        le.fit(x)
        le_classes = le.classes_.tolist()

        try:
            bisect.insort_left(le_classes, unseen_str)
            self.unseen = unseen_str

        except:
            bisect.insort_left(le_classes, unseen_int)
            self.unseen = unseen_int

        le.classes_ = le_classes
        self.encoder = le
        self.dmap = dict(zip(self.encoder.transform(l), l))
        self.feat_name = prefix + '_le'

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        if self.fillna == True:
            x = pd.Series(x).fillna(self.unknown)

        x = [s if s in self.encoder.classes_ else self.unseen for s in x]

        return self.encoder.transform(x)

    def ud_fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x and transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        self.ud_fit(x, y=None, prefix=prefix)

        return self.ud_transform(x)


class UD_OneHotEncoder(OneHotEncoder):
    """
    User-defined onehot encoder which can encode both categorical or continuous feature
    and deal with unseen value (overcome drawbacks of sklearn's OneHot encoder)
    mapping relations are stored in dmap
    """

    def __init__(self):
        """
        : param x_dtypes: types of input feature x, can be 'cat' or 'cov', representing
                          'categorical feature' and 'continuous feature respectively'
        """
        self.x_dtypes = 'cav'

    def ud_fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        x = pd.Series(x).astype(str)
        le = UD_LabelEncoder()
        le.ud_fit(x)
        self.le = le
        self.dmap = le.dmap
        x = self.le.ud_transform(x)
        onehot = OneHotEncoder()
        onehot.fit(x.reshape(-1, 1))
        self.encoder = onehot
        self.active_features_ = onehot.active_features_
        self.feat_name = [prefix + '_' + str(self.dmap[i]) for i in self.active_features_]

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples, n_categories)
        """
        x = pd.Series(x).astype(str)
        x = self.le.ud_transform(x)

        return self.encoder.transform(x.reshape(-1, 1)).toarray()

    def ud_fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x and transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples, n_categories)
        """
        self.ud_fit(x, y=None, prefix=prefix)

        return self.ud_transform(x)


class UD_CountEncoder(object):
    """
    User-defined count encoder which replaces categorical features with its counts and replace
    unseen values with 1. Can use log-transform to avoid the impact of outliers.
    Mapping relations are stored in dmap
    """

    def __init__(self, base_value=1, use_log_transform=False, smoothing_param=1, unknown='<NA>*'):
        """
        : param x_dtypes: types of input feature x, can be 'cat' or 'cov', representing
                          'categorical feature' and 'continuous feature respectively'
        : param base_value: count when it occurs unseen case 
        : param use_log_transform: whether to use log transformation
        : param smoothing_param: smoothing parameter
        : param unknown: string or value used to denote NA
        """
        self.x_dtypes = 'cav'
        self.base_value = base_value
        self.use_log_transform = use_log_transform
        self.smoothing_param = 1
        self.unknown = unknown

    def ud_fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        x = pd.Series(x).fillna(self.unknown)
        self.dmap = Counter(x)
        self.feat_name = prefix + '_ce'
        self.dmap.update({'<UNSEEN>*': self.base_value})
        if not self.unknown in self.dmap.keys():
            self.dmap.update({'<UNSEEN>*': self.base_value})

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        x = pd.Series(x).fillna(self.unknown)
        x = np.array([self.dmap[i] + self.smoothing_param if i in self.dmap.keys() \
                          else self.base_value for i in x])

        if self.use_log_transform:
            x = np.log(x)

        return x

    def ud_fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x and transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        self.ud_fit(x, y=None, prefix=prefix)

        return self.ud_transform(x)


class UD_BinEncoder(object):
    """
    User-defined bin encoder which converts continuous feature to discrete feature.
    Mapping relations are stored in dmap
    """

    def __init__(self, bins=10, cut_method='dt', labels=None, interval=True, unknown='np.nan',
                 **kwargs):
        """
        : param x_dtypes: types of input feature x, can be 'cat' or 'cov', representing
                          'categorical feature' and 'continuous feature respectively'
        : param bins: number of bins
        : param cut_method: cut method ('cut'(defualt), 'qcut' or 'dt')
        : param labels: category names for bins
        : param interval: if interval is True, param labels is deactivated.
        : param unknown: string or value used to denote NA
        : param kwargs: params for decision tree.
        """
        self.x_dtypes = 'cov'
        self.bins = bins
        self.labels = labels
        self.interval = interval
        self.cut_method = cut_method
        self.unknown = unknown
        self.kwargs = kwargs

    def ud_fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        if y is None:
            self.cut_method = 'qcut'

        if self.labels is not None and len(self.labels) != self.bins:
            raise ValueError('The length of bin labels must be equal to the number of bins.')

        self.cut_points = get_cut_points(x, y, self.bins, self.cut_method, **self.kwargs)

        if self.interval and self.labels is None:
            self.labels = np.arange(len(self.cut_points) - 1)
            self.dmap = dict(zip(pd.cut(x, self.cut_points).unique().sort_values(), self.labels))
            self.base_value = 'np.nan'
            self.dmap.update({'<UNSEEN>*': self.base_value})
            self.dmap.update({self.unknown: self.base_value})

        self.feat_name = prefix + '_be'

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        x = pd.cut(x, bins=self.cut_points, labels=self.labels)

        return np.array(x)

    def ud_fit_transform(self, x, y, prefix=''):
        """
        Fit transformer by checking x and transform x 

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        self.ud_fit(x, y, prefix=prefix)

        return self.ud_transform(x)


class UD_WoeEncoder(object):
    """
    User-defined woe encoder which converts both categorical and continuous features
    to discrete features.
    Mapping relations are stored in dmap
    """

    def __init__(self, x_dtypes, cut_method='dt', bins=10, woe_min=-20, woe_max=20, labels=None,
                 unknown='<NA>*', **kwargs):
        """
        Fit transformer by checking x

        : param x_dtypes: types of input feature x, can be 'cat' or 'cov', representing
                          'categorical feature' and 'continuous feature respectively'
        : param cut_method: cut method ('cut'(defualt), 'qcut', 'dt' and None)
        : param bins: number of bins
        : param woe_min: minimum of woe value
        : param woe_max: maximum of woe value
        : param unknown: string or value used to denote NA
        : param kwargs: params for decision tree
        """
        self.x_dtypes = x_dtypes
        self.cut_method = cut_method
        self.bins = bins
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.labels = labels
        self.unknown = unknown
        self.kwargs = kwargs

    def ud_fit(self, x, y, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        if y is None:
            raise Exception('Encoder need valid y label.')

        if self.x_dtypes == 'cov':
            self.unknown = np.mean(x)
        x = np.array(pd.Series(x).fillna(self.unknown))

        if self.cut_method is None:
            if len(x) / len(np.unique(x)) > 10:
                print('Cut method is None and the feature is {s}'.format(s=self.x_dtypes))
                self.dmap, _, _ = woe(x, y, self.woe_min, self.woe_max)

            else:
                self.cut_method = 'dt'
                print('{i} samples are too few for {j} categories').format(i=len(x), j=len(np.unique(x)))

        else:
            if self.x_dtypes == 'cav':
                self.cut_method = 'dt'
                self.dmap, _, _ = woe(x, y, self.woe_min, self.woe_max)
                x_tmp = [self.dmap[i] for i in x]
                self.cut_points = get_cut_points(x_tmp, y, self.bins, self.cut_method, **self.kwargs)
                x_tmp = pd.cut(x_tmp, self.cut_points)
                dmap2, _, _ = woe(x_tmp, y, self.woe_min, self.woe_max)

                for i in self.dmap.items():
                    self.dmap[i[0]] = dmap2[pd.cut([i[1]], self.cut_points)[0]]

            elif self.x_dtypes == 'cov':
                be = UD_BinEncoder(self.bins, self.cut_method, self.labels, interval=False)
                be.ud_fit(x, y, prefix)
                x = be.ud_transform(x)
                self.cut_points = be.cut_points
                self.dmap, _, _ = woe(x, y, self.woe_min, self.woe_max)

            else:
                raise ValueError("x_dtypes must chosen between \'nomi\' and \'cont\'")

        self.base_value = 0
        self.dmap.update({'<UNSEEN>*': self.base_value})
        if not self.unknown in self.dmap.keys():
            self.dmap.update({'<UNSEEN>*': self.base_value})

        self.feat_name = prefix + '_we'

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        x = np.array(pd.Series(x).fillna(self.unknown))

        if (self.cut_method is not None) & (self.x_dtypes == 'cov'):
            x = pd.cut(x, bins=self.cut_points)

        x = np.array([self.dmap[i] if i in self.dmap.keys() else self.base_value for i in x])

        return x

    def ud_fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x and transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        self.ud_fit(x, y=y, prefix=prefix)

        return self.ud_transform(x)


class UD_TargetEncoder(object):
    """
    User-defined target encoder which encodes features by target.
    Only support binary classification and regression.
    mapping relations are stored in dmap
    """

    def __init__(self, random_noise=0.005, smoothing_param=0.01, random_seed=10, unknown='<NA>*'):
        """
        : param x_dtypes: types of input feature x, can be 'cat' or 'cov', representing
                          'categorical feature' and 'continuous feature respectively'
        : param random_noise: random noise
        : param smoothing_param: smotthing parameter
        : param random_seed: seed
        : param unknown: string or value used to denote NA
        """
        self.x_dtypes = 'cav'
        self.random_noise = random_noise
        self.smoothing_param = smoothing_param
        self.random_seed = random_seed
        self.unknown = unknown

    def ud_fit(self, x, y, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        if y is None:
            raise Exception('Encoder need valid y label.')

        x = pd.Series(x).fillna(self.unknown)
        x = np.array(x)
        y = np.array(y)
        self.dmap = {}

        self.classes_ = np.unique(x)
        np.random.seed(self.random_seed)
        self.bias = np.random.normal(0, self.random_noise, len(self.classes_))
        self.base_value = y.mean()

        for i, key in enumerate(self.classes_):
            l = y[x == key]
            value = (sum(l) + self.smoothing_param * len(l) * self.base_value) \
                    / (len(l) + self.smoothing_param * len(l))
            value += self.bias[i]
            self.dmap[key] = value

        self.dmap.update({'<UNSEEN>*': self.base_value})
        if not self.unknown in self.dmap.keys():
            self.dmap.update({'<UNSEEN>*': self.base_value})
        self.feat_name = prefix + '_te'

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        x = pd.Series(x).fillna(self.unknown)
        x = np.array([self.dmap[i] if i in self.dmap.keys() else self.base_value for i in x])

        return x

    def ud_fit_transform(self, x, y, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        self.ud_fit(x, y, prefix=prefix)

        return self.ud_transform(x)


class UD_NaEncoder(object):
    """
    User-defined na encoder which encodes features according its missing value status
    mapping relations are stored in dmap
    """

    def __init__(self, base_value=1, unknown='<NA>*'):
        """
        : param x_dtypes: types of input feature x, can be 'cat' or 'cov', representing
                          'categorical feature' and 'continuous feature respectively'
        : param base_value: value when it occurs unseen case 
        : param unknown: string or value used to denote NA
        """
        self.x_dtypes = 'cav'
        self.base_value = base_value
        self.unknown = unknown

    def ud_fit(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)
        """
        x = pd.Series(x).copy()

        if x.isnull().sum() == 0:
            raise ValueError("Feature has no NAs")

        classes_ = pd.Series(x.unique())
        classes_1 = classes_[classes_.notnull()]
        classes_2 = classes_[classes_.isnull()]

        self.dmap = {}

        for i, key in enumerate(classes_1):
            self.dmap[key] = 1

        for i, key in enumerate(classes_2):
            self.dmap[key] = 0

        self.dmap.update({'<UNSEEN>*': self.base_value})
        self.dmap.update({self.unknown: 'np.nan'})
        self.feat_name = prefix + '_ne'

    def ud_transform(self, x):
        """
        Transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        x = pd.Series(x)
        x = np.array(x.isnull().astype(int))

        if sum(x) == len(x):
            raise ValueError("NaEncoder is not working properly")

        return x

    def ud_fit_transform(self, x, y=None, prefix=''):
        """
        Fit transformer by checking x and transform x

        : param x: feature to be encoded, array-like of shape (n_samples,)
        : param y: label
        : param prefix: prefix (eg. original column name)

        : return x: encoded feature, array-like of shape (n_samples,)
        """
        self.ud_fit(x, y=None, prefix=prefix)

        return self.ud_transform(x)


class UD_FEATURE_ENCODER():
    """
    User-defined imputer which imputes features seperately using defined imputation strategy
    imputation strategies for each feature are stored in imputr
    """

    def __init__(self, recoding_dict=None, feat_dict=None, target=TARGET, use_woe_encoder=True, we_cut_method='dt',
                 be_cut_method='dt',
                 drop_ori_feat=True):
        """
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : params target: list of default boolean targets
        : param use_woe_encoder: whether to use woe encoder for every feature
        : param we_cut_method: choose cut method for woe encoder
        : param be_cut_method: choose cut method for bin encoder
        : param drop_ori_feat: whether to drop original feature
        : param final_WoeEncoder_feat: feats transformed by WoeEncoder
        : param final_BinEncoder_feat: feats transformed by BinEncoder
        : param final_CountEncoder_feat: feats transformed by CountEncoder
        : param final_OneHotEncoder_feat: feats transformed by OneHotEncoder
        : param final_TargetEncoder_feat: feats transformed by TargetEncoder
        : param final_NaEncoder_feat: feats transformed by WoeEncoder
        """
        self.recoding_dict = recoding_dict
        self.feat_dict = feat_dict
        self.target = target
        self.use_woe_encoder = use_woe_encoder
        self.we_cut_method = we_cut_method
        self.be_cut_method = be_cut_method
        self.drop_ori_feat = drop_ori_feat
        self.final_WoeEncoder_feat = []
        self.final_BinEncoder_feat = []
        self.final_CountEncoder_feat = []
        self.final_OneHotEncoder_feat = []
        self.final_TargetEncoder_feat = []
        self.final_NaEncoder_feat = []

    def write_recoding_statement(self, encodr, ori_name):
        """
        : params encodr: input encoder 
        : param ori_name: original name of the feature to be transformed
        """
        if str(type(encodr)) == '<class \'feature_encoding.UD_OneHotEncoder\'>':
            try:
                # recoding_statement = "######### Recoding {i} using {j} ########" \
                #     .format(i=ori_name, j=type(encodr))
                recoding_statement = "\ndf['" + ori_name + "'] = df['" + ori_name + "'].astype(str)"

                for value in encodr.dmap.values():
                    recoding_statement_group1 = "df.loc[:, '" + ori_name + "_" + str(value) + "'] = 0"
                    recoding_statement_group2 = "df.loc[df['" + ori_name + "'] == '" + str(
                        value) + "', '" + ori_name + "_" + str(value) + "'] = 1"

                    recoding_statement += "\n" + recoding_statement_group1 + "\n" + recoding_statement_group2

                # print(recoding_statement + "\n")
                return recoding_statement

            except:
                raise ValueError("Recoding statement is incorrect")

        elif str(type(encodr)) == '<class \'feature_encoding.UD_NaEncoder\'>':
            try:
                # recoding_statement = "######### Recoding {i} using {j} ########" \
                #     .format(i=ori_name, j=type(encodr))
                recoding_statement = ""
                recoding_statement += "\n" + "df['" + encodr.feat_name + "'] = df['" + ori_name \
                                      + "'].isnull().astype(int)"

                # print(recoding_statement + "\n")
                return recoding_statement

            except:
                raise ValueError("Recoding statement is incorrect")

        elif encodr.x_dtypes == 'cov':
            try:
                # recoding_statement = "######### Recoding {i} using {j} ########" \
                #     .format(i=ori_name, j=type(encodr))
                recoding_statement = ""
                recoding_statement += "\n" + "df['" + ori_name + "'] = df['" + ori_name \
                                      + "'].fillna(" + str(encodr.unknown) + ")"
                recoding_statement += "\n" + "df.loc[:, '" + encodr.feat_name + "'] =" \
                                      + str(encodr.dmap.get('<UNSEEN>*'))

                for interval, value in encodr.dmap.items():
                    if type(interval) != str:
                        recoding_statement_group = "df.loc[(" + str(interval.left).replace('inf', 'np.inf') \
                                                   + " <= df['" + ori_name + "']) & (df['" + ori_name + "'] < " \
                                                   + str(interval.right).replace('inf', 'np.inf') + "),'" \
                                                   + encodr.feat_name + "'] = " + str(value)

                        recoding_statement += "\n" + recoding_statement_group

                # print(recoding_statement + "\n")
                return recoding_statement

            except:
                raise ValueError("Recoding statement is incorrect")

        elif encodr.x_dtypes == 'cav':
            try:
                # recoding_statement = "######### Recoding {i} using {j} ########" \
                #     .format(i=ori_name, j=type(encodr))
                recoding_statement = ""
                recoding_statement += "\n" + "df['" + ori_name + "'] = df['" + ori_name \
                                      + "'].fillna('" + str(encodr.unknown) + "')"
                recoding_statement += "\n" + "df.loc[:, '" + encodr.feat_name + "'] =" \
                                      + str(encodr.dmap.get('<UNSEEN>*'))
                df_map = pd.DataFrame([encodr.dmap], index=['value']).T.drop(['<UNSEEN>*'], axis=0)
                df_map = df_map.reset_index(drop=False).set_index('value')

                for value in df_map.index.unique():

                    if type(df_map.loc[value, 'index']) != pd.Series:
                        group_list = str([df_map.loc[value, 'index']])

                    else:
                        group_list = str(df_map.loc[value, 'index'].values.tolist())

                    recoding_statement_group = "df.loc[[x in " + group_list + " for x in df['" + \
                                               ori_name + "']],'" + encodr.feat_name + "'] = " + str(value)
                    recoding_statement += "\n" + recoding_statement_group

                    # print(recoding_statement + "\n")
                return recoding_statement

            except:
                raise ValueError("Recoding statement is incorrect")

    def ud_fit(self, df, label, WoeEncoder_feat=[], BinEncoder_feat=[], CountEncoder_feat=[],
               OneHotEncoder_feat=[], TargetEncoder_feat=[], NaEncoder_feat=[], exclude_list=[]):
        """
        Encode each feature in the dataframe

        : param df: feature to be imputed, array-like of shape (n_samples,)
        : param label: label will be used to train models
        : param WoeEncoder_feat: list of features manually chosen to apply woe encoder
        : param BinEncoder_feat: list of features manually chosen to apply bin encoder
        : param CountEncoder_feat: list of features manually chosen to apply count encoder
        : param OneHotEncoder_feat: list of features manually chosen to apply onehot encoder
        : param TargetEncoder_feat: list of features manually chosen to apply onehot encoder
        : param NaEncoder_feat: list of features manually chosen to apply na encoder
        : param exclude_list: list of features kept the same
        """
        if self.use_woe_encoder:
            feat_not_woetrans = WoeEncoder_feat + BinEncoder_feat + CountEncoder_feat + \
                                TargetEncoder_feat + OneHotEncoder_feat + NaEncoder_feat

            cav_list = list(set(df.select_dtypes(exclude=[float, int, 'int64']).columns.tolist()) - set(
                self.target + [label] + exclude_list + feat_not_woetrans))
            cov_list = list(set(df.select_dtypes(include=[float, int, 'int64']).columns.tolist()) - set(
                self.target + [label] + exclude_list + feat_not_woetrans))

            for i in cav_list:
                try:
                    we_tmp = UD_WoeEncoder(x_dtypes='cav', cut_method=self.we_cut_method)
                    we_tmp.ud_fit_transform(df[i], df[label], prefix=i)
                    self.__dict__.update({i + '_encoder': we_tmp})
                    print('Apply WoeEncoder to encode {i}'.format(i=i))
                    self.final_WoeEncoder_feat += [i]

                except:
                    raise ValueError("Failed to apply WoeEncoder")

            for i in cov_list:
                try:
                    we_tmp = UD_WoeEncoder(x_dtypes='cov', cut_method=self.we_cut_method)
                    we_tmp.ud_fit_transform(df[i], df[label], prefix=i)
                    self.__dict__.update({i + '_encoder': we_tmp})
                    print('Apply WoeEncoder to encode {i}'.format(i=i))
                    self.final_WoeEncoder_feat += [i]

                except:
                    raise ValueError("Failed to apply WoeEncoder")

        if WoeEncoder_feat != [] and not self.use_woe_encoder:
            for i in WoeEncoder_feat:
                if df[i].dtypes not in [float, int]:
                    try:
                        we_tmp = UD_WoeEncoder(x_dtypes='cav', cut_method=self.we_cut_method)
                        we_tmp.ud_fit_transform(df[i], df[label], prefix=i)
                        self.__dict__.update({i + '_encoder': we_tmp})
                        print('Apply WoeEncoder to encode {i}'.format(i=i))
                        self.final_WoeEncoder_feat += [i]

                    except:
                        raise ValueError("Failed to apply WoeEncoder")

                else:
                    try:
                        we_tmp = UD_WoeEncoder(x_dtypes='cov', cut_method=self.we_cut_method)
                        we_tmp.ud_fit_transform(df[i], df[label], prefix=i)
                        self.__dict__.update({i + '_encoder': we_tmp})
                        print('Apply WoeEncoder to encode {i}'.format(i=i))
                        self.final_WoeEncoder_feat += [i]

                    except:
                        raise ValueError("Failed to apply WoeEncoder")

        for i in BinEncoder_feat:
            try:
                be_tmp = UD_BinEncoder(cut_method=self.be_cut_method)
                be_tmp.ud_fit_transform(df[i], y=df[label], prefix=i)
                self.__dict__.update({i + '_encoder': be_tmp})
                print('Apply BinEncoder to encode {i}'.format(i=i))
                self.final_BinEncoder_feat += [i]

            except:
                raise ValueError("Failed to apply BinEncoder")

        for i in CountEncoder_feat:
            try:
                ce_tmp = UD_CountEncoder()
                ce_tmp.ud_fit_transform(df[i], prefix=i)
                self.__dict__.update({i + '_encoder': ce_tmp})
                print('Apply CountEncoder to encode {i}'.format(i=i))
                self.final_CountEncoder_feat += [i]

            except:
                raise ValueError("Failed to apply CountEncoder")

        for i in OneHotEncoder_feat:
            try:
                oe_tmp = UD_OneHotEncoder()
                oe_tmp.ud_fit_transform(df[i], prefix=i)
                self.__dict__.update({i + '_encoder': oe_tmp})
                print('Apply OneHotEncoder to encode {i}'.format(i=i))
                self.final_OneHotEncoder_feat += [i]

            except:
                raise ValueError("Failed to apply OneHotEncoder")

        for i in TargetEncoder_feat:
            try:
                te_tmp = UD_TargetEncoder()
                te_tmp.ud_fit_transform(df[i], df[label], prefix=i)
                self.__dict__.update({i + '_encoder': te_tmp})
                print('Apply TargetEncoder to encode {i}'.format(i=i))
                self.final_TargetEncoder_feat += [i]

            except:
                raise ValueError("Failed to apply TargetEncoder")

        for i in NaEncoder_feat:
            try:
                ne_tmp = UD_NaEncoder()
                ne_tmp.ud_fit_transform(df[i], prefix=i)
                self.__dict__.update({i + '_encoder': ne_tmp})
                print('Apply NaEncoder to encode {i}'.format(i=i))
                self.final_NaEncoder_feat += [i]

            except:
                raise ValueError("Failed to apply NaEncoder")

    def ud_transform(self, df, label=None, exclude_list=[], write=False):
        """
        Try to recode each feature in the dataframe

        : param df: the input dataframe where features need to be recoded
        : param label: label will be used to train models
        : param exclude_list: list of features kept the same
        : param write: whether to write recoding statement

        : return df: dataframe of encoded features
        """
        df = df.reset_index(drop=True)
        feat_to_transform = self.final_WoeEncoder_feat + self.final_BinEncoder_feat + \
                            self.final_CountEncoder_feat + self.final_TargetEncoder_feat + \
                            self.final_OneHotEncoder_feat + self.final_NaEncoder_feat
        for i in feat_to_transform:
            if i not in exclude_list:

                try:
                    ori_name = i
                    encodr = self.__dict__.get(i + '_encoder')

                    if type(encodr.feat_name) == str:
                        df_tmp = pd.DataFrame(encodr.ud_transform(df[i]), columns=[encodr.feat_name])
                    else:
                        df_tmp = pd.DataFrame(encodr.ud_transform(df[i]), columns=encodr.feat_name)

                    recoding_statement = self.write_recoding_statement(encodr, ori_name)
                    df = pd.concat([df, df_tmp], axis=1)

                    if write is True and self.recoding_dict is not None and self.feat_dict is not None:
                        key = list(self.feat_dict.keys())[list(self.feat_dict.values()).index(ori_name)]
                        self.feat_dict.update({key: encodr.feat_name})
                        self.recoding_dict[key] += recoding_statement

                    if self.drop_ori_feat:
                        del df[i]

                except:
                    print("Failed to transform", i)
            else:
                pass

        return df

    def ud_fit_transform(self, df, label, WoeEncoder_feat=[], BinEncoder_feat=[], CountEncoder_feat=[],
                         OneHotEncoder_feat=[], TargetEncoder_feat=[], NaEncoder_feat=[], exclude_list=[],
                         write=True):
        """
        Try to recode each feature in the dataframe

        : param df: the input dataframe where features need to be recoded
        : param label: label will be used to train models
        : param WoeEncoder_feat: list of features manually chosen to apply woe encoder
        : param BinEncoder_feat: list of features manually chosen to apply bin encoder
        : param CountEncoder_feat: list of features manually chosen to apply count encoder
        : param OneHotEncoder_feat: list of features manually chosen to apply onehot encoder
        : param TargetEncoder_feat: list of features manually chosen to apply onehot encoder
        : param NaEncoder_feat: list of features manually chosen to apply na encoder
        : param exclude_list: list of features kept the same
        : param write: whether to write recoding statement

        : return df: dataframe of encoded features
        """
        self.ud_fit(df, label, WoeEncoder_feat=WoeEncoder_feat, BinEncoder_feat=BinEncoder_feat,
                    CountEncoder_feat=CountEncoder_feat, OneHotEncoder_feat=OneHotEncoder_feat,
                    TargetEncoder_feat=TargetEncoder_feat, NaEncoder_feat=NaEncoder_feat,
                    exclude_list=exclude_list)

        return self.ud_transform(df, label, exclude_list=exclude_list, write=write)
