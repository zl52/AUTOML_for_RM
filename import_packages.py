# 0.path

import os
import sys
# path = os.path.join(os.path.abspath('').rsplit('Work', 1)[0], 'Work')
# path_packages = os.path.join(os.path.abspath('').rsplit('Work', 1)[0], 'Work','packages')
# sys.path.append(path)
# sys.path.append(path_packages)

# 1.basics

import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
pd.set_option('max_rows', 500)
pd.set_option('max_columns', 200)

# 2.matplotlib

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
# myfont = mpl.font_manager.FontProperties(fname="/Library/Fonts/SimHei.ttf")  # "/Library/Fonts/Songti.ttc")
# mpl.rcParams["axes.unicode_minus"] = False
mpl.style.use('ggplot')

# 3.sklearn

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.feature_selection import RFECV, f_regression

from sklearn import metrics
from sklearn.metrics import log_loss, roc_curve, auc, make_scorer, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.utils.multiclass import type_of_target
from sklearn.calibration import CalibratedClassifierCV as CCCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# 4. others

try:
    import pyecharts
except:
    os.system("pip install pyecharts")
    import pyecharts
try:
    from bayes_opt import BayesianOptimization
except:
    os.system("pip install bayesian-optimization")
    from bayes_opt import BayesianOptimization
try:
    import missingno as msno
except:
    os.system("pip install missingno")
    import missingno as msno
try:
    import jieba
    import jieba.posseg as pseg
except:
    os.system("pip install jieba")
    import jieba
    import jieba.posseg as pseg

try:
    import xgboost as xgb
    import xgboost.sklearn
    from xgboost import XGBClassifier
except:
    os.system("pip install xgboost")
    import xgboost as xgb
    import xgboost.sklearn
    from xgboost import XGBClassifier

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    import statsmodels.stats.api as sms
    import statsmodels.formula.api as smf
except:
    os.system("pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy==1.2.1 --upgrade")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    import statsmodels.stats.api as sms
    import statsmodels.formula.api as smf

import seaborn as sns
    
from scipy import sparse
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
from scipy.stats import ks_2samp
from scipy.stats import beta, norm
from scipy.optimize import leastsq

import time
import datetime
from patsy import dmatrices
import math
import bisect
import hashlib
from itertools import combinations
from collections import defaultdict, Counter
import operator
from operator import itemgetter
import re
import pickle
import json
import sys
import traceback
import warnings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings("ignore")
