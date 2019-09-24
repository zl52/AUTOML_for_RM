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
import os

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

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV, f_regression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, make_scorer, f1_score, fbeta_score, precision_score, \
    roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, \
    TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.calibration import CalibratedClassifierCV as CCCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# 4. others

import time
import datetime

import seaborn as sns
import pyecharts

import xgboost as xgb
import xgboost.sklearn
from xgboost import XGBClassifier

from bayes_opt import BayesianOptimization

import missingno as msno
from scipy import sparse
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
from scipy.stats import ks_2samp
from scipy.stats import beta, norm
from scipy.optimize import leastsq
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from patsy import dmatrices
import math
import bisect

import jieba
import jieba.posseg as pseg
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
pd.set_option('max_rows', 500)
pd.set_option('max_columns', 200)
