import pandas as pd
import numpy as np
import hashlib
import random
import string
import sys
import os


# from import_packages import *

class HiddenPrints:
    """
    eg.
    with HiddenPrints():
        print("This will not be printed")

    print("This will be printed as before")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def print_stype_line():
    """
    Create seperation lines
    """
    print('\n' + '#' * 120)
    print('#' * 120 + '\n')


def write_txt(statement, file, encoding="utf-8"):
    """
    Output txt file

    : params recoding_statement: recoding statement to output
    : params file: output file's name
    """
    # Open a file
    fo = open(file, "w", encoding=encoding)
    fo.write(statement)
    # Close opend file
    fo.close()


def md5(s):
    """
    Transform x to its hash value. Must be string
    : params x: string to be transformed
    """
    Str = s.encode('utf-8')
    m = hashlib.md5()
    m.update(Str)
    encodeStr = m.hexdigest()
    return encodeStr


def passwd():
    """
    Generate random password
    """
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    print(salt)
    return salt


def cutid(df, id, file):
    writer = open(file, 'w')
    k = 1
    for line in df[id]:
        line = str(line)
        line = line.strip()
        if k == 1:
            writer.write("\'" + line + "\'")
        else:
            writer.write("," + "\'" + line + "\'")
        k = k + 1
    writer.close()


def default_target(OD_TARGET):
    target = ['d{i}'.format(i=i) for i in OD_TARGET]
    return (target)


def select_uncovered_data(df_label, df_feature, left_on, right_on, key='individual_identity', na=-9999):
    """
    select samples which are not covered by df_feature based on the key(id) provided
    : params df_feature: dataframe containing features to be analyzed
    : params df_label: dataframe containing targets
    : params left_col: column that df_label uses when merging
    : params right_col: column that df_feature uses when merging
    : params key: key to the dataframes,default value is individualidentity
    : params na: value to impute dataframes,default value is -9999
    """
    df_feature = df_feature.fillna(na)
    df_label = df_label.fillna(na)

    df_inner = pd.merge(df_label, df_feature, left_on=left_on, right_on=right_on, how='inner')
    df_all = pd.merge(df_label, df_feature, left_on=left_on, right_on=right_on, how='left')

    idx_inner = df_inner[key].tolist()
    idx_all = df_all[key].tolist()
    idx_outer = set(idx_all) - set(idx_inner)
    uncovered_data = df_all.set_index(key).loc[idx_outer, :]

    return uncovered_data


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################    HYPER PARAMETERS    ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


## day of payment delay
OD_TARGET = [0, 1, 3, 7]

####################################################################################################
## labels： default labels are 'd0', 'd1', 'd3', 'd7'

TARGET = default_target(OD_TARGET)
# TARGET = 'label'
# TARGET = 'is_bad'

####################################################################################################

## default xgb model's parameters
XGB_PARAMS = {}
XGB_PARAMS['objective'] = 'binary:logistic'
XGB_PARAMS['eval_metric'] = 'auc'
XGB_PARAMS['min_child_weight'] = 7
XGB_PARAMS['eta'] = 0.01
XGB_PARAMS['max_depth'] = 2
XGB_PARAMS['subsample'] = 0.5
XGB_PARAMS['colsample_bytree'] = 0.5
XGB_PARAMS['silent'] = 0
XGB_PARAMS['seed'] = 2017
XGB_PARAMS['gamma'] = 0.3
XGB_PARAMS['lambda'] = 20
XGB_PARAMS['alpha'] = 10
XGB_PARAMS['max_delta_step'] = 0
XGB_PARAMS['scale_pos_weight'] = 5
