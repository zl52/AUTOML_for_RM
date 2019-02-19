import pandas as pd;
import numpy as np

from sklearn.model_selection import train_test_split


####################################################################################################
####################################################################################################
######################################                        ######################################
######################################   4. SAMPLE SPLITTER   ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def SAMPLE_SPLITTER(df, label, dt_col=None, val_size=0.2, oot_size=0.2, split_by_ratio=True,
                    method='oot', drop_dt_col=False, random_state=2019):
    """
    Split samples into two or three sets: train, validation (and oot(test)) sets.
    oot(test) set contains lastest samples

    : param df: the dataframe of samples to be splitted
    : param label: boolean label
    : param dt_col: column indicating datetime
    : param val_size: size of validation set (by ratio)
    : param oot_size: size of oot(test) set
    : param split_by_ratio: whether to split oot(test) set by defined ratio
    : param method: if 'oot' is chosen, outputs will be three sets: train, validation and oot(test) sets
                    if 'random' is chosen, outputs will be two sets: train and validation sets
    : param drop_dt_col: whether to drop dt_col
    : param random_state: seed

    : return: train set, train label, validation set, validation label (oot(test) set, oot(test) label)
    """
    if method == 'oot':
        if dt_col is None:
            raise ValueError("Valid dt_col indicating datetime is needed")

        df = df.sort_values(dt_col, ascending=False)

        if drop_dt_col:
            del df[dt_col]

        if split_by_ratio:
            test = df.iloc[:int(df.shape[0] * oot_size), ]
            tmp = df.iloc[int(df.shape[0] * oot_size):, ]
            train, val = train_test_split(tmp, test_size=val_size, random_state=random_state)

        else:
            test = df.iloc[:oot_size, ]
            tmp = df.iloc[oot_size:, ]
            train, val = train_test_split(tmp, test_size=val_size, random_state=random_state)

        train_y = train.pop(label)
        val_y = val.pop(label)
        test_y = test.pop(label)
        print('Average of real values for train set:', '{:.1f}%'.format(train_y.mean() * 100))
        print('Average of real values for validation set:', '{:.1f}%'.format(val_y.mean() * 100))
        print('Average of real values for test set:', '{:.1f}%'.format(test_y.mean() * 100))

        return train, train_y, val, val_y, test, test_y

    elif method == 'random':
        train, val = train_test_split(df, test_size=val_size, random_state=random_state)
        train_y = train.pop(label)
        val_y = val.pop(label)

        print('Average of real values for train set:', '{:.1f}%'.format(train_y.mean() * 100))
        print('Average of real values for validation set:', '{:.1f}%'.format(val_y.mean() * 100))

        return train, train_y, val, val_y

    else:
        raise Exception('method {method} is not defined.'.format(method=method),
                        ' Must chosen between \'random\' and \'oot\'')
