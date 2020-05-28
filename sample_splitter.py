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


def sample_splitter(df, label=None, dt_col=None, val_size=0.2, test_size=0.2, split_by_ratio=True,
                    method='oot', dt_format='%Y-%m-%d', drop_dt_col=False, random_state=2019):
    """
    Split samples into sets: train, test (validation sets).
    if 'oot' is chosen, test set contains lastest samples.

    Parameters
    ----------
    df: pd.DataFrame
			The dataframe of samples to be splitted
    label: str
			Column of boolean label
    dt_col: str
			Column of datetime
    val_size: float
			Size of validation set (ratio or number)
    test_size: float or int
			Size of oot(test) set (ratio or number)
    split_by_ratio: boolean
			If True, split oot(test) set by defined ratio
    method: str
			If 'oot' is chosen, test set is derived based on datetime.
            If 'random' is chosen, test set is splitted randomly.
    drop_dt_col: boolean
			If drop_dt_col is True, drop dt_col
    random_state: int
			Random seed

    Returns
    ----------
		    Return datasets(pd.DataFrame and pd.Series) according to val_size, test_size and method
    """
    if method == 'oot':
        if dt_col is None:
            raise ValueError("Valid \'dt_col\' refering to datetime is needed when \'oot\' is chosen")
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], format=dt_format)
        except:
            pass
        df = df.sort_values(dt_col, ascending=False)
        if drop_dt_col:
            del df[dt_col]

        if split_by_ratio:
            test = df.iloc[:int(df.shape[0] * test_size), ]
            train = df.iloc[int(df.shape[0] * test_size):, ]
        else:
            test = df.iloc[:test_size, ]
            train = df.iloc[test_size:, ]
        print('train set ranges from %s to %s'%(train[dt_col].min(), train[dt_col].max()))
        print('test set ranges from %s to %s\n'%(test[dt_col].min(), test[dt_col].max()))
    elif method == 'random':
        if drop_dt_col:
            del df[dt_col]
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        raise ValueError('Method %s is not defined. Must be chosen between \'random\' and \'oot\''%method)

    if val_size!=0:
        train, val = train_test_split(train, test_size=val_size/(1-test_size), random_state=random_state)
        if label is not None:
            train_y = train.pop(label)
            test_y = test.pop(label)
            val_y = val.pop(label)
            print('Average of real values for train set:', '{:.1f}%'.format(train_y.mean() * 100))
            print('Average of real values for validation set:', '{:.1f}%'.format(val_y.mean() * 100))
            print('Average of real values for test set:', '{:.1f}%'.format(test_y.mean() * 100))
            return train, train_y, val, val_y, test, test_y
        else:
            return train, val, test
    else:
        if label is not None:
            train_y = train.pop(label)
            test_y = test.pop(label)
            print('Average of real values for train set:', '{:.1f}%'.format(train_y.mean() * 100))
            print('Average of real values for test set:', '{:.1f}%'.format(test_y.mean() * 100))
            return train, train_y, test, test_y
        else:
            return train, test
