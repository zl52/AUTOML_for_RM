import pandas as pd
import numpy as np

from tools import *


class generate_recoding_statement:

    def __init__(self, df, label, target=TARGET, exclude_list=[]):
        """
        : params df: the input dataframe
        : param label: label will be used in the modeling process
        : params target: list of default boolean targets
        : param exclude_list: list of features excluded from being recoded
        : params col: features used to train the model
        : params recoding_dict: dictionary recording recoding statements of features
        : params feat_dict: dictionary recording changes of feature names
        : param recoding_statement: recoding statement to output
        """
        self.target = target
        self.label = label
        self.exclude_list = exclude_list
        self.col = set(df.columns) - set(self.exclude_list + [self.label] + self.target)
        self.recoding_dict = dict(zip(self.col, [''] * len(self.col)))
        self.feat_dict = dict(zip(self.col, self.col))
        self.recoding_statement = ''

    def write_recoding_txt(self, file, encoding="utf-8"):
        """
        Output recoding statements

        : params file: out file's name
        : param encoding: encoded standard
        """
        for k, v in self.recoding_dict.items():
            self.recoding_statement += '\n' + '#' * 40 + ' Recoding Statement for ' \
                                       + str(k) + ' ' + '#' * 40 + '\n' + v + '\n'

        self.file = file
        fo = open(self.file, "w", encoding=encoding)
        fo.write(self.recoding_statement)
        fo.close()

    def exec_recoding(self, df, encoding="utf-8"):
        """
        Execute recoding statement to the input dataframe

        : params df: the input dataframe
        : param encoding: encoded standard

        : return df_copy: recoded dataframe
        """
        df_copy = df.copy()
        fo = open(self.file, 'r', encoding=encoding)
        recoding_text = fo.read()
        fo.close()
        exec(recoding_text)

        return df_copy
