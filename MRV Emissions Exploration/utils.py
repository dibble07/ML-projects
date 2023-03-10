import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import unittest


# define function to extract type and value of technical efficiency
def tech_eff_extract(x):
    # extract numerical value
    match_obj = re.search("([0-9]*\.?[0-9]*E?\+?[0-9]*) gCO₂/t·nm", x)
    value = np.nan if match_obj is None else match_obj.group(1)

    # delete numerical value and surrounding characters
    type_ = re.sub("([0-9]*\.?[0-9]*E?\+?[0-9]* gCO₂/t·nm)", "", x).replace(" ()", "")
    if type_ == "":
        type_ = "nan"

    return type_, value


# take log10 of pandas frame
def log10_fun(frame_lin):
    # ignore warning about zeros as this is handled
    with np.errstate(divide="ignore"):
        # take logarithm
        frame_log = np.log10(frame_lin)

        # replace 0's with value 10x smaller than smallest existing numerical value
        min_numeric = frame_log.replace(-np.inf, np.nan).min(numeric_only=True) - 1
        frame_log.replace(-np.inf, min_numeric, inplace=True)

    return frame_log


# define custom transformer to replace outliers
class OutlierRemoval(BaseEstimator, TransformerMixin):
    # initiailise
    def __init__(self, sd_threshold=3, linear_cols=[], log10_cols=[]):
        # save input(s) to instance
        self.sd_threshold = sd_threshold
        self.linear_cols = linear_cols
        self.log10_cols = log10_cols
        self.all_cols = list(set(self.linear_cols + self.log10_cols))

        return

    # fit nested standard scaler
    def fit(self, X_in, y=None):
        # copy input frame to avoid altering input frame
        X = X_in.copy()

        # fit and save standard scaler
        df = X[self.all_cols]
        self.scaler_lin = StandardScaler().fit(df)
        self.scaler_log = StandardScaler().fit(log10_fun(df))

        return self

    def transform(self, X_in, y=None):
        # copy input frame to avoid altering input frame
        X = X_in.copy()

        # check input is pandas object
        assert isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)

        # calculate absolute standard deviations from mean for each column
        df = X[self.all_cols]
        X_scaled_lin = pd.DataFrame(
            self.scaler_lin.transform(df), index=df.index, columns=self.all_cols
        ).abs()
        X_scaled_log = pd.DataFrame(
            self.scaler_log.transform(log10_fun(df)),
            index=df.index,
            columns=self.all_cols,
        ).abs()

        # replace outliers with nans
        for name in X.columns:
            if name in self.linear_cols:
                mask = X_scaled_lin[name] > self.sd_threshold
                X.loc[mask, name] = np.nan
            elif name in self.log10_cols:
                mask = X_scaled_log[name] > self.sd_threshold
                X.loc[mask, name] = np.nan

        return X


# unit test for transformers
class test_transformer(unittest.TestCase):
    # test outlier removal
    def test_outlier_removal(self):
        # calculate values
        X_in = pd.DataFrame(
            {
                "a": [1.0, 5.0, 1.0, 0.0, 0.0, 0.0],
                "b": [1.0, 1.0, 1.0, 0.0, 5.0, 0.0],
                "c": [10, 100000, 10, 0.0, 0.0, 0.0],
            }
        )
        X_out = OutlierRemoval(
            sd_threshold=1, linear_cols=["b"], log10_cols=["c"]
        ).fit_transform(X_in)
        X_out_correct = pd.DataFrame(
            {
                "a": [1.0, 5.0, 1.0, 0.0, 0.0, 0.0],
                "b": [1.0, 1.0, 1.0, 0.0, np.nan, 0.0],
                "c": [10, np.nan, 10, 0.0, 0.0, 0.0],
            }
        )

        # assert on values
        self.assertTrue(X_out.equals(X_out_correct))
