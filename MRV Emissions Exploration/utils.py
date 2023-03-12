import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import pandas as pd
import re
from scipy.cluster.hierarchy import dendrogram
from scipy import stats
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import unittest
import warnings


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


# take log2 of pandas frame
def log2_fun(frame_lin):
    # ignore warning about zeros as this is handled
    with np.errstate(divide="ignore"):
        # take logarithm
        frame_log = np.log2(frame_lin)

        # replace 0's with value 2x smaller than smallest existing numerical value
        min_numeric = frame_log.replace(-np.inf, np.nan).min(numeric_only=True) - 1
        frame_log.replace(-np.inf, min_numeric, inplace=True)

    return frame_log


# decorate axis in log style
def decorate_log_axis(axis, direction):
    if direction == "y":
        sub_axis = axis.yaxis
    elif direction == "x":
        sub_axis = axis.xaxis
    sub_axis.set_minor_locator(ticker.MaxNLocator(integer=True))
    sub_axis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    sub_axis.set_major_formatter(ticker.StrMethodFormatter("$2^{{{x:.0f}}}$"))


# define custom transformer to replace outliers
class OutlierRemoval(BaseEstimator, TransformerMixin):
    # initiailise
    def __init__(self, sd_threshold=3, linear_cols=[], log2_cols=[]):
        # save input(s) to instance
        self.sd_threshold = sd_threshold
        self.linear_cols = linear_cols
        self.log2_cols = log2_cols
        self.all_cols = list(set(self.linear_cols + self.log2_cols))

        return

    # fit nested standard scaler
    def fit(self, X_in, y=None):
        # copy input frame to avoid altering input frame
        X = X_in.copy()

        # fit and save standard scaler
        df = X[self.all_cols]
        self.scaler_lin = StandardScaler().fit(df)
        self.scaler_log = StandardScaler().fit(log2_fun(df))

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
            self.scaler_log.transform(log2_fun(df)),
            index=df.index,
            columns=self.all_cols,
        ).abs()

        # replace outliers with nans
        for name in X.columns:
            if name in self.linear_cols:
                mask = X_scaled_lin[name] > self.sd_threshold
                X.loc[mask, name] = np.nan
            elif name in self.log2_cols:
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
            sd_threshold=1, linear_cols=["b"], log2_cols=["c"]
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


# plot structure of hierarchical model
def dendrogram_from_model(model, **kwargs):
    # create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# extract column (e.g. `col`) from feature name (e.g. `col_value`)
def feat_2_col(feat, col_pot):
    # identify all potential columns
    col = [x for x in col_pot if feat.startswith(x)]

    # sort by longest (most specific) first
    col.sort(key=len, reverse=True)

    return col[0]


# extract feature names from column transformer
def get_feat_names(trans, df):
    # initialise blank lists
    feat_names = []
    col_names = []

    # loop over all sub transformers in the column transformer
    for sub_trans, cols in zip(trans.named_transformers_.values(), trans._columns):
        # manually created pipeline has no method for feature extractions so extract inner vectoriser
        if sub_trans.__class__.__name__ in ["Pipeline"]:
            sub_trans_use = sub_trans["vect"]
        else:
            sub_trans_use = sub_trans

        # check class is one that that can be handled by this function
        class_name = sub_trans_use.__class__.__name__
        assert class_name in ["OneHotEncoder", "StandardScaler", "OrdinalEncoder"]

        # extract feature and column names
        if class_name == "OneHotEncoder":
            feat_names_temp = sub_trans_use.get_feature_names_out(cols)
            feat_names.extend(feat_names_temp)
            col_names_temp = [feat_2_col(feat, cols) for feat in feat_names_temp]
            col_names.extend(col_names_temp)
        elif class_name in ["StandardScaler", "OrdinalEncoder"]:
            feat_names.extend(cols)
            col_names.extend(cols)

    # add unprocessed columns to lists
    passthrough_cols = [x for x in df.columns if x not in col_names]
    feat_names.extend(passthrough_cols)
    col_names.extend(passthrough_cols)

    return feat_names, col_names


# plot the accuracy and standard deviation for given subgroup
def grouped_df(ax, df, group):
    # calculate mean test score
    df = df[["mean_test_score", "mean_train_score"] + [group]].rename(
        columns={"mean_test_score": "Test", "mean_train_score": "Train"}
    )
    df = df.groupby(group, dropna=False).min()
    if df.index.dtype == float:
        df.index = df.index.map(lambda x: f"{x:.2f}")
    else:
        df.index = df.index.map(lambda x: str(x))

    # create bar plot
    sns.barplot(
        data=df.unstack().reset_index(),
        x=group,
        y=0,
        hue="level_0",
        ax=ax,
    )

    # decorate axis
    ax.legend(loc="center left", title=None)
    ax.set_ylabel("Mean Squared error")
    ax.set_title(group[6:])

    # annotate bar plot
    for rect_sub in ax.containers:
        for rect in rect_sub:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


# drop unwanted columns from cv search results
def drop_columns(df):
    # identify columns to drop
    drop_cols = [
        col
        for col in df.columns
        if (
            (col.endswith("_time") and col != "mean_fit_time")
            or col.startswith("split")
            or col.startswith("rank_")
        )
    ]
    # drop selected columns
    df.drop(columns=drop_cols, inplace=True)
    return df


# plot results for grouped dataframe
def grouped_plot(df):
    # initialise figure
    param_cols = [col for col in df.columns if col.startswith("param_")]
    no_params = len(param_cols)
    no_cols = min(no_params, 4)
    no_rows = int(np.ceil(no_params / no_cols))
    fig, ax_all = plt.subplots(
        ncols=no_cols, nrows=no_rows, figsize=(18, no_rows * 3), squeeze=False
    )

    # loop through all groups
    for group, ax in zip(param_cols, ax_all.flatten()):
        grouped_df(ax, df, group)

    # define figure layout
    fig.tight_layout()


# calculate confidence interval for model coefficients
def coef_ci(X, res, model, alpha=0.05):
    # combine coefficients and intercept
    coefs = np.insert(model.coef_, 0, model.intercept_)

    # copy input array and add constant
    X_aux = X.copy()
    X_aux.insert(0, "const", 1)

    # t table lookup
    dof = abs(np.diff(X_aux.shape)[0])
    t_val = stats.t.isf(alpha / 2, dof)

    # MSE of residuals
    mse = np.sum(res**2) / dof

    # inverse of the variance of the parameters
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))

    # distance between lower and upper bound of CI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gap = t_val * np.sqrt(mse * var_params)

    return pd.DataFrame(
        {"lower": coefs - gap, "upper": coefs + gap}, index=X_aux.columns
    )
